"""
Entailment-Based Provenance Analysis for MuSiQue Multi-Hop QA

Decomposes aggregate model answers into atomic claims, traces each claim's
provenance (parametric vs. search-retrieved vs. hallucinated), and computes
grounding scores per 2x2 outcome cell (all_hops_correct x aggregate_correct).

Outputs per model:
  - {model}_claim_level.csv  — one row per claim
  - {model}_question_level.csv — one row per question with provenance fractions

Usage:
  uv run python scripts/analyze_entailment_provenance.py \
    --results_dir results/musique \
    --output_dir results/musique/analysis \
    --judge_model gemini-3-flash-preview \
    --judge_provider Google
"""

import os
import sys
import json
import csv
import glob
import argparse
from typing import Optional, Literal
from dataclasses import dataclass, asdict

import numpy as np
from pydantic import BaseModel
from tqdm import tqdm
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.services.base_agent import BaseAgent

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini-3-flash-preview")
DEFAULT_JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "Google")
SIM_THRESHOLD = 0.4          # min cosine sim to assign claim to a hop
BRIDGING_THRESHOLD = 0.4     # sim threshold for "bridging" classification
SEARCH_TEXT_LIMIT = 4000     # max chars of search snippets for NLI prompt

PROVENANCE_LABELS = [
    "reinforced",
    "parametric_only",
    "search_rescued",
    "unsupported",
    "conflict_pk_chosen",
    "conflict_search_chosen",
]

# ── Pydantic output types ──────────────────────────────────────────────────────

class ClaimsOutput(BaseModel):
    claims: list[str]


class NLIOutput(BaseModel):
    label: Literal["SUPPORT", "CONTRADICT", "NEUTRAL"]
    reasoning: str


# ── LLM judge prompts ─────────────────────────────────────────────────────────

CLAIMS_PROMPT = """\
Given the following answer+explanation to a multi-hop question, decompose it into
individual atomic factual claims. Each claim must be a single, self-contained
assertion that could be independently verified. Aim for one claim per reasoning step.

Question: {question}
Answer + Explanation: {answer_text}

Output a JSON list of claim strings."""

NLI_PROMPT = """\
Given the search results below and a factual claim, determine whether the search
results SUPPORT, CONTRADICT, or are NEUTRAL regarding the claim.

Search Results:
{search_results_text}

Claim: {claim}

Answer with SUPPORT, CONTRADICT, or NEUTRAL, and one sentence of reasoning."""

# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ClaimRecord:
    example_id: str
    hop_0_correct: bool
    hop_1_correct: bool
    hop_2_correct: bool
    hop_3_correct: bool
    all_hops_correct: bool
    aggregate_correct: bool
    cell: str
    claim_idx: int
    claim_text: str
    assigned_hop: Optional[int]
    hop_similarity: float
    is_bridging: bool
    search_entailment: str
    pk_available: bool
    provenance: str


@dataclass
class QuestionRecord:
    example_id: str
    all_hops_correct: bool
    aggregate_correct: bool
    cell: str
    n_claims: int
    reinforced_frac: float
    parametric_only_frac: float
    search_rescued_frac: float
    unsupported_frac: float
    conflict_pk_chosen_frac: float
    conflict_search_chosen_frac: float
    grounded_frac: float
    n_bridge_claims: int
    bridge_unsupported_frac: float


# ── Helpers ───────────────────────────────────────────────────────────────────

def cell_label(all_hops: bool, agg: bool) -> str:
    """2×2 cell: A=both correct, B=hops yes/agg no, C=hops no/agg yes, D=both wrong."""
    if all_hops and agg:
        return "A"
    if all_hops and not agg:
        return "B"
    if not all_hops and agg:
        return "C"
    return "D"


def classify_provenance(search_label: str, pk_available: bool) -> str:
    if search_label == "SUPPORT" and pk_available:
        return "reinforced"
    if search_label == "SUPPORT" and not pk_available:
        return "search_rescued"
    if search_label == "NEUTRAL" and pk_available:
        return "parametric_only"
    if search_label == "NEUTRAL" and not pk_available:
        return "unsupported"
    if search_label == "CONTRADICT" and pk_available:
        return "conflict_pk_chosen"
    # CONTRADICT + not pk_available
    return "conflict_search_chosen"


def collect_search_snippets(trace: dict) -> str:
    """Collect ALL search tool_call_response snippets from a trace, joined and truncated."""
    snippets = []
    for msg in trace.get("message_trace", []):
        for part in msg.get("parts", []):
            if part.get("type") == "tool_call_response":
                result = part.get("result", [])
                if isinstance(result, list):
                    for r in result:
                        s = r.get("snippet", "")
                        if s:
                            snippets.append(str(s))
                elif isinstance(result, str) and result.strip():
                    snippets.append(result)
    combined = "\n\n".join(snippets)
    return combined[:SEARCH_TEXT_LIMIT]


def build_traces_map(traces: list[dict]) -> dict[str, str]:
    """Map aggregate_question_prefix (80 chars) → search_snippets string."""
    result = {}
    for t in traces:
        problem = t.get("problem", "")
        key = problem[:80]
        if key not in result:
            result[key] = collect_search_snippets(t)
        else:
            # merge snippets from multiple traces with same question prefix
            existing = result[key]
            new = collect_search_snippets(t)
            combined = (existing + "\n\n" + new)[:SEARCH_TEXT_LIMIT]
            result[key] = combined
    return result


def build_no_search_map(no_search_data: list[dict]) -> dict[str, dict[int, bool]]:
    """Map example_id → {hop_index: is_correct}."""
    result = {}
    for ex in no_search_data:
        eid = ex["example_id"]
        hop_map = {}
        for sq in ex.get("sub_questions_results", []):
            hi = sq.get("hop_index", 0)
            hop_map[hi] = bool(sq.get("is_correct", False))
        result[eid] = hop_map
    return result


# ── Main analyzer class ───────────────────────────────────────────────────────

class EntailmentProvenanceAnalyzer:
    def __init__(
        self,
        results_dir: str,
        output_dir: str,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        judge_provider: str = DEFAULT_JUDGE_PROVIDER,
    ):
        self.results_dir = results_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        print(f"Initializing LLM judges ({judge_provider}/{judge_model}) ...")
        self.claim_extractor = BaseAgent(
            model_name=judge_model,
            provider_name=judge_provider,
            output_type=ClaimsOutput,
            agent_name="ClaimExtractor",
        )
        self.nli_judge = BaseAgent(
            model_name=judge_model,
            provider_name=judge_provider,
            output_type=NLIOutput,
            agent_name="NLIJudge",
        )

        print("Loading SentenceTransformer (intfloat/multilingual-e5-large) ...")
        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer("intfloat/multilingual-e5-large")

    # ── Public entry points ───────────────────────────────────────────────────

    def run_all_models(self) -> None:
        model_dirs = [
            d for d in sorted(os.listdir(self.results_dir))
            if os.path.isdir(os.path.join(self.results_dir, d))
            and d != "analysis"
        ]
        print(f"Found model directories: {model_dirs}")
        for model_name in model_dirs:
            self.run_model(os.path.join(self.results_dir, model_name), model_name)

    def run_model(self, model_dir: str, model_name: str) -> None:
        print(f"\n{'='*60}")
        print(f"Processing model: {model_name}")
        print(f"  Directory: {model_dir}")

        # Locate files
        with_search_files = glob.glob(os.path.join(model_dir, "*_with_search.json"))
        no_search_files = glob.glob(os.path.join(model_dir, "*_no_search.json"))
        traces_files = glob.glob(os.path.join(model_dir, "*_traces_*.json"))

        if not with_search_files:
            print(f"  No with_search.json found, skipping.")
            return
        if not no_search_files:
            print(f"  No no_search.json found, skipping.")
            return

        with_search_path = with_search_files[0]
        no_search_path = no_search_files[0]

        print(f"  with_search: {os.path.basename(with_search_path)}")
        print(f"  no_search:   {os.path.basename(no_search_path)}")

        with open(with_search_path) as f:
            with_search_data = json.load(f)
        with open(no_search_path) as f:
            no_search_data = json.load(f)

        no_search_map = build_no_search_map(no_search_data)

        # Load and merge all traces files
        all_traces = []
        for tf in sorted(traces_files):
            print(f"  traces: {os.path.basename(tf)}")
            with open(tf) as f:
                all_traces.extend(json.load(f))
        traces_map = build_traces_map(all_traces)

        # Output paths
        safe_name = model_name.replace("/", "_").replace(":", "_")
        claim_path = os.path.join(self.output_dir, f"{safe_name}_claim_level.csv")
        question_path = os.path.join(self.output_dir, f"{safe_name}_question_level.csv")

        # Resume support
        processed_ids = self._load_processed_ids(question_path)
        print(f"  Already processed: {len(processed_ids)} examples")

        claim_fields = [f.name for f in ClaimRecord.__dataclass_fields__.values()]
        question_fields = [f.name for f in QuestionRecord.__dataclass_fields__.values()]

        claim_mode = "a" if os.path.exists(claim_path) else "w"
        question_mode = "a" if os.path.exists(question_path) else "w"

        with (
            open(claim_path, claim_mode, newline="", encoding="utf-8") as cf,
            open(question_path, question_mode, newline="", encoding="utf-8") as qf,
        ):
            claim_writer = csv.DictWriter(cf, fieldnames=claim_fields)
            question_writer = csv.DictWriter(qf, fieldnames=question_fields)
            if claim_mode == "w":
                claim_writer.writeheader()
            if question_mode == "w":
                question_writer.writeheader()

            for ex in tqdm(with_search_data, desc=f"  {model_name}"):
                eid = ex["example_id"]
                if eid in processed_ids:
                    continue

                # Find search snippets via trace match
                agg_q = ex.get("aggregate_question", "")
                search_snippets = traces_map.get(agg_q[:80], "")

                no_search_hops = no_search_map.get(eid, {})

                claim_records, question_record = self.analyze_example(
                    ex, no_search_hops, search_snippets
                )

                for cr in claim_records:
                    claim_writer.writerow(asdict(cr))
                cf.flush()

                question_writer.writerow(asdict(question_record))
                qf.flush()

        print(f"  Wrote: {claim_path}")
        print(f"  Wrote: {question_path}")

    # ── Per-example analysis ──────────────────────────────────────────────────

    def analyze_example(
        self,
        ex: dict,
        no_search_hops: dict[int, bool],
        search_snippets: str,
    ) -> tuple[list[ClaimRecord], QuestionRecord]:

        eid = ex["example_id"]
        agg_q = ex.get("aggregate_question", "")
        agg_response = ex.get("aggregate_result", {}).get("model_response", "")
        all_hops = bool(ex.get("all_factoids_correct", False))
        agg_correct = bool(ex.get("aggregate_correct", False))
        cell = cell_label(all_hops, agg_correct)

        # Per-hop correctness (with_search)
        hop_correct = {0: False, 1: False, 2: False, 3: False}
        for sq in ex.get("sub_questions_results", []):
            hi = sq.get("hop_index", 0)
            hop_correct[hi] = bool(sq.get("is_correct", False))

        # Step 1: extract claims
        claims = self.extract_claims(agg_q, agg_response)

        # Hop text for alignment: "question answer"
        hop_texts = []
        for sq in sorted(ex.get("sub_questions_results", []), key=lambda x: x.get("hop_index", 0)):
            q = sq.get("question", "")
            a = sq.get("gold_answer", "")
            hop_texts.append(f"{q} {a}")

        # Step 2: align claims to hops
        assignments, similarities, bridging_flags = self.align_claims_to_hops(claims, hop_texts)

        # Steps 3–5: NLI + provenance per claim
        claim_records = []
        provenance_counts: dict[str, int] = {k: 0 for k in PROVENANCE_LABELS}
        n_bridge = 0
        bridge_unsupported = 0

        for idx, (claim, hop_idx, sim, is_bridge) in enumerate(
            zip(claims, assignments, similarities, bridging_flags)
        ):
            # NLI
            if search_snippets:
                nli_label = self.get_search_entailment(claim, search_snippets)
            else:
                nli_label = "NEUTRAL"

            # Parametric proxy
            if is_bridge:
                # Need ALL bridged hops to be correct in no-search
                # Find hops with sim >= threshold
                if hop_texts:
                    hop_sims = self._cosine_sims_one_to_many(
                        "query: " + claim,
                        ["passage: " + h for h in hop_texts],
                    )
                    bridged_hops = [i for i, s in enumerate(hop_sims) if s >= BRIDGING_THRESHOLD]
                    pk_available = all(no_search_hops.get(h, False) for h in bridged_hops)
                else:
                    pk_available = False
            elif hop_idx is not None:
                pk_available = no_search_hops.get(hop_idx, False)
            else:
                pk_available = False  # conservative

            prov = classify_provenance(nli_label, pk_available)
            provenance_counts[prov] += 1

            if is_bridge:
                n_bridge += 1
                if prov == "unsupported":
                    bridge_unsupported += 1

            cr = ClaimRecord(
                example_id=eid,
                hop_0_correct=hop_correct[0],
                hop_1_correct=hop_correct[1],
                hop_2_correct=hop_correct[2],
                hop_3_correct=hop_correct[3],
                all_hops_correct=all_hops,
                aggregate_correct=agg_correct,
                cell=cell,
                claim_idx=idx,
                claim_text=claim,
                assigned_hop=hop_idx,
                hop_similarity=round(sim, 4),
                is_bridging=is_bridge,
                search_entailment=nli_label,
                pk_available=pk_available,
                provenance=prov,
            )
            claim_records.append(cr)

        # Build question record
        n = len(claim_records)
        if n > 0:
            fracs = {k: provenance_counts[k] / n for k in PROVENANCE_LABELS}
            grounded = (
                provenance_counts["reinforced"]
                + provenance_counts["parametric_only"]
                + provenance_counts["search_rescued"]
            ) / n
            bridge_unsupported_frac = (bridge_unsupported / n_bridge) if n_bridge > 0 else 0.0
        else:
            fracs = {k: 0.0 for k in PROVENANCE_LABELS}
            grounded = 0.0
            bridge_unsupported_frac = 0.0

        qr = QuestionRecord(
            example_id=eid,
            all_hops_correct=all_hops,
            aggregate_correct=agg_correct,
            cell=cell,
            n_claims=n,
            reinforced_frac=round(fracs["reinforced"], 4),
            parametric_only_frac=round(fracs["parametric_only"], 4),
            search_rescued_frac=round(fracs["search_rescued"], 4),
            unsupported_frac=round(fracs["unsupported"], 4),
            conflict_pk_chosen_frac=round(fracs["conflict_pk_chosen"], 4),
            conflict_search_chosen_frac=round(fracs["conflict_search_chosen"], 4),
            grounded_frac=round(grounded, 4),
            n_bridge_claims=n_bridge,
            bridge_unsupported_frac=round(bridge_unsupported_frac, 4),
        )

        return claim_records, qr

    # ── LLM calls ─────────────────────────────────────────────────────────────

    def extract_claims(self, question: str, answer_text: str) -> list[str]:
        if not answer_text or not answer_text.strip():
            return []
        prompt = CLAIMS_PROMPT.format(question=question, answer_text=answer_text)
        try:
            res = self.claim_extractor.run(prompt)
            claims = res.output.claims
            return [c.strip() for c in claims if c.strip()]
        except Exception as e:
            print(f"    ClaimExtractor error: {e}")
            return []

    def get_search_entailment(self, claim: str, search_text: str) -> str:
        prompt = NLI_PROMPT.format(search_results_text=search_text, claim=claim)
        try:
            res = self.nli_judge.run(prompt)
            return res.output.label
        except Exception as e:
            print(f"    NLI judge error: {e}")
            return "NEUTRAL"

    # ── Semantic alignment ────────────────────────────────────────────────────

    def align_claims_to_hops(
        self,
        claims: list[str],
        hop_texts: list[str],
    ) -> tuple[list[Optional[int]], list[float], list[bool]]:
        """
        Returns (assigned_hop_indices, max_similarities, is_bridging) per claim.
        hop_texts are the gold sub-question+answer texts.
        """
        if not claims or not hop_texts:
            return [None] * len(claims), [0.0] * len(claims), [False] * len(claims)

        claim_vecs = self.encoder.encode(
            ["query: " + c for c in claims], normalize_embeddings=True
        )
        hop_vecs = self.encoder.encode(
            ["passage: " + h for h in hop_texts], normalize_embeddings=True
        )
        # sim matrix: (n_claims, n_hops)
        sim_matrix = claim_vecs @ hop_vecs.T

        assignments: list[Optional[int]] = []
        max_sims: list[float] = []
        bridging: list[bool] = []

        for row in sim_matrix:
            best_idx = int(np.argmax(row))
            best_sim = float(row[best_idx])
            above = [i for i, s in enumerate(row) if s >= BRIDGING_THRESHOLD]
            is_bridge = len(above) >= 2
            if best_sim < SIM_THRESHOLD:
                assignments.append(None)
            else:
                assignments.append(best_idx)
            max_sims.append(best_sim)
            bridging.append(is_bridge)

        return assignments, max_sims, bridging

    def _cosine_sims_one_to_many(self, query: str, passages: list[str]) -> list[float]:
        """Encode one query vs many passages, return cosine similarities."""
        q_vec = self.encoder.encode([query], normalize_embeddings=True)
        p_vecs = self.encoder.encode(passages, normalize_embeddings=True)
        sims = (q_vec @ p_vecs.T).flatten().tolist()
        return sims

    # ── Resume support ────────────────────────────────────────────────────────

    @staticmethod
    def _load_processed_ids(question_path: str) -> set[str]:
        if not os.path.exists(question_path):
            return set()
        processed = set()
        with open(question_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and "example_id" in reader.fieldnames:
                for row in reader:
                    processed.add(row["example_id"])
        return processed


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Entailment-based provenance analysis for MuSiQue results."
    )
    parser.add_argument(
        "--results_dir",
        default="results/musique",
        help="Root directory containing per-model subdirectories.",
    )
    parser.add_argument(
        "--output_dir",
        default="results/musique/analysis",
        help="Directory for output CSVs.",
    )
    parser.add_argument(
        "--judge_model",
        default=DEFAULT_JUDGE_MODEL,
        help="LLM judge model name.",
    )
    parser.add_argument(
        "--judge_provider",
        default=DEFAULT_JUDGE_PROVIDER,
        help="LLM judge provider (Google, OpenAI, Anthropic, ollama).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Process a single model subdirectory name (e.g. gemini-3-pro).",
    )
    args = parser.parse_args()

    analyzer = EntailmentProvenanceAnalyzer(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        judge_model=args.judge_model,
        judge_provider=args.judge_provider,
    )

    if args.model:
        model_dir = os.path.join(args.results_dir, args.model)
        if not os.path.isdir(model_dir):
            print(f"ERROR: Model directory not found: {model_dir}")
            sys.exit(1)
        analyzer.run_model(model_dir, args.model)
    else:
        analyzer.run_all_models()


if __name__ == "__main__":
    main()
