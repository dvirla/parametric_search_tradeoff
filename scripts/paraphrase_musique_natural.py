"""
Paraphrase MusiQue benchmark questions into natural, goal-oriented user questions.

The benchmark format exposes the hop chain explicitly:
  "What administrative territorial entity includes the place where Bill Cockcroft was educated?"

A natural user would ask the same information need without revealing the retrieval strategy:
  "I'm curious about Bill Cockcroft's background — what region of the country did he study in?"

Two input modes (mutually exclusive):
  --staleness-csv   Load directly from data/musique_train_staleness.csv + HuggingFace dataset.
                    Can run in parallel with run_musique_parametric_uncertainty.py.
  --source          Load from a completed run_musique_parametric_uncertainty.py JSON output.
                    Backward-compatible default.

Output JSONL schema matches sharechat_info_seeking_v3.jsonl plus an `answer` field.

Usage:
  # Parallel mode (no dependency on uncertainty script):
  uv run python scripts/paraphrase_musique_natural.py \\
      --staleness-csv data/musique_train_staleness.csv \\
      --output data/musique_train_natural.jsonl \\
      --all-hops

  # Legacy mode (reads uncertainty JSON):
  uv run python scripts/paraphrase_musique_natural.py \\
      --source results/musique_parametric_train/musique_parametric_uncertainty_<model>.json \\
      --output data/musique_train_natural.jsonl \\
      --all-hops
"""

import argparse
import asyncio
import csv
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.services.base_agent import BaseAgent
from scripts.archive.run_musique_experiment import resolve_subquestion_text

_DEFAULT_SOURCE = "results/musique_parametric/musique_parametric_uncertainty_gemini-3-pro-preview.json"
_DEFAULT_OUTPUT = "data/musique_natural.jsonl"

SYSTEM_PROMPT = """\
You are a prompt rewriter. You will be given a multi-hop benchmark question and the \
chain of sub-questions that reveal its structure. Your task is to rewrite the question \
so it sounds like a genuine, curious user asking — not like a database traversal.

Rules:
1. Do NOT mention any intermediate entity that appears only as a bridge in the hop chain. \
The user asking the question would not know that entity exists.
2. Do NOT use the "X of Y where Z did W" pattern. Avoid nested relative clauses that signal \
the hop structure.
3. Frame the question so the answer naturally warrants 2–4 explanatory sentences, not a \
single word or number. Prefer "why", "how", "tell me about", "explain", "what can you tell \
me about" phrasing where it fits naturally. If the answer is inherently factual (a location, \
a date), still frame it as "I'm curious about…" or "Could you tell me a bit about…" to \
invite context.
4. The rewritten question must preserve the same ultimate information need and have the same \
gold answer.
5. Return only the rewritten question text — no explanation, no preamble, no quotes.
"""

PARAPHRASE_TEMPLATE = """\
Original benchmark question: {question}

Hop chain (do NOT expose these steps in the rewrite):
{hops}

Gold answer: {answer}

Rewrite this as a natural user question following the rules:"""


def format_hops(sub_questions: list[dict]) -> str:
    lines = []
    for sq in sub_questions:
        lines.append(f"  Step {sq['hop_index'] + 1}: {sq['question']} → {sq['gold_answer']}")
    return "\n".join(lines)


def load_existing_ids(output_path: str) -> set[str]:
    if not os.path.exists(output_path):
        return set()
    ids = set()
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    ids.add(json.loads(line)["example_id"])
                except (KeyError, json.JSONDecodeError):
                    pass
    return ids


# ---------------------------------------------------------------------------
# Input loaders — both produce the same internal dict format
# ---------------------------------------------------------------------------

def _normalise(example_id, aggregate_question, aggregate_answer, sub_questions_results, is_stale) -> dict:
    """Canonical internal format consumed by paraphrase_one."""
    return {
        "example_id": example_id,
        "aggregate_question": aggregate_question,
        "aggregate_answer": aggregate_answer,
        "sub_questions_results": sub_questions_results,  # list of {hop_index, question, gold_answer}
        "is_stale": is_stale,
    }


def load_from_uncertainty_json(source_file: str, hops_filter: int | None) -> list[dict]:
    """Load from a completed run_musique_parametric_uncertainty.py output."""
    with open(source_file) as f:
        raw = json.load(f)
    data = []
    for item in raw:
        subs = item.get("sub_questions_results", [])
        if hops_filter is not None and len(subs) != hops_filter:
            continue
        data.append(_normalise(
            example_id=item["example_id"],
            aggregate_question=item["aggregate_question"],
            aggregate_answer=item["aggregate_answer"],
            sub_questions_results=[
                {"hop_index": s["hop_index"], "question": s["question"], "gold_answer": s["gold_answer"]}
                for s in subs
            ],
            is_stale=item.get("is_stale", False),
        ))
    return data


def load_from_staleness_csv(staleness_csv: str, hops_filter: int | None) -> list[dict]:
    """Load directly from the staleness CSV + HuggingFace dataset.

    Resolves sub-question placeholders (#1, #2 …) so the hop chain is in the
    same resolved format as the uncertainty-JSON mode.
    """
    # Read non-stale IDs from CSV
    non_stale_ids: set[str] = set()
    is_stale_map: dict[str, bool] = {}
    with open(staleness_csv, newline="") as f:
        for row in csv.DictReader(f):
            eid = row["example_id"]
            stale = row.get("is_stale", "") == "True"
            is_stale_map[eid] = stale
            if not stale:
                non_stale_ids.add(eid)

    print(f"Staleness CSV: {len(is_stale_map)} total, {len(non_stale_ids)} non-stale")

    from datasets import load_dataset
    print("Loading MuSiQue from HuggingFace...")
    ds = load_dataset("dgslibisey/MuSiQue")

    data = []
    for split in ["train", "validation"]:
        if split not in ds:
            continue
        for example in ds[split]:
            eid = example["id"]
            if eid not in non_stale_ids:
                continue
            if not example.get("answerable", True):
                continue

            decomp = example["question_decomposition"]
            n_hops = len(decomp)
            if hops_filter is not None and n_hops != hops_filter:
                continue

            sub_questions_results = [
                {
                    "hop_index": hop_idx,
                    "question": resolve_subquestion_text(decomp, hop_idx),
                    "gold_answer": decomp[hop_idx]["answer"],
                }
                for hop_idx in range(n_hops)
            ]
            data.append(_normalise(
                example_id=eid,
                aggregate_question=example["question"],
                aggregate_answer=example["answer"],
                sub_questions_results=sub_questions_results,
                is_stale=is_stale_map.get(eid, False),
            ))

    print(f"Loaded {len(data)} answerable non-stale examples from HuggingFace")
    return data


# ---------------------------------------------------------------------------
# Paraphrasing
# ---------------------------------------------------------------------------

async def paraphrase_one(agent: BaseAgent, item: dict, feedback: str = "") -> str:
    prompt = PARAPHRASE_TEMPLATE.format(
        question=item["aggregate_question"],
        hops=format_hops(item["sub_questions_results"]),
        answer=item["aggregate_answer"],
    )
    if feedback:
        prompt += ("\n\nThe previous attempt was rejected by an auditor. Fix it:\n"
                   f"{feedback}\nReturn only the corrected rewritten question.")
    response = await agent.arun(prompt)
    return response.output.strip().strip('"').strip("'")


async def paraphrase_validated(agent: BaseAgent, judge_agent, item: dict,
                               max_retries: int) -> tuple[str, dict]:
    """Generate→judge→regenerate. Returns (best_text, validation_metadata).

    Keeps the first paraphrase that passes the leak + equivalence audit. If none
    passes within max_retries+1 attempts, returns the last attempt flagged failed
    so downstream can drop it (the example_id is preserved for pairing).
    """
    from src.services.paraphrase_validation import audit_paraphrase, audit_feedback

    subs = item["sub_questions_results"]
    num_hops = len(subs)
    feedback, last_text, last_audit = "", None, None
    for attempt in range(max_retries + 1):
        last_text = await paraphrase_one(agent, item, feedback)
        try:
            audit = await audit_paraphrase(
                judge_agent, item["aggregate_question"], last_text, subs,
                item["aggregate_answer"], num_hops)
        except Exception as e:
            # Judge failed this attempt — retry; if it never succeeds, flag it.
            last_audit = None
            feedback = ""
            print(f"      [audit error attempt {attempt + 1}] {e}")
            continue
        last_audit = audit
        if audit.ok:
            return last_text, {"validation_status": "pass", "validation_attempts": attempt + 1,
                               "leaked_hops": [], "equivalent": True, "drift": "none"}
        feedback = audit_feedback(audit, subs)

    if last_audit is None:
        meta = {"validation_status": "judge_error", "validation_attempts": max_retries + 1,
                "leaked_hops": None, "equivalent": None, "drift": None}
    else:
        meta = {"validation_status": "fail", "validation_attempts": max_retries + 1,
                "leaked_hops": last_audit.leaked_hop_indices,
                "equivalent": last_audit.equivalent, "drift": last_audit.drift}
    return last_text, meta


async def main(
    hops_filter: int | None,
    model_name: str,
    limit: int | None,
    source_file: str | None,
    staleness_csv: str | None,
    output_file: str,
    validate: bool = False,
    validate_model: str = "gpt-oss:120b",
    validate_provider: str = "ollama",
    max_retries: int = 3,
):
    if staleness_csv:
        data = load_from_staleness_csv(staleness_csv, hops_filter)
    else:
        data = load_from_uncertainty_json(source_file, hops_filter)

    if limit is not None:
        data = data[:limit]

    existing_ids = load_existing_ids(output_file)
    pending = [d for d in data if d["example_id"] not in existing_ids]

    print(f"Total questions: {len(data)}")
    print(f"Already processed: {len(existing_ids)}")
    print(f"To process: {len(pending)}")

    if not pending:
        print("Nothing to do.")
        return

    agent = BaseAgent(
        provider_name="ollama",
        model_name=model_name,
        system_prompt=SYSTEM_PROMPT,
    )

    judge_agent = None
    if validate:
        from src.services.paraphrase_validation import make_judge_agent
        print(f"Validation ON: leak+equivalence audit via {validate_provider}/{validate_model}, "
              f"up to {max_retries} regenerations.")
        judge_agent = make_judge_agent(validate_model, validate_provider)

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    stats = {"pass": 0, "fail": 0, "judge_error": 0, "unvalidated": 0}
    with open(output_file, "a") as out_f:
        for i, item in enumerate(pending):
            try:
                if validate:
                    paraphrased, vmeta = await paraphrase_validated(agent, judge_agent, item, max_retries)
                else:
                    paraphrased = await paraphrase_one(agent, item)
                    vmeta = {"validation_status": "unvalidated"}
            except Exception as e:
                print(f"  [!] Failed on {item['example_id']}: {e}")
                continue

            stats[vmeta["validation_status"]] = stats.get(vmeta["validation_status"], 0) + 1
            record = {
                "text": paraphrased,
                "answer": item["aggregate_answer"],
                "example_id": item["example_id"],
                "original_question": item["aggregate_question"],
                "reasoning_hops": len(item["sub_questions_results"]),
                "sub_questions": [
                    {"question": sq["question"], "answer": sq["gold_answer"]}
                    for sq in item["sub_questions_results"]
                ],
                "is_info_seeking": True,
                "is_time_dependent": item.get("is_stale", False),
                "category": "factual_lookup",
                "source": "musique_natural",
                **vmeta,
            }

            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

            tag = vmeta["validation_status"]
            print(f"[{i + 1}/{len(pending)}] ({tag}) {item['aggregate_question'][:55]}")
            print(f"        → {paraphrased[:80]}")

    print(f"\nDone. Output: {output_file}")
    if validate:
        print(f"Validation summary: {stats}")
        print("Note: 'fail'/'judge_error' rows are kept (flagged) so the example_id set "
              "stays paired; filter on validation_status downstream.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paraphrase MusiQue questions into natural user questions")
    parser.add_argument("--hops", type=int, default=2, help="Filter to N-hop questions (default: 2)")
    parser.add_argument("--all-hops", action="store_true", help="Include all hop counts (overrides --hops)")
    parser.add_argument("--model", default="gpt-oss:20b", help="Ollama model name (default: gpt-oss:20b)")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N questions")
    # Input source — mutually exclusive
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--staleness-csv", default=None,
                     help="Run independently: load from staleness CSV + HuggingFace (no uncertainty JSON needed)")
    src.add_argument("--source", default=_DEFAULT_SOURCE,
                     help="Load from uncertainty JSON (default: gemini val run)")
    parser.add_argument("--output", default=_DEFAULT_OUTPUT, help="Output JSONL path")
    # Generate→judge→regenerate validation loop (leak + equivalence)
    parser.add_argument("--validate", action="store_true",
                        help="Audit each rewrite for hop-leak + equivalence and regenerate on failure")
    parser.add_argument("--validate-model", default="gpt-oss:120b", help="Judge model (default: gpt-oss:120b)")
    parser.add_argument("--validate-provider", default="ollama", help="Judge provider (default: ollama)")
    parser.add_argument("--max-retries", type=int, default=3, help="Max regenerations per question (default: 3)")
    args = parser.parse_args()

    hops_filter = None if args.all_hops else args.hops
    asyncio.run(main(
        hops_filter=hops_filter,
        model_name=args.model,
        limit=args.limit,
        source_file=args.source if not args.staleness_csv else None,
        staleness_csv=args.staleness_csv,
        output_file=args.output,
        validate=args.validate,
        validate_model=args.validate_model,
        validate_provider=args.validate_provider,
        max_retries=args.max_retries,
    ))
