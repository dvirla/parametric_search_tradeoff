"""LLM judge — does the natural2 paraphrase still REQUIRE each missed hop?

The verbatim-substring detector in inspect_missed_leakage.py misses *semantic*
leakage, where the paraphrase preserves the information need but collapses the
reasoning chain — naming the bridge entity via a synonym/description, or eliding
an intermediate hop entirely. Example:

    formal: "The military unit that operates destroyers and once had a ship named
             the USS Sanderling has a force called the seals. What does seal stand for?"
    natural2: "I've heard about a special-operations force called the SEALs — could
             you explain what the letters in that acronym stand for?"

Hops 0-2 (identify the Navy via the USS Sanderling chain) are no longer required:
the paraphrase already names the SEALs. If the model "misses" those hops it is an
ARTIFACT of paraphrasing, not a behavioral miss — and it inflates the Missed cell
under natural2 without any accuracy cost.

This script asks a gemini-3-flash judge, per example, whether the PARAPHRASE still
requires resolving each gold reasoning hop (given the formal question + the ordered
hop decomposition with gold answers). It then recomputes ΔM (natural2 - benchmark
Missed rate) keeping only genuinely-required missed hops, to decide whether the
missed-hop increase under paraphrasing is real or an artifact.

Usage
-----
    # judge (resumable) then recompute:
    uv run python scripts/judge_paraphrase_leakage.py --workers 8
    # quick pilot on a few examples to eyeball judge quality:
    uv run python scripts/judge_paraphrase_leakage.py --limit 5
    # recompute only, from an existing judgments file:
    uv run python scripts/judge_paraphrase_leakage.py --recompute-only
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import src.viz as viz
from src.services.base_agent import BaseAgent
# Reuse the exact data assembly from the inspector so definitions stay in sync.
from scripts.inspect_missed_leakage import (
    _interplay_frame, _commitment_frame, _question_text)

DEFAULT_OUT = "results/natural2_paper_figures/paraphrase_leakage_judgments.jsonl"


# ── structured judge output ─────────────────────────────────────────────────--
class HopLeak(BaseModel):
    hop_index: int = Field(description="0-based index of the gold reasoning hop.")
    still_required: bool = Field(
        description="True if the PARAPHRASE still requires the solver to figure out "
                    "this sub-question's answer to reach the final answer. False if the "
                    "paraphrase already supplies it (names the entity, verbatim or via a "
                    "synonym/description) or restructures the question so the hop is "
                    "unnecessary.")
    leak_type: str = Field(
        description="One of: 'none' (still fully required), 'verbatim' (gold answer text "
                    "appears in the paraphrase), 'semantic' (entity named via synonym or "
                    "unambiguous description), 'elided' (hop dropped / made unnecessary by "
                    "restructuring).")
    evidence: str = Field(description="The span of the PARAPHRASE that supplies the hop, or '' if none.")
    rationale: str = Field(description="One short clause justifying the decision.")


class ParaphraseLeakage(BaseModel):
    hops: list[HopLeak] = Field(description="One assessment per gold reasoning hop, in order.")
    information_need_preserved: bool = Field(
        description="True if the paraphrase asks for the SAME final answer as the formal question.")


SYSTEM_PROMPT = """You are an expert annotator analyzing whether a paraphrased question \
preserves the multi-hop reasoning burden of an original benchmark question.

A benchmark (formal) question is a chain of reasoning hops: each hop resolves an \
intermediate entity/fact that the next hop depends on, ending in the final answer. \
A paraphrase rewrites the question in natural, user-like language. A good paraphrase \
keeps the SAME final information need AND the SAME reasoning burden. A leaky paraphrase \
preserves the information need but COLLAPSES the chain — it names a bridge entity \
directly (verbatim or via an unambiguous synonym/description) or restructures the \
question so an intermediate hop no longer needs to be resolved.

For EACH gold hop you must decide: given ONLY the paraphrase, must the solver still \
figure out this sub-question's answer to reach the final answer? If the paraphrase \
already states or unambiguously pins that intermediate fact, the hop is NOT still \
required (mark the leak_type). Be strict about 'semantic' leakage: if a competent \
reader of the paraphrase already knows the intermediate entity without doing the hop's \
reasoning, it is leaked. Judge the reasoning burden, not surface wording."""

TEMPLATE = """FORMAL (benchmark) question:
{formal_q}

PARAPHRASE (natural, user-like) question:
{natural_q}

FINAL gold answer: {final_gold}

Gold reasoning hops (in order). [MODEL-SKIPPED] marks hops the model did not search \
under the paraphrase — these are the ones we must decide are genuine misses or leakage \
artifacts:
{hop_block}

For each hop, decide whether the PARAPHRASE still requires resolving it."""


def build_judge(provider: str, model: str) -> BaseAgent:
    print(f"Initializing leakage judge ({model} via {provider})...")
    return BaseAgent(provider_name=provider, model_name=model,
                     agent_name="paraphrase_leakage_judge",
                     system_prompt=SYSTEM_PROMPT, output_type=ParaphraseLeakage,
                     use_thinking=True)


# ── data assembly ─────────────────────────────────────────────────────────────
def candidate_examples() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per-hop frame, candidate examples). Candidates = natural2 examples
    with >=1 truly-missed hop (those are the ones whose Missed status we must vet)."""
    ip = _interplay_frame()
    cl = _commitment_frame()
    qt = _question_text()
    keys = ["model", "variant", "example_id", "hop_index"]
    hops = ip.merge(cl[keys + ["hop_question", "gold_answer"]], on=keys, how="left")

    nat = hops[hops.variant == "natural"].copy()
    nat["truly_missed"] = nat["truly_missed"].fillna(False)
    exposed = (nat.groupby(["model", "example_id"])["truly_missed"].sum()
                  .reset_index().query("truly_missed >= 1")[["model", "example_id"]])
    cand = exposed.merge(qt[["model", "example_id", "formal_q", "natural2_q", "natural2_gold"]],
                         on=["model", "example_id"], how="left").dropna(subset=["formal_q", "natural2_q"])
    return hops, cand


def _hop_block(nat_hops: pd.DataFrame) -> str:
    lines = []
    for _, h in nat_hops.sort_values("hop_index").iterrows():
        tag = " [MODEL-SKIPPED]" if bool(h["truly_missed"]) else ""
        lines.append(f"  hop {int(h['hop_index'])}{tag}: {h['hop_question']}  "
                     f"=> gold: {h['gold_answer']}")
    return "\n".join(lines)


# ── judging ─────────────────────────────────────────────────────────────────--
async def judge_one(judge: BaseAgent, ex: dict, hops: pd.DataFrame, max_retries=3) -> dict:
    nat_hops = hops[(hops.variant == "natural") & (hops.model == ex["model"]) &
                    (hops.example_id == ex["example_id"])]
    prompt = TEMPLATE.format(formal_q=ex["formal_q"], natural_q=ex["natural2_q"],
                             final_gold=ex["natural2_gold"], hop_block=_hop_block(nat_hops))
    for attempt in range(max_retries):
        try:
            out: ParaphraseLeakage = (await judge.arun(prompt)).output
            return {"model": ex["model"], "example_id": ex["example_id"],
                    "formal_q": ex["formal_q"], "natural2_q": ex["natural2_q"],
                    "information_need_preserved": bool(out.information_need_preserved),
                    "hops": [h.model_dump() for h in out.hops]}
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                return {"model": ex["model"], "example_id": ex["example_id"],
                        "error": str(e), "hops": []}


async def run_judge(cand: pd.DataFrame, hops: pd.DataFrame, out_path: str,
                    provider: str, model: str, workers: int):
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["model"], r["example_id"]))
                except Exception:
                    pass
    pending = [r for _, r in cand.iterrows() if (r["model"], r["example_id"]) not in done]
    print(f"Candidates: {len(cand)} | already judged: {len(done)} | to judge: {len(pending)}")
    if not pending:
        return

    judge = build_judge(provider, model)
    sem = asyncio.Semaphore(workers)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    f = open(out_path, "a")
    n = 0

    async def work(ex):
        nonlocal n
        async with sem:
            res = await judge_one(judge, ex, hops)
        f.write(json.dumps(res) + "\n"); f.flush()
        n += 1
        if n % 20 == 0:
            print(f"  judged {n}/{len(pending)}")
        return res

    await asyncio.gather(*[work(ex) for ex in pending])
    f.close()
    print(f"Wrote judgments -> {out_path}")


# ── recompute ΔM keeping only genuinely-required missed hops ──────────────────-
def recompute(hops: pd.DataFrame, out_path: str) -> pd.DataFrame:
    if not os.path.exists(out_path):
        sys.exit(f"No judgments file at {out_path}; run the judge first.")
    req = {}          # (model, example_id, hop_index) -> still_required
    leak_type = {}
    with open(out_path) as f:
        for line in f:
            r = json.loads(line)
            for h in r.get("hops", []):
                k = (r["model"], r["example_id"], int(h["hop_index"]))
                req[k] = bool(h["still_required"])
                leak_type[k] = h.get("leak_type", "none")

    h = hops.copy()
    h["truly_missed"] = h["truly_missed"].fillna(False)

    def required(row):
        if row["variant"] != "natural":
            return True  # formal/benchmark hops are the full chain — always required
        return req.get((row["model"], row["example_id"], int(row["hop_index"])), True)
    h["still_required"] = h.apply(required, axis=1)
    h["genuine_missed"] = h["truly_missed"] & h["still_required"]
    h["artifact_missed"] = h["truly_missed"] & (~h["still_required"])

    rows = []
    for (model, variant), sub in h.groupby(["model", "variant"]):
        n = len(sub)
        rows.append({"model": viz.display_name(model), "variant": variant, "n_hops": n,
                     "missed_raw": sub["truly_missed"].sum() / n,
                     "missed_genuine": sub["genuine_missed"].sum() / n,
                     "artifact": sub["artifact_missed"].sum() / n,
                     "n_missed": int(sub["truly_missed"].sum()),
                     "n_artifact": int(sub["artifact_missed"].sum())})
    pm = pd.DataFrame(rows)

    drows = []
    for model, sub in pm.groupby("model"):
        b = sub[sub.variant == "benchmark"]; nt = sub[sub.variant == "natural"]
        if b.empty or nt.empty:
            continue
        b, nt = b.iloc[0], nt.iloc[0]
        dm_raw = nt["missed_raw"] - b["missed_raw"]
        dm_gen = nt["missed_genuine"] - b["missed_raw"]
        drows.append({"model": model, "dM_raw": dm_raw, "dM_corrected": dm_gen,
                      "artifact_share_of_dM": (dm_raw - dm_gen) / dm_raw if dm_raw else np.nan,
                      "nat2_missed": nt["n_missed"], "nat2_artifact": nt["n_artifact"]})
    return pm, pd.DataFrame(drows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--provider", default="Google")
    ap.add_argument("--model", default="gemini-3-flash-preview")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="Pilot: judge only N candidates.")
    ap.add_argument("--recompute-only", action="store_true")
    args = ap.parse_args()

    hops, cand = candidate_examples()
    if args.limit:
        cand = cand.head(args.limit)

    if not args.recompute_only:
        asyncio.run(run_judge(cand, hops, args.out, args.provider, args.model, args.workers))

    pm, dl = recompute(hops, args.out)
    pd.set_option("display.width", 160, "display.max_columns", 30)
    print("\n=== Missed rate: raw vs genuine (artifact removed) ===")
    print(pm.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n=== ΔM (natural2 - benchmark): raw vs corrected ===")
    print(dl.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n=== Verdict ===")
    for _, r in dl.iterrows():
        share = r["artifact_share_of_dM"]
        if pd.notna(share) and share >= 0.66:
            v = "increase is mostly an ARTIFACT of paraphrasing"
        elif pd.notna(share) and share >= 0.34:
            v = "increase is PART artifact, part real"
        else:
            v = "increase is mostly REAL behavior"
        print(f"  {r['model']:<22} ΔM {r['dM_raw']:+.3f} -> corrected {r['dM_corrected']:+.3f}  "
              f"(artifact share {share:.0%})  -> {v}")


if __name__ == "__main__":
    main()
