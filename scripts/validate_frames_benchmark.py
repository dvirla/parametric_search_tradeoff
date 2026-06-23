"""
Validate (and optionally rebuild) the FRAMES formal "neutral" anchor used by the prompt-cue
experiment.

`data/frames_benchmark.jsonl` was produced natural->formal by scripts/paraphrase_frames_benchmark.py
with an audit (BenchmarkStyleAudit) that checked equivalence + impersonal style but NEVER checked
for LEAKAGE of an intermediate answer. Since every cue variant in the cue-sensitivity experiment is
measured against this anchor, a leaky or drifted anchor contaminates every contrast. This tool
re-audits each anchor with the strict NeutralAnchorAudit (equivalence + NO-LEAK + neutral register)
and, in regenerate mode, rebuilds the failing ones fresh from the original FRAMES prompt.

Grounding is the ORIGINAL FRAMES prompt + gold final answer ONLY. We deliberately do NOT use any
hop decomposition (FRAMES has no gold per-hop answers; the staleness-CSV hops are a classification
byproduct, not ground truth, and would mislead the leak judgement).

Scope: the 541 NON-STALE FRAMES questions (the set the cue experiment uses), filtered via
results/frames_staleness_flash.csv joined to the HF prompt text.

Modes:
  audit       — strict-audit every anchor, write a report CSV, print leak/drift/register rates.
                Makes no changes.
  regenerate  — keep anchors that PASS the strict audit verbatim; regenerate the FAILING ones fresh
                from the original prompt (strict prompt + audit loop). Writes a NEW file
                data/frames_benchmark_strict.jsonl (never clobbers the original). Use --force to
                regenerate every question regardless of whether it already passes.

Usage:
  # 1. see how leaky the current anchor is (no changes)
  uv run python scripts/validate_frames_benchmark.py --mode audit

  # 2. build the strict anchor (keep clean rows, regenerate the bad ones)
  uv run python scripts/validate_frames_benchmark.py --mode regenerate

Then point the cue pipeline at the strict anchor:
  uv run python scripts/paraphrase_frames_cues.py --base data/frames_benchmark_strict.jsonl --validate
"""
import argparse
import asyncio
import csv
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.services.base_agent import BaseAgent

_DEFAULT_BASE = "data/frames_benchmark.jsonl"
_DEFAULT_STALENESS = "results/frames_staleness_flash.csv"
_DEFAULT_STRICT_OUT = "data/frames_benchmark_strict.jsonl"
_DEFAULT_REPORT = "results/frames_benchmark_audit.csv"

# Strict natural->formal rewriter. Same direction as paraphrase_frames_benchmark.py but with an
# explicit NO-LEAK rule and explicit register-neutrality.
STRICT_SYSTEM_PROMPT = """\
You are a precise prompt rewriter. You are given a real, naturally-phrased multi-hop user question. \
Rewrite it as a terse, formal BENCHMARK question (the way HotpotQA or MuSiQue would phrase it) that \
preserves the EXACT same information need and the same final answer, and that will serve as a NEUTRAL \
control in a controlled experiment.

Hard rules:
1. PRESERVE THE INFORMATION NEED. Do not add, drop, or alter any constraint, entity, date, or number. \
Do not combine the question with an extra sub-question. The rewrite must resolve to the SAME final answer.
2. NO LEAK. Keep every INTERMEDIATE result implicit, exactly as the original does — refer to each \
intermediate target only by its defining description (e.g. "the director of the film adaptation of X's \
second novel"), NEVER by its resolved name/value, and never remove or short-circuit a reasoning hop. The \
same chain of reasoning must still be required. Spelling the chain out explicitly with nested \
descriptions is fine and encouraged; revealing an intermediate ANSWER is not.
3. NEUTRAL REGISTER. Terse, impersonal, self-contained. NO confidence markers (no "exactly/surely", no \
"I think/maybe"), NO first-person goal framing, NO politeness ("could you/please"), NO greetings or \
narrative, and NO answer-length/format instruction.
4. Return ONLY the rewritten question text — no quotes, no commentary, no preamble.
"""

REWRITE_TEMPLATE = """\
Original natural user question:
{question}

Intended final answer (must stay the same): {answer}

Rewrite it as a terse, formal, register-neutral benchmark question following the hard rules. Return only the rewritten question."""


def load_nonstale_ids(base_path: str, staleness_csv: str) -> list[dict]:
    """Load existing anchors joined to non-stale FRAMES rows. Returns list of
    {example_id, original_question, formal_text, answer}."""
    import pandas as pd

    base = pd.read_json(base_path, lines=True)
    stale = pd.read_csv(staleness_csv)
    nonstale_q = set(
        stale.loc[stale["is_stale"] == False, "aggregate_question"].str.strip()  # noqa: E712
    )
    rows = []
    for _, r in base.iterrows():
        orig = str(r.get("original_question", "")).strip()
        if orig not in nonstale_q:
            continue
        rows.append({
            "example_id": str(r["example_id"]),
            "original_question": orig,
            "formal_text": str(r["text"]).strip(),
            "answer": r["answer"],
            "reasoning_types": r.get("reasoning_types"),
        })
    print(f"Non-stale anchors loaded: {len(rows)} (of {len(base)} total)")
    return rows


def load_existing_ids(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    ids = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    ids.add(json.loads(line)["example_id"])
                except (KeyError, json.JSONDecodeError):
                    pass
    return ids


async def rewrite_one(agent: BaseAgent, item: dict, feedback: str = "") -> str:
    prompt = REWRITE_TEMPLATE.format(question=item["original_question"], answer=item["answer"])
    if feedback:
        prompt += ("\n\nThe previous attempt was rejected by an auditor. Fix it:\n"
                   f"{feedback}\nReturn only the corrected rewritten question.")
    response = await agent.arun(prompt)
    return response.output.strip().strip('"').strip("'")


def _audit_meta(audit, status: str, attempts: int, origin: str) -> dict:
    return {
        "validation_status": status, "validation_attempts": attempts, "origin": origin,
        "equivalent": audit.equivalent, "leaks_intermediate": audit.leaks_intermediate,
        "is_neutral_register": audit.is_neutral_register, "drift": audit.drift,
        "register_issues": audit.register_issues,
        "audit_reasoning": audit.reasoning,
    }


async def validate_or_regenerate(rewriter, judge, item: dict, regenerate: bool,
                                 force: bool, max_retries: int) -> tuple[str, dict]:
    """Audit the existing anchor; in regenerate mode keep it if it passes (unless --force), else
    regenerate fresh with feedback. Returns (text, meta)."""
    from src.services.paraphrase_validation import audit_neutral_anchor, neutral_anchor_feedback

    # Attempt 0: the EXISTING formal text.
    existing = item["formal_text"]
    try:
        audit = await audit_neutral_anchor(judge, item["original_question"], existing, str(item["answer"]))
    except Exception as e:
        if not regenerate:
            raise
        audit = None
        feedback = ""
        print(f"      [audit error on existing {item['example_id']}] {e}")

    if audit is not None and audit.ok and not force:
        return existing, _audit_meta(audit, "pass", 1, "kept")

    if not regenerate:
        # audit-only mode: report the existing text's audit (failed or, with --force semantics N/A)
        if audit is None:
            return existing, {"validation_status": "judge_error", "validation_attempts": 1,
                              "origin": "existing", "equivalent": None, "leaks_intermediate": None,
                              "is_neutral_register": None, "drift": None, "register_issues": None,
                              "audit_reasoning": None}
        return existing, _audit_meta(audit, "fail", 1, "existing")

    # regenerate mode: loop, seeded with feedback from the failed existing audit.
    feedback = neutral_anchor_feedback(audit) if audit is not None else ""
    last_text, last_audit = existing, audit
    for attempt in range(max_retries + 1):
        last_text = await rewrite_one(rewriter, item, feedback)
        try:
            last_audit = await audit_neutral_anchor(
                judge, item["original_question"], last_text, str(item["answer"]))
        except Exception as e:
            last_audit = None
            feedback = ""
            print(f"      [audit error attempt {attempt + 1} {item['example_id']}] {e}")
            continue
        if last_audit.ok:
            return last_text, _audit_meta(last_audit, "pass", attempt + 2, "regenerated")
        feedback = neutral_anchor_feedback(last_audit)

    if last_audit is None:
        meta = {"validation_status": "judge_error", "validation_attempts": max_retries + 2,
                "origin": "regenerated", "equivalent": None, "leaks_intermediate": None,
                "is_neutral_register": None, "drift": None, "register_issues": None,
                "audit_reasoning": None}
    else:
        meta = _audit_meta(last_audit, "fail", max_retries + 2, "regenerated")
    return last_text, meta


async def main(args):
    rows = load_nonstale_ids(args.base, args.staleness_csv)
    if args.limit is not None:
        rows = rows[:args.limit]

    regenerate = args.mode == "regenerate"

    rewriter = BaseAgent(provider_name=args.provider, model_name=args.model,
                         system_prompt=STRICT_SYSTEM_PROMPT, use_thinking=False, temperature=0.4)
    from src.services.paraphrase_validation import make_judge_agent, NeutralAnchorAudit
    judge = make_judge_agent(args.validate_model, args.validate_provider, output_type=NeutralAnchorAudit)
    print(f"Rewriter: {args.provider}/{args.model} | Judge: {args.validate_provider}/{args.validate_model}")

    if not regenerate:
        # AUDIT MODE — report only.
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        fields = ["example_id", "validation_status", "equivalent", "leaks_intermediate",
                  "is_neutral_register", "drift", "register_issues", "audit_reasoning",
                  "original_question", "formal_text", "answer"]
        counts = {"pass": 0, "fail": 0, "judge_error": 0}
        leak = drift = nonneutral = 0
        with open(args.report, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for i, item in enumerate(rows):
                _, meta = await validate_or_regenerate(rewriter, judge, item, False, False, args.max_retries)
                counts[meta["validation_status"]] = counts.get(meta["validation_status"], 0) + 1
                leak += 1 if meta.get("leaks_intermediate") else 0
                drift += 1 if (meta.get("drift") not in (None, "none")) else 0
                nonneutral += 1 if (meta.get("is_neutral_register") is False) else 0
                w.writerow({
                    "example_id": item["example_id"], "validation_status": meta["validation_status"],
                    "equivalent": meta.get("equivalent"), "leaks_intermediate": meta.get("leaks_intermediate"),
                    "is_neutral_register": meta.get("is_neutral_register"), "drift": meta.get("drift"),
                    "register_issues": "; ".join(meta.get("register_issues") or []),
                    "audit_reasoning": meta.get("audit_reasoning"),
                    "original_question": item["original_question"], "formal_text": item["formal_text"],
                    "answer": item["answer"],
                })
                f.flush()
                if (i + 1) % 25 == 0 or i == len(rows) - 1:
                    print(f"  audited {i + 1}/{len(rows)} | pass={counts['pass']} fail={counts['fail']} "
                          f"leak={leak} drift={drift} non-neutral={nonneutral}")
        print(f"\nAudit report: {args.report}")
        print(f"Strict audit of {len(rows)} non-stale anchors:")
        print(f"  PASS: {counts['pass']}  FAIL: {counts['fail']}  judge_error: {counts.get('judge_error', 0)}")
        print(f"  leaks_intermediate: {leak}   drift!=none: {drift}   non-neutral register: {nonneutral}")
        print("Re-run with --mode regenerate to rebuild the failing anchors.")
        return

    # REGENERATE MODE — write strict file (resumable).
    existing_done = load_existing_ids(args.output)
    pending = [r for r in rows if r["example_id"] not in existing_done]
    print(f"Regenerate mode: {len(existing_done)} already in {args.output}, {len(pending)} pending"
          + (" (--force: regenerate even passing)" if args.force else ""))
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    stats = {}
    with open(args.output, "a") as out_f:
        for i, item in enumerate(pending):
            try:
                text, meta = await validate_or_regenerate(
                    rewriter, judge, item, True, args.force, args.max_retries)
            except Exception as e:
                print(f"  [!] failed on {item['example_id']}: {e}")
                continue
            key = f"{meta['validation_status']}/{meta['origin']}"
            stats[key] = stats.get(key, 0) + 1
            record = {
                "text": text, "answer": item["answer"], "example_id": item["example_id"],
                "original_question": item["original_question"],
                "reasoning_types": item.get("reasoning_types"),
                "source": "frames_benchmark_strict", **meta,
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            if (i + 1) % 10 == 0 or i == len(pending) - 1:
                print(f"  [{i + 1}/{len(pending)}] stats={stats}")
    print(f"\nDone. Strict anchor: {args.output}")
    print(f"Breakdown (status/origin): {stats}")
    print("NOTE: 'fail/*' rows are kept (flagged) so the example_id set stays complete; "
          "filter on validation_status when building cue variants if you want pass-only anchors.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Strict-audit / rebuild the FRAMES neutral anchor")
    p.add_argument("--mode", choices=["audit", "regenerate"], default="audit",
                   help="audit: report only; regenerate: keep passing rows, rebuild failing ones")
    p.add_argument("--base", default=_DEFAULT_BASE, help="Existing anchor JSONL (frames_benchmark.jsonl)")
    p.add_argument("--staleness-csv", default=_DEFAULT_STALENESS, help="Staleness CSV for the non-stale filter")
    p.add_argument("--output", default=_DEFAULT_STRICT_OUT, help="Strict anchor output (regenerate mode)")
    p.add_argument("--report", default=_DEFAULT_REPORT, help="Audit report CSV (audit mode)")
    p.add_argument("--force", action="store_true", help="Regenerate every question, even passing ones")
    p.add_argument("--limit", type=int, default=None, help="Process at most N anchors")
    p.add_argument("--model", default="gpt-oss:120b", help="Rewriter model (default: gpt-oss:120b)")
    p.add_argument("--provider", default="ollama", help="Rewriter provider (default: ollama)")
    p.add_argument("--validate-model", default="gpt-oss:120b", help="Judge model (default: gpt-oss:120b)")
    p.add_argument("--validate-provider", default="ollama", help="Judge provider (default: ollama)")
    p.add_argument("--max-retries", type=int, default=3, help="Max regenerations per failing anchor (default: 3)")
    args = p.parse_args()

    asyncio.run(main(args))
