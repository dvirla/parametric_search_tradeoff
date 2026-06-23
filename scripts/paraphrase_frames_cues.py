"""
Generate single-cue paraphrases of FRAMES questions for the prompt-cue sensitivity experiment.

Goal: find the *source* of behavioral change in a search agent. Prior work found formal vs
natural phrasing does not change accuracy, so we now isolate *which linguistic cue* shifts the
agent's retrieval policy. We do this One-Factor-At-A-Time (OFAT) around a NEUTRAL formal anchor —
the terse benchmark rewrite in data/frames_benchmark.jsonl — producing one variant per cue level
that differs from the anchor ONLY along the target cue.

Cue dimensions (need-irrelevant surface cues should NOT change retrieval; need-relevant cues may):
  epistemic stance (5-pt), framing/personalization, pragmatic mitigation  [need-irrelevant]
  structural explicitness, output constraints                            [need-relevant]

INVARIANT for every condition (incl. the need-relevant ones): the variant must preserve the exact
information need, resolve to the same gold answer, and never leak/short-circuit an intermediate
reasoning step. Otherwise the measured search-behavior difference is confounded by the question
being genuinely easier, not just phrased differently. Enforced by the CueComplianceAudit
generate->judge->regenerate loop (reused from src/services/paraphrase_validation.py).

We restrict to the 541 NON-STALE FRAMES questions (results/frames_staleness_flash.csv,
is_stale == False), joined to the anchor on the original FRAMES question text.

Usage:
  # full run, all conditions, validated (ollama by default)
  uv run python scripts/paraphrase_frames_cues.py --validate

  # quick pilot against a cloud model
  uv run python scripts/paraphrase_frames_cues.py --limit 5 --conditions epi_strong_hedge \
      --provider Google --model gemini-flash-latest \
      --validate --validate-provider Google --validate-model gemini-flash-latest
"""

import argparse
import asyncio
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.services.base_agent import BaseAgent

_DEFAULT_BASE = "data/frames_benchmark.jsonl"
_DEFAULT_STALENESS = "results/frames_staleness_flash.csv"
_DEFAULT_OUTDIR = "data/frames_cues"

SYSTEM_PROMPT = """\
You are a precise prompt rewriter for a controlled linguistics experiment. You are given a NEUTRAL \
benchmark question and ONE specific linguistic cue to apply. Rewrite the question so it applies \
THAT cue and nothing else.

Cardinal rules (these override everything):
1. PRESERVE THE INFORMATION NEED. Do not add, drop, or alter any constraint, entity, date, or \
number. The rewrite must resolve to the EXACT SAME answer as the neutral question.
2. NO LEAK. Keep every intermediate reasoning step implicit. Never name, reveal, or make directly \
identifiable any bridge entity, and never short-circuit a reasoning hop. The same chain of \
reasoning must still be required to answer.
3. CHANGE ONLY THE TARGET CUE. Leave every other dimension (confidence, first-person goal-framing, \
politeness, explicit vs hidden reasoning steps, output-length instructions) exactly as in the \
neutral question, except the one cue you are told to apply.
4. Return ONLY the rewritten question text — no quotes, no commentary, no preamble.
"""

# Per-dimension descriptions, used both to instruct the rewriter and to tell the auditor which
# OTHER cues must remain unchanged (the "forbidden" set).
DIMENSION_DESCRIPTIONS = {
    "epistemic": ("the SPEAKER'S expressed confidence in the answer (or in their own recall of "
                  "it). It is realized as the asker's OWN stance framing the question — boosters: "
                  "'surely you know', 'you must know this', 'I'm certain there's a clear answer', "
                  "'no doubt you can tell me', 'everyone knows this'; hedges: 'I might be wrong, "
                  "but', 'I'm not sure, but isn't it...?', 'I could be misremembering', 'I think', "
                  "'if I recall correctly'. CRITICAL: this is NOT a bare adverb dropped inside the "
                  "interrogative ('What is CLEARLY the full name...?', 'What conflict POSSIBLY took "
                  "place...?') — that attributes the (un)certainty to the asked entity, reads "
                  "hollow, and for hedges even alters the factual claim. The (un)certainty must "
                  "sit on the SPEAKER and must never loosen or alter any date, number, name, or "
                  "constraint. A bare imperative ('Tell me exactly') is request-directness, not an "
                  "epistemic stance, and does not count."),
    "framing": ("first-person GOAL or task framing that presents the question as the user's own "
                "ongoing effort (e.g. 'I'm trying to figure out', 'I'm working on', 'for my project')"),
    "mitigation": ("politeness / softened-request markers (e.g. 'could you', 'please', "
                   "'would you mind helping me')"),
    "structural": ("whether the multi-step reasoning chain is spelled out explicitly vs hidden/buried "
                   "in relative clauses without naming the intermediate steps"),
    "output": ("instructions about answer length or format (e.g. 'just the final answer' vs "
               "'explain in 2-4 sentences')"),
}


def _forbidden(target_dim: str) -> str:
    """The cue dimensions (other than the target) that must stay unchanged from the anchor."""
    return "\n".join(
        f"- {dim}: {desc}" for dim, desc in DIMENSION_DESCRIPTIONS.items() if dim != target_dim
    )


# Each spec: slug -> {dimension, level, cue_class, instruction}.
# `instruction` is the per-cue user-message directive; `cue_description` (for the audit) is built
# from the dimension + the concrete instruction. "neutral" has no instruction (pure copy).
CUE_SPECS = {
    "epi_strong_boost": {
        "dimension": "epistemic", "level": "strong_booster", "cue_class": "need_irrelevant",
        "instruction": ("Rewrite so the ASKER sounds personally and strongly certain — as if they "
                        "half-know the answer already and fully expect a single definite one. The "
                        "certainty must be the SPEAKER'S stance framing the question. Pick the "
                        "confident frame that fits THIS sentence most naturally, e.g. 'Surely you "
                        "know — <question>', 'You must know this one: <question>', 'I'm certain "
                        "there's a clear answer here — <question>', 'No doubt you can tell me "
                        "<question>', 'This is surely an easy one — <question>', 'Everyone knows "
                        "this: <question>', 'I'm positive about this — <question>'. "
                        "FORBIDDEN realizations: (a) a bare adverb dropped inside the question "
                        "('What is CLEARLY the full name...?', 'Who is OBVIOUSLY the eponym...?') — "
                        "this makes the entity sound obvious, not the speaker confident, and reads "
                        "hollow; (b) a detached dash-tag commenting that the answer is "
                        "'well-documented'/'definitive'/'well-established'; (c) a bare imperative "
                        "('Tell me exactly') — that is request-directness, not confidence. Use ONE "
                        "confident frame, integrated naturally. Keep every date, number, name, and "
                        "constraint identical, and do NOT add first-person GOAL-framing ('I'm "
                        "trying to figure out') or politeness ('could you please')."),
    },
    "epi_mild_boost": {
        "dimension": "epistemic", "level": "mild_booster", "cue_class": "need_irrelevant",
        "instruction": ("Rewrite so the ASKER sounds lightly, casually confident — mildly assured "
                        "the answer is gettable, without strong insistence. The confidence must be "
                        "the SPEAKER'S stance framing the question, not a bare adverb inside it. "
                        "Pick the light frame that fits THIS sentence most naturally, e.g. 'You "
                        "probably know this — <question>', 'I'd think this has a clear answer: "
                        "<question>', 'This should be an easy one — <question>', 'I'm fairly sure "
                        "you can tell me <question>', 'I'd imagine you know — <question>'. Do NOT "
                        "drop a bare adverb inside the question and do NOT use a detached meta-tag "
                        "or a bare imperative. Use ONE light frame, subtle and integrated. Keep "
                        "every date, number, name, and constraint identical, and do NOT add "
                        "goal-framing or politeness."),
    },
    "epi_mild_hedge": {
        "dimension": "epistemic", "level": "mild_hedge", "cue_class": "need_irrelevant",
        "instruction": ("Rewrite so the ASKER sounds slightly unsure about the answer or their own "
                        "recall of it. The doubt must sit on the SPEAKER, not as an adverb inside "
                        "the proposition (NOT 'What conflict PROBABLY took place...?', which alters "
                        "the claim). Pick the light hedge frame that fits THIS sentence most "
                        "naturally, e.g. 'I think <question>', 'If I recall correctly, <question>', "
                        "'Isn't it the case that <question>', 'I'd guess, but <question>', 'I "
                        "believe <question>'. Use ONE marker, subtle and integrated. Keep the doubt "
                        "on the answer/recall and NEVER loosen a constraint — do NOT turn an exact "
                        "value like '400 years' into 'about 400 years' or alter any date, number, "
                        "or name. The 'I' may only express doubt; do NOT add goal-framing ('I'm "
                        "trying to')."),
    },
    "epi_strong_hedge": {
        "dimension": "epistemic", "level": "strong_hedge", "cue_class": "need_irrelevant",
        "instruction": ("Rewrite so the ASKER sounds very tentative and unsure about the answer. "
                        "The doubt must sit on the SPEAKER, not as an adverb inside the proposition "
                        "(NOT 'What conflict POSSIBLY took place...?', which alters the claim). Pick "
                        "the strong hedge frame that fits THIS sentence most naturally, e.g. 'I "
                        "might be wrong, but <question>', 'I could be misremembering, but "
                        "<question>', 'I'm really not sure, but <question>', 'I have a vague "
                        "feeling about this — <question>', 'Maybe you can tell me, because I'm "
                        "blanking — <question>'. Use exactly ONE such frame — never stack two doubt "
                        "phrases ('I'm really not sure AND I might be totally wrong'). Keep the "
                        "doubt on the answer/recall and NEVER loosen a constraint — keep every "
                        "date, number, name, and constraint exact (e.g. '400 years', not 'about "
                        "400 years'). Do NOT add goal-framing ('I'm trying to figure out') or "
                        "politeness."),
    },
    "framing_first_person": {
        "dimension": "framing", "level": "first_person", "cue_class": "need_irrelevant",
        "instruction": ("Frame the question as the user's OWN ongoing effort/goal, in the first "
                        "person (e.g. 'I'm trying to figure out...', 'I'm working on...'). Keep the "
                        "confidence neutral (neither boosted nor hedged) and do NOT add politeness."),
    },
    "mitigation_polite": {
        "dimension": "mitigation", "level": "polite", "cue_class": "need_irrelevant",
        "instruction": ("Soften the question into a polite, mitigated request (e.g. 'Could you help "
                        "me find...', 'Would you mind telling me...', add 'please'). Keep confidence "
                        "neutral and do NOT add first-person goal-framing beyond the politeness."),
    },
    "structural_implicit": {
        "dimension": "structural", "level": "implicit", "cue_class": "need_relevant",
        "instruction": ("Bury the reasoning chain: instead of spelling the steps out explicitly, "
                        "refer to the intermediate targets only by oblique description folded into the "
                        "sentence, so the user never names the intermediate steps. CRITICAL: the SAME "
                        "reasoning hops must still be required and you must NOT reveal or make "
                        "identifiable any intermediate answer (no leak). Keep tone neutral and add no "
                        "output instruction."),
    },
    "output_final_only": {
        "dimension": "output", "level": "final_only", "cue_class": "need_relevant",
        "instruction": ("Keep the question itself unchanged in meaning and tone, but explicitly ask "
                        "for ONLY the final answer with no explanation (e.g. append 'Answer with just "
                        "the final answer, nothing else.'). Do NOT change confidence, framing, or "
                        "politeness."),
    },
    "output_detailed": {
        "dimension": "output", "level": "detailed", "cue_class": "need_relevant",
        "instruction": ("Keep the question itself unchanged in meaning and tone, but explicitly ask "
                        "for a detailed answer (e.g. append 'Explain your answer in 2-4 sentences.'). "
                        "Do NOT change confidence, framing, or politeness."),
    },
}

# "neutral" is the anchor itself — produced by copying, not by an LLM call.
ALL_CONDITIONS = ["neutral"] + list(CUE_SPECS.keys())

TRANSFORM_TEMPLATE = """\
NEUTRAL benchmark question:
{anchor}

Intended answer (must stay the same): {answer}

TARGET cue to apply (and ONLY this cue):
{instruction}

Rewrite the neutral question applying ONLY that cue, following the cardinal rules. Return only the rewritten question."""


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


def load_base(base_path: str, staleness_csv: str) -> list[dict]:
    """Load the neutral anchor (frames_benchmark.jsonl), keep only NON-STALE rows whose anchor
    passed the equivalence audit. Join staleness on the original FRAMES question text."""
    import pandas as pd

    base = pd.read_json(base_path, lines=True)
    stale = pd.read_csv(staleness_csv)
    nonstale_q = set(
        stale.loc[stale["is_stale"] == False, "aggregate_question"].str.strip()  # noqa: E712
    )

    rows = []
    skipped_stale = skipped_unvalidated = 0
    for _, r in base.iterrows():
        orig = str(r.get("original_question", "")).strip()
        if orig not in nonstale_q:
            skipped_stale += 1
            continue
        # Only build cue variants on top of a clean (equivalent) anchor. "anchor" is the
        # status stamped on a pre-audited neutral file (e.g. neutral_audited.jsonl) used as --base.
        status = r.get("validation_status")
        if status is not None and status not in ("pass", "unvalidated", "anchor"):
            skipped_unvalidated += 1
            continue
        rows.append({
            "example_id": str(r["example_id"]),
            "anchor": str(r["text"]).strip(),
            "answer": r["answer"],
            "original_question": orig,
        })
    print(f"Anchor rows: kept {len(rows)} non-stale "
          f"(skipped {skipped_stale} stale/unmatched, {skipped_unvalidated} non-pass anchors)")
    return rows


async def transform_one(agent: BaseAgent, item: dict, instruction: str, feedback: str = "") -> str:
    prompt = TRANSFORM_TEMPLATE.format(
        anchor=item["anchor"], answer=item["answer"], instruction=instruction)
    if feedback:
        prompt += ("\n\nThe previous attempt was rejected by an auditor. Fix it:\n"
                   f"{feedback}\nReturn only the corrected rewritten question.")
    response = await agent.arun(prompt)
    return response.output.strip().strip('"').strip("'")


async def transform_validated(agent: BaseAgent, judge_agent, item: dict, spec: dict,
                              max_retries: int) -> tuple[str, dict]:
    """Generate->judge->regenerate for a single cue. Returns (best_text, validation_metadata)."""
    from src.services.paraphrase_validation import audit_cue_compliance, cue_feedback

    cue_description = f"{DIMENSION_DESCRIPTIONS[spec['dimension']]}. Specifically: {spec['instruction']}"
    forbidden = _forbidden(spec["dimension"])

    feedback, last_text, last_audit = "", None, None
    for attempt in range(max_retries + 1):
        last_text = await transform_one(agent, item, spec["instruction"], feedback)
        try:
            audit = await audit_cue_compliance(
                judge_agent, item["anchor"], last_text, str(item["answer"]),
                cue_description, forbidden)
        except Exception as e:
            last_audit = None
            feedback = ""
            print(f"      [audit error attempt {attempt + 1}] {e}")
            continue
        last_audit = audit
        if audit.ok:
            return last_text, {"validation_status": "pass", "validation_attempts": attempt + 1,
                               "equivalent": True, "leaks_intermediate": False,
                               "cue_applied": True, "only_target_cue": True,
                               "drift": audit.drift, "issues": []}
        feedback = cue_feedback(audit)

    if last_audit is None:
        meta = {"validation_status": "judge_error", "validation_attempts": max_retries + 1,
                "equivalent": None, "leaks_intermediate": None, "cue_applied": None,
                "only_target_cue": None, "drift": None, "issues": None}
    else:
        meta = {"validation_status": "fail", "validation_attempts": max_retries + 1,
                "equivalent": last_audit.equivalent, "leaks_intermediate": last_audit.leaks_intermediate,
                "cue_applied": last_audit.cue_applied, "only_target_cue": last_audit.only_target_cue,
                "drift": last_audit.drift, "issues": last_audit.issues}
    return last_text, meta


def write_neutral(base_rows: list[dict], out_path: str):
    """The neutral condition is the anchor itself — copy it (resumable)."""
    existing = load_existing_ids(out_path)
    pending = [r for r in base_rows if r["example_id"] not in existing]
    if not pending:
        print(f"[neutral] nothing to do ({len(existing)} already written).")
        return
    with open(out_path, "a") as f:
        for r in pending:
            record = {
                "text": r["anchor"], "answer": r["answer"], "example_id": r["example_id"],
                "original_question": r["original_question"], "neutral_base": r["anchor"],
                "cue_dimension": "neutral", "cue_level": "neutral", "cue_class": "control",
                "source": "frames_cues", "validation_status": "anchor",
            }
            f.write(json.dumps(record) + "\n")
    print(f"[neutral] wrote {len(pending)} rows -> {out_path}")


async def run_condition(slug: str, base_rows: list[dict], out_dir: str, agent: BaseAgent,
                        judge_agent, validate: bool, max_retries: int):
    out_path = os.path.join(out_dir, f"{slug}.jsonl")

    if slug == "neutral":
        write_neutral(base_rows, out_path)
        return

    spec = CUE_SPECS[slug]
    existing = load_existing_ids(out_path)
    pending = [r for r in base_rows if r["example_id"] not in existing]
    print(f"\n=== {slug} ({spec['dimension']}/{spec['level']}, {spec['cue_class']}) ===")
    print(f"    {len(existing)} done, {len(pending)} pending")
    if not pending:
        return

    stats = {}
    with open(out_path, "a") as out_f:
        for i, item in enumerate(pending):
            try:
                if validate:
                    text, vmeta = await transform_validated(agent, judge_agent, item, spec, max_retries)
                else:
                    text = await transform_one(agent, item, spec["instruction"])
                    vmeta = {"validation_status": "unvalidated"}
            except Exception as e:
                print(f"  [!] {slug} failed on {item['example_id']}: {e}")
                continue

            stats[vmeta["validation_status"]] = stats.get(vmeta["validation_status"], 0) + 1
            record = {
                "text": text, "answer": item["answer"], "example_id": item["example_id"],
                "original_question": item["original_question"], "neutral_base": item["anchor"],
                "cue_dimension": spec["dimension"], "cue_level": spec["level"],
                "cue_class": spec["cue_class"], "source": "frames_cues", **vmeta,
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            if (i + 1) % 10 == 0 or i == len(pending) - 1:
                print(f"  [{i + 1}/{len(pending)}] {slug} stats={stats}")
    print(f"[{slug}] done. stats={stats}")


async def main(args):
    conditions = args.conditions or ALL_CONDITIONS
    unknown = [c for c in conditions if c not in ALL_CONDITIONS]
    if unknown:
        raise SystemExit(f"Unknown conditions: {unknown}. Valid: {ALL_CONDITIONS}")

    base_rows = load_base(args.base, args.staleness_csv)
    if args.limit is not None:
        base_rows = base_rows[:args.limit]
    os.makedirs(args.output_dir, exist_ok=True)

    needs_llm = any(c != "neutral" for c in conditions)
    agent = judge_agent = None
    if needs_llm:
        agent = BaseAgent(provider_name=args.provider, model_name=args.model,
                          system_prompt=SYSTEM_PROMPT, use_thinking=True)
        if args.validate:
            from src.services.paraphrase_validation import make_judge_agent, CueComplianceAudit
            print(f"Validation ON: cue-compliance audit via {args.validate_provider}/"
                  f"{args.validate_model}, up to {args.max_retries} regenerations.")
            judge_agent = make_judge_agent(args.validate_model, args.validate_provider,
                                           output_type=CueComplianceAudit)

    for slug in conditions:
        await run_condition(slug, base_rows, args.output_dir, agent, judge_agent,
                            args.validate, args.max_retries)

    print(f"\nDone. Condition files in {args.output_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Single-cue FRAMES paraphrases for the cue-sensitivity experiment")
    p.add_argument("--base", default=_DEFAULT_BASE, help="Neutral anchor JSONL (frames_benchmark.jsonl)")
    p.add_argument("--staleness-csv", default=_DEFAULT_STALENESS, help="Staleness CSV for non-stale filter")
    p.add_argument("--output-dir", default=_DEFAULT_OUTDIR, help="Directory for per-condition JSONL files")
    p.add_argument("--conditions", nargs="+", default=None,
                   help=f"Subset of conditions (default all). Valid: {ALL_CONDITIONS}")
    p.add_argument("--model", default="gpt-oss:120b", help="Rewriter model (default: gpt-oss:120b)")
    p.add_argument("--provider", default="ollama", help="Rewriter provider (default: ollama)")
    p.add_argument("--limit", type=int, default=None, help="Process at most N anchor questions")
    p.add_argument("--validate", action="store_true",
                   help="Audit each rewrite for equivalence + no-leak + cue-isolation and regenerate on failure")
    p.add_argument("--validate-model", default="gpt-oss:120b", help="Judge model (default: gpt-oss:120b)")
    p.add_argument("--validate-provider", default="ollama", help="Judge provider (default: ollama)")
    p.add_argument("--max-retries", type=int, default=3, help="Max regenerations per question (default: 3)")
    args = p.parse_args()

    asyncio.run(main(args))
