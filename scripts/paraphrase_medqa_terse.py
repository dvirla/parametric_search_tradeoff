"""
Paraphrase MedQA clinical vignettes into TERSE questions.

MedQA vignettes are verbose narrative clinical stems, e.g.:
  "A 44-year-old African-American woman comes to the physician for a routine
   examination. She is concerned about cancer because her uncle died of
   metastatic melanoma 1 year ago. She has no history of serious illness and
   does not take any medication. She has been working in a law firm for the
   past 20 years and travels to the Carribean regularly with her husband.
   Examination of her skin shows no abnormal moles or warts. This woman is at
   greatest risk of which of the following types of melanoma?"

To run the phrasing experiment on MedQA (mirroring the verbose/terse axis
already run on FRAMES via paraphrase_frames_benchmark.py) we need a TERSE
counterpart that strips narrative filler while preserving EVERY
answer-relevant clinical fact (labs, vitals, history, exam findings, timing):
  "44F, uncle died of metastatic melanoma 1y ago, no PMHx, no meds, no
   abnormal moles/warts on exam. Greatest risk for which melanoma subtype?"

The key failure mode is over-compression: dropping a diagnostic clue that
changes which option is correct. This reuses the same generate->judge->
regenerate validation loop as paraphrase_frames_benchmark.py /
paraphrase_musique_natural.py, via the new MedqaTerseAudit in
src/services/paraphrase_validation.py.

Usage:
  uv run python scripts/paraphrase_medqa_terse.py \\
      --input data/medqa_500.jsonl \\
      --output data/medqa_terse.jsonl \\
      --model gemini-3-flash-preview --provider Google \\
      --validate --validate-model gemini-3-flash-preview --validate-provider Google \\
      --max-retries 3

  # quick test on 5 rows:
  uv run python scripts/paraphrase_medqa_terse.py --limit 5 \\
      --model gemini-3-flash-preview --provider Google \\
      --validate --validate-provider Google --validate-model gemini-3-flash-preview
"""

import argparse
import asyncio
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.services.base_agent import BaseAgent

_DEFAULT_INPUT = "data/medqa_500.jsonl"
_DEFAULT_OUTPUT = "data/medqa_terse.jsonl"

SYSTEM_PROMPT = """\
You are a clinical-vignette editor. You are given a MedQA USMLE-style clinical \
vignette. Rewrite it in a TERSE form that strips ALL narrative filler while \
preserving EVERY answer-relevant clinical fact: lab values, vitals, past \
medical/surgical/family/social history, exam findings, medication/exposure \
history, and timing (onset, duration, sequence).

Rules:
1. Do NOT drop any detail needed to select the correct answer option. When in \
doubt, KEEP the detail — under-compression is fine, over-compression that \
removes a diagnostic clue is a failure.
2. Strip only narrative scene-setting, redundant phrasing, and filler clauses \
that carry no diagnostic weight (e.g. "comes to the physician for a routine \
examination", incidental occupation/hobby details never referenced by the \
answer options).
3. Keep the final question (what is being asked) intact and unambiguous.
4. Do NOT name or closely paraphrase the correct answer, or any of the four \
answer options, inside the rewritten stem.
5. Use compact clinical shorthand where natural (e.g. "44F" for a 44-year-old \
female, common abbreviations like PMHx, BP, HR) but stay unambiguous.
6. Return only the rewritten vignette text — no explanation, no commentary, \
no quotes, no restated options.

Example:
Original: "A 44-year-old African-American woman comes to the physician for a \
routine examination. She is concerned about cancer because her uncle died of \
metastatic melanoma 1 year ago. She has no history of serious illness and does \
not take any medication. She has been working in a law firm for the past 20 \
years and travels to the Carribean regularly with her husband. Examination of \
her skin shows no abnormal moles or warts. This woman is at greatest risk of \
which of the following types of melanoma?"
Terse: "44-year-old African-American woman, uncle died of metastatic melanoma \
1 year ago, no PMHx, no medications, skin exam shows no abnormal moles or \
warts. She is at greatest risk of which type of melanoma?"
"""

PARAPHRASE_TEMPLATE = """\
Original vignette: {question}

Answer options:
{options_block}

Correct option (keyed): {answer_idx}
Correct answer: {answer}

Rewrite this vignette in terse form following the rules. Do not reveal the correct option or answer text in your rewrite:"""


def format_options(options: dict) -> str:
    return "\n".join(f"{k}) {v}" for k, v in options.items())


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


def load_medqa_500(input_path: str) -> list[dict]:
    """Load the fixed MedQA-500 set (data/medqa_500.jsonl) into the internal
    {example_id, question, answer, options, answer_idx} format."""
    data = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            data.append({
                "example_id": row["example_id"],
                "question": row["problem"],
                "answer": row["gold answer"],
                "options": row["options"],
                "answer_idx": row["answer_idx"],
            })
    print(f"Loaded {len(data)} MedQA-500 questions from {input_path}")
    return data


async def paraphrase_one(agent: BaseAgent, item: dict, feedback: str = "") -> str:
    prompt = PARAPHRASE_TEMPLATE.format(
        question=item["question"],
        options_block=format_options(item["options"]),
        answer_idx=item["answer_idx"],
        answer=item["answer"],
    )
    if feedback:
        prompt += ("\n\nThe previous attempt was rejected by an auditor. Fix it:\n"
                   f"{feedback}\nReturn only the corrected rewritten vignette.")
    response = await agent.arun(prompt)
    return response.output.strip().strip('"').strip("'")


async def paraphrase_validated(agent: BaseAgent, judge_agent, item: dict,
                               max_retries: int) -> tuple[str, dict]:
    """Generate->judge->regenerate. Returns (best_text, validation_metadata).

    Keeps the first rewrite that passes the equivalence + fact-preservation +
    terse-register + no-leak audit. If none passes within max_retries+1
    attempts, returns the last attempt flagged failed so downstream can filter
    on validation_status while the example_id (and hence the 500-row pairing)
    is preserved.
    """
    from src.services.paraphrase_validation import audit_medqa_terse, medqa_terse_feedback

    feedback, last_text, last_audit = "", None, None
    for attempt in range(max_retries + 1):
        last_text = await paraphrase_one(agent, item, feedback)
        try:
            audit = await audit_medqa_terse(
                judge_agent, item["question"], last_text, item["options"],
                item["answer_idx"], item["answer"])
        except Exception as e:
            last_audit = None
            feedback = ""
            print(f"      [audit error attempt {attempt + 1}] {e}")
            continue
        last_audit = audit
        if audit.ok:
            return last_text, {
                "validation_status": "pass", "validation_attempts": attempt + 1,
                "equivalent": True, "facts_preserved": True, "is_terse_register": True,
                "leaks_answer": False, "dropped_facts": [], "drift": audit.drift,
            }
        feedback = medqa_terse_feedback(audit)

    if last_audit is None:
        meta = {
            "validation_status": "judge_error", "validation_attempts": max_retries + 1,
            "equivalent": None, "facts_preserved": None, "is_terse_register": None,
            "leaks_answer": None, "dropped_facts": None, "drift": None,
        }
    else:
        meta = {
            "validation_status": "fail", "validation_attempts": max_retries + 1,
            "equivalent": last_audit.equivalent, "facts_preserved": last_audit.facts_preserved,
            "is_terse_register": last_audit.is_terse_register, "leaks_answer": last_audit.leaks_answer,
            "dropped_facts": last_audit.dropped_facts, "drift": last_audit.drift,
        }
    return last_text, meta


async def main(model_name: str, provider_name: str, limit: int | None, input_file: str, output_file: str,
               validate: bool, validate_model: str, validate_provider: str, max_retries: int):
    data = load_medqa_500(input_file)
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

    agent = BaseAgent(provider_name=provider_name, model_name=model_name, system_prompt=SYSTEM_PROMPT)

    judge_agent = None
    if validate:
        from src.services.paraphrase_validation import make_judge_agent, MedqaTerseAudit
        print(f"Validation ON: equivalence + fact-preservation + terse-register + no-leak audit via "
              f"{validate_provider}/{validate_model}, up to {max_retries} regenerations.")
        judge_agent = make_judge_agent(validate_model, validate_provider, output_type=MedqaTerseAudit)

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
                "example_id": item["example_id"],
                "text": paraphrased,
                "answer": item["answer"],
                "options": item["options"],
                "answer_idx": item["answer_idx"],
                "original_question": item["question"],
                "source": "medqa_terse",
                **vmeta,
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

            tag = vmeta["validation_status"]
            print(f"[{i + 1}/{len(pending)}] ({tag}) {item['question'][:55]}")
            print(f"        -> {paraphrased[:80]}")

    print(f"\nDone. Output: {output_file}")
    if validate:
        print(f"Validation summary: {stats}")
        print("Note: 'fail'/'judge_error' rows are kept (flagged) so the example_id set "
              "stays paired with the 500-row set; filter on validation_status downstream.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paraphrase MedQA clinical vignettes into terse questions")
    parser.add_argument("--input", default=_DEFAULT_INPUT, help="Input MedQA-500 JSONL path")
    parser.add_argument("--model", default="gemini-3-flash-preview", help="Rewriter model (default: gemini-3-flash-preview)")
    parser.add_argument("--provider", default="Google", help="Rewriter provider (default: Google)")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N questions")
    parser.add_argument("--output", default=_DEFAULT_OUTPUT, help="Output JSONL path")
    parser.add_argument("--validate", action="store_true",
                        help="Audit each rewrite for equivalence/fact-preservation/terse-register/no-leak and regenerate on failure")
    parser.add_argument("--validate-model", default="gemini-3-flash-preview", help="Judge model (default: gemini-3-flash-preview)")
    parser.add_argument("--validate-provider", default="Google", help="Judge provider (default: Google)")
    parser.add_argument("--max-retries", type=int, default=3, help="Max regenerations per question (default: 3)")
    args = parser.parse_args()

    asyncio.run(main(
        model_name=args.model,
        provider_name=args.provider,
        limit=args.limit,
        input_file=args.input,
        output_file=args.output,
        validate=args.validate,
        validate_model=args.validate_model,
        validate_provider=args.validate_provider,
        max_retries=args.max_retries,
    ))
