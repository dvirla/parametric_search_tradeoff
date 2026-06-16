"""
Paraphrase FRAMES questions into terse, formal BENCHMARK-style questions.

FRAMES prompts are already written as naturalistic user questions, e.g.:
  "If my future wife has the same first name as the 15th first lady of the United
   States' mother and her surname is the same as the second assassinated
   president's mother's maiden name, what is my future wife's name?"

To run the within-dataset phrasing experiment (formal/benchmark vs natural/user,
same information need) we need the *formal* counterpart — the way a QA benchmark
like HotpotQA/MuSiQue would phrase it, with conversational framing stripped:
  "What is the name formed by the first name of the 15th U.S. first lady's mother
   and the maiden name of the second assassinated U.S. president's mother?"

This is the natural->benchmark direction (same as the ShareChat benchmark-style
paraphrase in the paper), so the audit checks EQUIVALENCE + BENCHMARK-STYLE
compliance rather than the bridge-hop leak used for MuSiQue-Natural. It reuses the
same generate->judge->regenerate validation loop as
scripts/paraphrase_musique_natural.py.

Usage:
  uv run python scripts/paraphrase_frames_benchmark.py \\
      --output data/frames_benchmark.jsonl \\
      --validate --max-retries 3

  # quick test against a cloud model instead of ollama:
  uv run python scripts/paraphrase_frames_benchmark.py --limit 5 \\
      --provider Google --model gemini-flash-latest \\
      --validate --validate-provider Google --validate-model gemini-flash-latest
"""

import argparse
import asyncio
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.services.base_agent import BaseAgent

_DEFAULT_OUTPUT = "data/frames_benchmark.jsonl"

SYSTEM_PROMPT = """\
You are a prompt rewriter. You are given a real, naturally-phrased user question. \
Rewrite it as a terse, formal multi-hop BENCHMARK question — the way a QA dataset \
like HotpotQA or MuSiQue would phrase it — while preserving the exact same \
information need and the same answer.

Rules:
1. Strip ALL conversational framing: no first person, no "I'm curious", "can you", \
"I've been wondering", greetings, hedging, or politeness. State the question directly.
2. Use compact, formal phrasing with nested relative clauses that chain the \
reasoning (e.g. "the X of the Y that Z"). The compositional structure may be explicit.
3. Do NOT add, drop, or alter any constraint, entity, date, or number. The rewrite \
must resolve to the SAME answer as the original.
4. Produce a single self-contained question (one sentence where possible). No \
explanation, no context, no preamble.
5. Return only the rewritten question text — no quotes, no commentary.

Example:
User: "Can you name any times in history where a military had the chance to concede \
peace, but chose war to reclaim lost territory and ended up losing the entire country?"
Benchmark: "Which country lost its entire national territory after its military chose \
war to reclaim lost land instead of conceding peace?"
"""

PARAPHRASE_TEMPLATE = """\
Natural user question: {question}

Expected answer: {answer}

Rewrite this as a terse, formal benchmark-style question following the rules:"""


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


def load_frames() -> list[dict]:
    """Load FRAMES (test split) into the internal {example_id, question, answer} format."""
    from datasets import load_dataset
    print("Loading FRAMES from HuggingFace...")
    df = load_dataset("google/frames-benchmark", split="test").to_pandas()
    data = []
    for idx, row in df.iterrows():
        data.append({
            "example_id": str(idx),
            "question": row["Prompt"],
            "answer": str(row["Answer"]),
            "reasoning_types": row.get("reasoning_types"),
        })
    print(f"Loaded {len(data)} FRAMES questions")
    return data


async def paraphrase_one(agent: BaseAgent, item: dict, feedback: str = "") -> str:
    prompt = PARAPHRASE_TEMPLATE.format(question=item["question"], answer=item["answer"])
    if feedback:
        prompt += ("\n\nThe previous attempt was rejected by an auditor. Fix it:\n"
                   f"{feedback}\nReturn only the corrected rewritten question.")
    response = await agent.arun(prompt)
    return response.output.strip().strip('"').strip("'")


async def paraphrase_validated(agent: BaseAgent, judge_agent, item: dict,
                               max_retries: int) -> tuple[str, dict]:
    """Generate→judge→regenerate. Returns (best_text, validation_metadata).

    Keeps the first rewrite that passes the equivalence + benchmark-style audit. If
    none passes within max_retries+1 attempts, returns the last attempt flagged
    failed so downstream can drop it (the example_id is preserved for pairing).
    """
    from src.services.paraphrase_validation import audit_benchmark_style, benchmark_style_feedback

    feedback, last_text, last_audit = "", None, None
    for attempt in range(max_retries + 1):
        last_text = await paraphrase_one(agent, item, feedback)
        try:
            audit = await audit_benchmark_style(
                judge_agent, item["question"], last_text, item["answer"])
        except Exception as e:
            last_audit = None
            feedback = ""
            print(f"      [audit error attempt {attempt + 1}] {e}")
            continue
        last_audit = audit
        if audit.ok:
            return last_text, {"validation_status": "pass", "validation_attempts": attempt + 1,
                               "equivalent": True, "is_benchmark_style": True,
                               "drift": audit.drift, "style_issues": []}
        feedback = benchmark_style_feedback(audit)

    if last_audit is None:
        meta = {"validation_status": "judge_error", "validation_attempts": max_retries + 1,
                "equivalent": None, "is_benchmark_style": None, "drift": None, "style_issues": None}
    else:
        meta = {"validation_status": "fail", "validation_attempts": max_retries + 1,
                "equivalent": last_audit.equivalent, "is_benchmark_style": last_audit.is_benchmark_style,
                "drift": last_audit.drift, "style_issues": last_audit.style_issues}
    return last_text, meta


async def main(model_name: str, provider_name: str, limit: int | None, output_file: str,
               validate: bool, validate_model: str, validate_provider: str, max_retries: int):
    data = load_frames()
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
        from src.services.paraphrase_validation import make_judge_agent, BenchmarkStyleAudit
        print(f"Validation ON: equivalence + benchmark-style audit via "
              f"{validate_provider}/{validate_model}, up to {max_retries} regenerations.")
        judge_agent = make_judge_agent(validate_model, validate_provider, output_type=BenchmarkStyleAudit)

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
                "answer": item["answer"],
                "example_id": item["example_id"],
                "original_question": item["question"],
                "reasoning_types": item.get("reasoning_types"),
                "source": "frames_benchmark",
                **vmeta,
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

            tag = vmeta["validation_status"]
            print(f"[{i + 1}/{len(pending)}] ({tag}) {item['question'][:55]}")
            print(f"        → {paraphrased[:80]}")

    print(f"\nDone. Output: {output_file}")
    if validate:
        print(f"Validation summary: {stats}")
        print("Note: 'fail'/'judge_error' rows are kept (flagged) so the example_id set "
              "stays paired; filter on validation_status downstream.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paraphrase FRAMES questions into benchmark-style questions")
    parser.add_argument("--model", default="gpt-oss:20b", help="Rewriter model (default: gpt-oss:20b)")
    parser.add_argument("--provider", default="ollama", help="Rewriter provider (default: ollama)")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N questions")
    parser.add_argument("--output", default=_DEFAULT_OUTPUT, help="Output JSONL path")
    parser.add_argument("--validate", action="store_true",
                        help="Audit each rewrite for equivalence + benchmark style and regenerate on failure")
    parser.add_argument("--validate-model", default="gpt-oss:120b", help="Judge model (default: gpt-oss:120b)")
    parser.add_argument("--validate-provider", default="ollama", help="Judge provider (default: ollama)")
    parser.add_argument("--max-retries", type=int, default=3, help="Max regenerations per question (default: 3)")
    args = parser.parse_args()

    asyncio.run(main(
        model_name=args.model,
        provider_name=args.provider,
        limit=args.limit,
        output_file=args.output,
        validate=args.validate,
        validate_model=args.validate_model,
        validate_provider=args.validate_provider,
        max_retries=args.max_retries,
    ))
