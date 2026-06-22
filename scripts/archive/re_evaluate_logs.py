import os
import sys
import json
import glob
import argparse
import re
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.services.base_agent import BaseAgent
from src.services.agent_sampler import AgentAsSampler
from src.services.qa_eval import STANDARD_GRADER_TEMPLATE

NEW_GRADER_TEMPLATE = """
You are an explanation containment judge.
Given **[question]**, **[response]**, and **[correct_answer]**, decide whether the *correct_answer* is explicitly or implicitly contained within the *response*.

1. **Extract the final term** - Identify the last definitive statement in *response* that could represent an answer. Remove any surrounding formatting (e.g., `**`, `*`, backticks, LaTeX `\boxed{{…}}`, etc.) and trim whitespace. If no such statement exists, set **extracted_final_answer = "None"**.

2. **Check for containment**
   a. Normalize both *response* and *correct_answer* by lower-casing, applying Unicode-NFKC normalization, and stripping leading/trailing spaces.
   b. Consider synonyms, paraphrasing, and logical equivalence. For example, "Paris is the capital of France" contains "Paris" as the answer to "What is the capital of France?".
   c. If *correct_answer* is explicitly stated or logically implied in *response*, mark it as contained.

3. **Output** exactly the following lines:

```
"extracted_final_answer": "<term or None>"
"contained": "yes" | "no"
"reasoning": "<concise factual explanation of containment or lack thereof>"
"confidence": "<percentage>%"
```

*Constraints*  
- `extracted_final_answer` must be a plain string (or the literal text `None`).  
- `contained` may only contain **yes** or **no**.  
- `reasoning` must explain why the *correct_answer* is or is not contained in *response*. Avoid extra context.  
- `confidence` is a number followed optionally by `%`. If omitted, default to **100%**.

### Current Evaluation Task

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}
"""

# Maps each selectable template to (template_string, regex extracting the yes/no verdict).
TEMPLATES = {
    # Same template the production eval used (qa_eval.STANDARD_GRADER_TEMPLATE). It asks the
    # grader to EXTRACT a discrete final answer — appropriate for terse answers, but it
    # systematically under-counts free-form prose where the gold is embedded, not stated.
    "standard": (STANDARD_GRADER_TEMPLATE, r"correct:\s*(yes|no)"),
    # Containment rubric: "is the correct answer explicitly OR IMPLICITLY contained in the
    # response". This is the right grader for natural-phrasing logs, whose answers are
    # conversational prose. 'natural' is an alias to make the intent explicit at the CLI.
    "containment": (NEW_GRADER_TEMPLATE, r"contained\"\s*:\s*\"([^\"]+)"),
    "natural": (NEW_GRADER_TEMPLATE, r"contained\"\s*:\s*\"([^\"]+)"),
}


def grade_sample(grader_model, question, correct_answer, response_text, template, verdict_re):
    grader_prompt = template.format(
        question=question,
        correct_answer=correct_answer,
        response=response_text
    )

    prompt_messages = [
        grader_model._pack_message(content=grader_prompt, role="user")
    ]
    sampler_response = grader_model(prompt_messages)
    # sampler_response.response_text is the pydantic-ai RunResult
    grading_output = sampler_response.response_text.output

    match = re.search(verdict_re, grading_output, re.IGNORECASE)
    return match.group(1).lower() if match else "no"

def main():
    parser = argparse.ArgumentParser(description="Re-evaluate (audit) existing eval log files with a different grader model/template.")
    parser.add_argument("input_files", nargs='+', help="Path(s) or globs to JSON log files.")
    parser.add_argument("--output_suffix", type=str, default="_reevaluated", help="Suffix for re-evaluated files.")
    parser.add_argument("--grader_provider", type=str, default="ollama", help="Grader provider.")
    parser.add_argument("--grader_model", type=str, default="gpt-oss:120b", help="Grader model name.")
    parser.add_argument("--template", choices=list(TEMPLATES), default="natural",
                        help="Grading rubric. 'natural'/'containment' check whether the gold is contained in free-form prose (right for natural-phrasing logs); 'standard' matches the production extract-final-answer rubric.")
    parser.add_argument("--only-wrong", action="store_true",
                        help="Only re-grade entries previously marked wrong (legacy one-directional behaviour). Default is a full bidirectional re-grade.")
    parser.add_argument("--inplace", action="store_true", help="Overwrite the input files.")
    args = parser.parse_args()

    template, verdict_re = TEMPLATES[args.template]

    # Expand globs / dedup while preserving order.
    files = []
    for pat in args.input_files:
        matched = sorted(glob.glob(pat)) or [pat]
        for f in matched:
            if f not in files:
                files.append(f)

    print(f"Initializing Grader Agent ({args.grader_provider}/{args.grader_model}), template='{args.template}', "
          f"mode={'only-wrong' if args.only_wrong else 'bidirectional'}...")
    grader_agent_raw = BaseAgent(provider_name=args.grader_provider, model_name=args.grader_model, agent_name="grader_agent")
    grader_agent = AgentAsSampler(grader_agent_raw)

    for file_path in files:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        print(f"\nProcessing {file_path}...")
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

        correct_before = 0
        correct_after = 0
        # Confusion of old vs new verdict (only over entries we actually re-graded).
        tp = fp = fn = tn = 0   # tp: stayed correct, fp: was correct->now wrong, fn: was wrong->now correct, tn: stayed wrong
        skipped = 0

        for entry in tqdm(data):
            old_correct = entry.get("sampler_correct", False)
            if old_correct:
                correct_before += 1

            # Legacy one-directional mode: trust prior 'correct' verdicts, only re-check wrongs.
            if args.only_wrong and old_correct:
                correct_after += 1
                continue

            try:
                problem = entry.get("problem")
                correct_answer = entry.get("correct_answer")
                sampler_response = entry.get("sampler_response")
                sampler_explanation = entry.get("sampler_explanation")
                sampler_response = f'Explanation: {sampler_explanation}\n\nAnswer: {sampler_response}' if sampler_explanation else sampler_response

                if not all([problem, correct_answer, sampler_response]):
                    print(f"Skipping entry due to missing fields: {list(entry.keys())}")
                    skipped += 1
                    # Preserve the prior verdict for skipped entries.
                    if old_correct:
                        correct_after += 1
                    continue

                grade_result = grade_sample(
                    grader_agent, problem, correct_answer, sampler_response, template, verdict_re
                )
            except Exception as e:
                print(f"Error grading problem '{str(entry.get('problem', 'unknown'))[:50]}...': {e}")
                skipped += 1
                if old_correct:
                    correct_after += 1
                continue

            new_correct = (grade_result == "yes")
            if new_correct:
                correct_after += 1

            if old_correct and new_correct:
                tp += 1
            elif old_correct and not new_correct:
                fp += 1
            elif (not old_correct) and new_correct:
                fn += 1
            else:
                tn += 1

            # Update entry
            entry["sampler_correct"] = new_correct
            if "metrics" in entry:
                entry["metrics"]["correct"] = new_correct

        # Save result
        if args.inplace:
            output_path = file_path
        else:
            base, ext = os.path.splitext(file_path)
            output_path = f"{base}{args.output_suffix}{ext}"

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4)

        n = len(data)
        print(f"Results for {file_path}:")
        print(f"  N total:        {n}")
        print(f"  Correct before: {correct_before}  (acc {correct_before/n:.3f})" if n else "  Correct before: 0")
        print(f"  Correct after:  {correct_after}  (acc {correct_after/n:.3f})" if n else "  Correct after: 0")
        print(f"  Delta:          {correct_after - correct_before:+d}")
        if not args.only_wrong:
            print(f"  Flips:          old✓→new✗ (false positives fixed): {fp}   old✗→new✓ (false negatives fixed): {fn}")
            print(f"  Confusion:      stay✓={tp} stay✗={tn} ✓→✗={fp} ✗→✓={fn}")
        else:
            print(f"  (only-wrong mode) ✗→✓ recovered: {fn}")
        if skipped:
            print(f"  Skipped (errors/missing): {skipped}")
        print(f"  Saved to: {output_path}")

if __name__ == "__main__":
    main()
