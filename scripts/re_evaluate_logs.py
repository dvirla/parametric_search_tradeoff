import os
import sys
import json
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

def grade_sample(grader_model, question, correct_answer, response_text):
    grader_prompt = NEW_GRADER_TEMPLATE.format(
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

    match = re.search("contained\"\s*:\s*\"([^\"]+)", grading_output, re.IGNORECASE)
    return match.group(1).lower() if match else "no"

def main():
    parser = argparse.ArgumentParser(description="Re-evaluate existing log files.")
    parser.add_argument("input_files", nargs='+', help="Path to JSON log files.")
    parser.add_argument("--output_suffix", type=str, default="_reevaluated", help="Suffix for re-evaluated files.")
    parser.add_argument("--grader_provider", type=str, default="ollama", help="Grader provider.")
    parser.add_argument("--grader_model", type=str, default="gpt-oss:20b", help="Grader model name.")
    parser.add_argument("--inplace", action="store_true", help="Overwrite the input files.")
    args = parser.parse_args()

    print(f"Initializing Grader Agent ({args.grader_provider}/{args.grader_model})...")
    grader_agent_raw = BaseAgent(provider_name=args.grader_provider, model_name=args.grader_model, agent_name="grader_agent")
    grader_agent = AgentAsSampler(grader_agent_raw)

    for file_path in args.input_files:
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
        changes = 0
        improvements = 0
        regressions = 0

        for entry in tqdm(data):
            old_correct = entry.get("sampler_correct", False)
            if old_correct:
                correct_before += 1
                continue

            # Re-grade
            try:
                # Some logs might have different keys, but based on the sample it is 'problem', 'correct_answer', 'sampler_response'
                problem = entry.get("problem")
                correct_answer = entry.get("correct_answer")
                sampler_response = entry.get("sampler_response")
                sampler_explanation = entry.get("sampler_explanation")
                sampler_response = f'Explanation: {sampler_explanation}\n\nAnswer: {sampler_response}' if sampler_explanation else sampler_response
                
                if not all([problem, correct_answer, sampler_response]):
                    print(f"Skipping entry due to missing fields: {entry.keys()}")
                    continue

                grade_result = grade_sample(
                    grader_agent, 
                    problem, 
                    correct_answer, 
                    sampler_response
                )
            except Exception as e:
                print(f"Error grading problem '{entry.get('problem', 'unknown')[:50]}...': {e}")
                continue

            new_correct = (grade_result == "yes")
            
            if new_correct:
                correct_after += 1
            
            if old_correct != new_correct:
                changes += 1
                if new_correct:
                    improvements += 1
                else:
                    regressions += 1

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

        print(f"Results for {file_path}:")
        print(f"  Correct before: {correct_before}")
        print(f"  Correct after:  {correct_after}")
        print(f"  Delta:          {correct_after - correct_before:+d}")
        print(f"  Total changes:  {changes} (Improvements: {improvements}, Regressions: {regressions})")
        print(f"  Saved to: {output_path}")

if __name__ == "__main__":
    main()
