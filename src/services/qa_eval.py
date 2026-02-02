"This file is adapted from browsecomp_dual_eval.py to support the evaluation of agents on different datasets (Natural Quesions, FACTS)."

import json
import random
import re
import os
import time
import pandas as pd
from tqdm import tqdm
from src.services import common
from src.services.service_types import Eval, EvalResult, SamplerBase, SingleEvalResult
import httpx

# Template for agent responses
QUERY_TEMPLATE = """
{Question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}
""".strip()

# Standard Grader Template
STANDARD_GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|%| and 100|%| from [response]. Put 100 if there is no confidence score available.
""".strip()

class EvaluationService(Eval):
    def __init__(self, 
                 grader_model: SamplerBase, 
                 dataset_name: str = "facts",
                 dataset_path: str | None = None,
                 num_examples: int | None = None, 
                 n_repeats: int = 1, 
                 output_path: str | None = None, 
                 resume_incomplete: bool = False,
                 custom_grader_template: str | None = None):
        
        self.grader_model = grader_model
        self.output_path = output_path
        self.resume_incomplete = resume_incomplete
        self.grader_template = custom_grader_template or STANDARD_GRADER_TEMPLATE
        
        # Load Dataset
        self.examples = self._load_dataset(dataset_name, dataset_path)
        
        if num_examples:
            assert n_repeats == 1, "n_repeats only supported when max_examples = None"
            rng = random.Random(0)
            # Ensure we don't sample more than available
            sample_size = min(num_examples, len(self.examples))
            self.examples = rng.sample(self.examples, sample_size)

        self.examples = self.examples * n_repeats
        
        # Load existing results if output_path exists
        self.existing_results = []
        self.completed_problems = set()
        self._load_existing_results()

    def _load_dataset(self, dataset_name: str, dataset_path: str | None) -> list[dict]:
        """Loads the dataset based on name or path."""
        print(f"Loading dataset: {dataset_name}...")
        
        if dataset_path:
            path = dataset_path
        elif dataset_name.lower() == "facts-param":
            path = "data/facts/FACTS-Parametric-public.csv"
        elif dataset_name.lower() == "facts-search":
            path = "data/facts/FACTS-Search-public.csv"
        elif dataset_name.lower() == "nq":
            path = "data/nq/nq-dev-sample.csv"
            df = pd.read_json(path, lines=True)
            df = df.rename(columns={"question": "problem", "answer": "gold answer"})

        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found at: {path}")

        df = pd.read_csv(path)
        # Ensure standardized keys
        if "problem" not in df.columns and "question" in df.columns:
            df = df.rename(columns={"question": "problem"})
        if "gold answer" not in df.columns and "answer" in df.columns:
            df = df.rename(columns={"answer": "gold answer"})
            
        return df.to_dict('records')

    def _load_existing_results(self):
        if self.output_path and os.path.exists(self.output_path):
            try:
                with open(self.output_path, 'r') as f:
                    self.existing_results = json.load(f)
                    
                    for r in self.existing_results:
                        problem = r.get('problem')
                        if not problem:
                            continue
                            
                        # Check if we should consider this problem completed
                        is_completed = True
                        if self.resume_incomplete:
                            stop_reason = r.get('stop_reason')
                            search_calls = r.get('sampler_search_calls', 0)
                            
                            # Heuristic: If stop_reason is max_steps OR search_calls >= 4 (assuming old max was 4), treat as incomplete
                            if stop_reason == 'max_steps' or (stop_reason is None and search_calls >= 4):
                                is_completed = False
                        
                        if is_completed:
                            self.completed_problems.add(problem)
                            
                    print(f"Loaded {len(self.existing_results)} existing results from {self.output_path}")
                    print(f"Considering {len(self.completed_problems)} problems as completed.")
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load existing results from {self.output_path}: {e}")
                self.existing_results = []
                self.completed_problems = set()

    def grade_sample(self, question: str, correct_answer: str, response: str) -> str:
        grader_prompt = self.grader_template.format(
            question=question,
            correct_answer=correct_answer,
            response=response
        )

        prompt_messages = [
            self.grader_model._pack_message(content=grader_prompt, role="user")
        ]
        sampler_response = self.grader_model(prompt_messages)
        grading_response = sampler_response.response_text

        # Using simpler extraction logic or the model's output directly if it's structured
        # Assuming the grader model returns a string that contains "correct: yes" or "correct: no"
        match = re.search(r"correct:\s*(yes|no)", grading_response.output, re.IGNORECASE)
        return match.group(1).lower() if match else "no"

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        results = []
        
        final_json_results_map = {}
        for r in self.existing_results:
            if r['problem'] in self.completed_problems:
                final_json_results_map[r['problem']] = r
        
        json_results = list(final_json_results_map.values())

        for row in tqdm(self.examples):
            problem = row.get("problem", "")
            if problem in self.completed_problems:
                continue

            max_retries = 5
            retry_attempt = 0
            success = False
            
            while retry_attempt < max_retries and not success:
                try:
                    answer = str(row.get("gold answer", ""))
                    prompt_messages = [
                        sampler._pack_message(content=QUERY_TEMPLATE.format(Question=problem), role="user")
                    ]

                    # Sampler
                    sampler_response = sampler(prompt_messages)
                    response1_text = sampler_response.response_text
                    
                    # Grade the response
                    grade_result = self.grade_sample(problem, answer, response1_text.output)
                    is_correct = grade_result == "yes"

                    score = 1.0 if is_correct else 0.0
                    html = "" 
                    
                    metadata = sampler_response.response_metadata
                    metrics = {
                        "correct": is_correct,
                        "search_calls": metadata.get("search_calls", 0),
                    }

                    convo = [
                        dict(content=f"Problem: {problem}", role="user"),
                        dict(content=f"Sampler response: {response1_text.output}", role="assistant"),
                    ]

                    result = SingleEvalResult(html=html, score=score, convo=convo, metrics=metrics)
                    results.append(result)

                    result_entry = {
                        "problem": problem,
                        "correct_answer": answer,
                        "sampler_response": response1_text.output,
                        "sampler_correct": is_correct,
                        "sampler_search_calls": metadata.get("search_calls", 0),
                        "stop_reason": metadata.get("stop_reason"),
                        "metrics": metrics # store all other metrics
                    }
                    
                    # Update map and list
                    final_json_results_map[problem] = result_entry
                    json_results = list(final_json_results_map.values())
                    
                    if self.output_path:
                        with open(self.output_path, "w") as f:
                            json.dump(json_results, f, indent=4)
                    
                    success = True
                    
                except (httpx.ConnectError, httpx.RemoteProtocolError, ConnectionError) as e:
                    if retry_attempt < max_retries - 1:
                        wait_time = 2 ** retry_attempt
                        print(f"\nNetwork error on attempt {retry_attempt + 1}/{max_retries}: {e}")
                        print(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        retry_attempt += 1
                    else:
                        print(f"\nFailed after {max_retries} attempts, skipping this example")
                        retry_attempt += 1

        # Add back results from already completed problems (that we didn't re-run)
        all_eval_results = list(results)
        for p, existing in final_json_results_map.items():
            if any(r.convo[0]['content'] == f"Problem: {p}" for r in results):
                continue
            
            score = 1.0 if existing.get("sampler_correct", False) else 0.0
            metrics = {
                "correct": existing.get("sampler_correct", False),
                "search_calls": existing.get("sampler_search_calls", 0),
            }
            convo = [
                dict(content=f"Problem: {existing.get('problem', '')}", role="user"),
                dict(content=f"Sampler response: {existing.get('sampler_response', '')}", role="assistant"),
            ]
            all_eval_results.append(SingleEvalResult(html="", score=score, convo=convo, metrics=metrics))

        return common.aggregate_results(all_eval_results)