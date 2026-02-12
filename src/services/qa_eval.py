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
from src.services.entity_questions import (
    load_entity_questions, load_popqa, stratified_sample, exact_match_grade,
    any_match_grade, extract_answer_text, extract_explanation,
    ENTITY_QUESTIONS_QUERY_TEMPLATE, ENTITY_STYLE_DATASETS,
)
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
        self.dataset_name = dataset_name
        self.output_path = output_path
        self.resume_incomplete = resume_incomplete
        self.grader_template = custom_grader_template or STANDARD_GRADER_TEMPLATE

        # Load Dataset
        self.examples = self._load_dataset(dataset_name, dataset_path)

        if num_examples:
            assert n_repeats == 1, "n_repeats only supported when max_examples = None"
            sample_size = min(num_examples, len(self.examples))
            if dataset_name.lower() in ENTITY_STYLE_DATASETS:
                self.examples = stratified_sample(self.examples, sample_size)
            else:
                rng = random.Random(0)
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
            df = pd.read_csv(path)
            df = df.rename(columns={"query": "problem", "answer": "gold answer"})
        elif dataset_name.lower() == "facts-search":
            path = "data/facts/facts_classified.csv"
            df = pd.read_csv(path)
            df = df[df["classification"] == "one-hop"].reset_index(drop=True)
        elif dataset_name.lower() == "nq":
            path = "data/NQ-open.train.jsonl"
            df = pd.read_json(path, lines=True)
            df = df.rename(columns={"question": "problem", "answer": "gold answer"})
        elif dataset_name.lower() == "entity-questions":
            return load_entity_questions()
        elif dataset_name.lower() == "popqa":
            return load_popqa()

        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found at: {path}")

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
                    gold_answer = row.get("gold answer", "")
                    is_entity_q = self.dataset_name.lower() in ENTITY_STYLE_DATASETS

                    # Select prompt template
                    if is_entity_q:
                        query = ENTITY_QUESTIONS_QUERY_TEMPLATE.format(Question=problem)
                    else:
                        query = QUERY_TEMPLATE.format(Question=problem)

                    prompt_messages = [
                        sampler._pack_message(content=query, role="user")
                    ]

                    # Sampler
                    sampler_response = sampler(prompt_messages)
                    response1_text = sampler_response.response_text

                    # Grade the response
                    if is_entity_q:
                        answer_text = extract_answer_text(response1_text.output)
                        aliases = row.get("answer_aliases", [])
                        if aliases:
                            all_acceptable = list(gold_answer) + aliases
                            is_correct = any_match_grade(answer_text, all_acceptable)
                        else:
                            is_correct = exact_match_grade(answer_text, gold_answer)
                    else:
                        answer_text = str(response1_text.output)
                        grade_result = self.grade_sample(problem, str(gold_answer), answer_text)
                        is_correct = grade_result == "yes"

                    score = 1.0 if is_correct else 0.0
                    html = ""

                    metadata = sampler_response.response_metadata
                    metrics = {
                        "correct": is_correct,
                        "search_calls": metadata.get("search_calls", 0),
                    }

                    # For display/logging, use extracted text for entity-questions
                    display_response = extract_answer_text(response1_text.output) if is_entity_q else response1_text.output

                    convo = [
                        dict(content=f"Problem: {problem}", role="user"),
                        dict(content=f"Sampler response: {display_response}", role="assistant"),
                    ]

                    result = SingleEvalResult(html=html, score=score, convo=convo, metrics=metrics)
                    results.append(result)

                    result_entry = {
                        "problem": problem,
                        "correct_answer": gold_answer,
                        "sampler_response": extract_answer_text(response1_text.output) if is_entity_q else response1_text.output,
                        "sampler_correct": is_correct,
                        "sampler_search_calls": metadata.get("search_calls", 0),
                        "stop_reason": metadata.get("stop_reason"),
                        "metrics": metrics,
                    }
                    if is_entity_q:
                        result_entry["sampler_explanation"] = extract_explanation(response1_text.output)
                        source_file = row.get("source_file")
                        if source_file:
                            result_entry["source_file"] = source_file
                    
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