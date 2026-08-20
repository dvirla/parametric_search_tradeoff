"This file is adapted from browsecomp_dual_eval.py to support the evaluation of agents on different datasets (Natural Quesions, FACTS)."

import asyncio
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
from pydantic_ai.exceptions import ModelHTTPError, ModelAPIError, UnexpectedModelBehavior

# Template for agent responses
QUERY_TEMPLATE = """
{Question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}
""".strip()

PLAIN_QUERY_TEMPLATE = "{Question}"
NATURAL_QUERY_TEMPLATE = """{Question}

Please answer in 2-4 sentences."""
# Long extreme of the output-length axis (opposite end from NATURAL's "2-4 sentences"). Used via
# --query_template elaborate to test whether a detail/length output cue pushes search the other way.
ELABORATE_QUERY_TEMPLATE = """{Question}

Please answer with a detailed explanation - at least 8-10 sentences"""
# Politeness axis. Wraps the question in extreme, deferential politeness while carrying NO
# output-length or output-format directive (unlike NATURAL/ELABORATE). This isolates politeness
# itself, to stress-test whether a conversational/polite "please help me" framing — not the
# length directive — is what drives the reduced search seen for NATURAL/ELABORATE. Used via
# --query_template polite.
POLITE_QUERY_TEMPLATE = """{Question}

If it isn't too much trouble, I would be ever so grateful if you could very kindly help me with this. Please don't go to any bother on my account — I truly appreciate your time and effort, and thank you so very much in advance for your help."""
# Answer-directive axis, maximized. Stress-tests the hypothesis that the search reduction from
# NATURAL/ELABORATE comes from the *answer directive* (commit-to-answering), NOT the requested
# length (ELABORATE≈NATURAL ⇒ length null) and NOT politeness (POLITE≠NATURAL, POLITE doesn't
# reduce). DIRECT is a pure, maximal commit-to-answer output directive: NO sentence-count/length,
# NO politeness softeners, NO epistemic/confidence words, and it constrains the OUTPUT (the answer)
# not the PROCESS (never "don't search"). Used via --query_template direct.
DIRECT_QUERY_TEMPLATE = """{Question}

Just answer the question directly — final answer only."""
# Explicit capability-framing cue: tells the model directly that it already has the knowledge and
# doesn't need to search, without disabling the search tool. The most literal, direct test of
# whether an explicit "no need to search" instruction suppresses tool use, vs. the indirect cues
# above (NATURAL/ELABORATE/POLITE/DIRECT all shift search behavior without saying so explicitly).
# Used via --query_template confident_parametric.
CONFIDENT_PARAMETRIC_QUERY_TEMPLATE = """{Question}

You already have the relevant knowledge to answer this question accurately from your training. You do not need to use the search tool — though it remains available if you want it."""
# `frames-cues` is the prompt-cue sensitivity experiment: each variant injects ONE linguistic cue
# (incl. the output-constraint cue) into the question text, so the question text we craft must be
# the ENTIRE prompt. It must use the plain passthrough template — NATURAL_QUERY_TEMPLATE would
# append "Please answer in 2-4 sentences" (its own output-constraint cue) and the default
# QUERY_TEMPLATE forces an Explanation/Exact-Answer/Confidence format; both would contaminate the
# manipulation.
PLAIN_QUERY_DATASETS = {"sharechat", "sharechat-benchmark", "curated-sharechat-benchmark", "frames-cues"}
# Naturalistic / real-user phrasings get the conversational template. `frames`
# (native FRAMES prompts) and `curated-sharechat` are real-user style, so they live
# here; their formal counterparts (`frames-benchmark`, `curated-sharechat-benchmark`)
# use the default benchmark template. NOTE: PLAIN_QUERY_DATASETS is matched first,
# so curated-sharechat must NOT also appear there.
NATURAL_QUERY_DATASETS = {"musique-natural", "musique-natural2", "frames", "curated-sharechat"}

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

# MedQA-specific grader. The four options were HIDDEN from the responder (question-only,
# open-ended-with-search). To avoid marking a clinically-defensible-but-differently-phrased
# answer wrong, the grader is shown the original options and the keyed-correct one, and must
# map the response's single committed final answer onto an option — scored correct ONLY if it
# maps to the keyed option. Extracting a single final answer first neutralises the length
# confound (an ELABORATE answer must not score by merely name-dropping the gold term).
MEDQA_GRADER_TEMPLATE = """
Judge whether the following open-ended [response] to a medical [question] corresponds to the intended correct answer.

The question was originally multiple-choice, but the options were HIDDEN from the responder — they answered free-form. Below are the original options and which one is keyed as correct. Decide which single option the [response]'s final answer corresponds to (or NONE), and mark it correct ONLY if it corresponds to the keyed-correct option. Match on clinical/semantic equivalence (synonyms, equivalent terminology), not surface wording.

[question]: {question}

[options]:
{options}

[keyed_correct_answer]: {correct_answer}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The single final answer the [response] commits to (a diagnosis / finding / choice). If the response hedges across several candidates, pick the one it most endorses as its final answer. Put 'None' if there is no identifiable final answer.

mapped_option: Which option letter (A/B/C/D) does extracted_final_answer correspond to, judged by clinical equivalence? Answer just the letter, or 'None' if it corresponds to none of the listed options.

reasoning: Briefly justify the mapping, focusing only on whether extracted_final_answer is clinically equivalent to the keyed-correct answer versus a different option.

correct: Answer 'yes' if mapped_option is the keyed-correct option; 'no' otherwise (including if it maps to a distractor or to None).

confidence: 100.
""".strip()


def _format_medqa_options(options, answer_idx=None) -> str | None:
    """Render a MedQA option dict/list as 'A) text' lines for the options-aware grader.

    Returns None when no options are available (e.g. non-MedQA datasets), which routes
    grading back to the standard grader template.
    """
    if options is None:
        return None
    try:
        if hasattr(options, "items"):
            items = list(options.items())
        elif isinstance(options, (list, tuple)):
            items = [(chr(65 + i), v) for i, v in enumerate(options)]
        else:
            return None
    except Exception:
        return None
    if not items:
        return None
    lines = [f"{k}) {v}" for k, v in items]
    if answer_idx is not None and str(answer_idx).strip():
        lines.append(f"(keyed-correct option: {answer_idx})")
    return "\n".join(lines)

class EvaluationService(Eval):
    def __init__(self,
                 grader_model: SamplerBase,
                 dataset_name: str = "facts",
                 dataset_path: str | None = None,
                 num_examples: int | None = None,
                 n_repeats: int = 1,
                 output_path: str | None = None,
                 resume_incomplete: bool = False,
                 custom_grader_template: str | None = None,
                 split: str = "dev",
                 relations: list[str] | None = None,
                 seed: int = 0,
                 num_workers: int = 1,
                 query_template_override: str | None = None,
                 history_path: str | None = None,
                 retry_model_api_errors: bool = True):

        self.grader_model = grader_model
        self.dataset_name = dataset_name
        self.output_path = output_path
        self.resume_incomplete = resume_incomplete
        self.grader_template = custom_grader_template or STANDARD_GRADER_TEMPLATE
        self.num_workers = num_workers
        # Default ON: a raw ModelAPIError (e.g. a provider request timeout) is treated as
        # transient and retried with backoff, same as ModelHTTPError. Without this, a single
        # timed-out example crashes the whole asyncio.gather() batch -- observed on 122B/120B-class
        # models under load, where it silently zeroed out an entire run despite the grid driver's
        # retry-a-fresh-pass wrapper reporting the job as done. Set False to revert to the old
        # (pre-fix) behavior of letting it propagate and crash the batch -- e.g. to A/B the fix
        # itself, or if the retry-and-skip masking turns out to hide a real problem.
        self.retry_model_api_errors = retry_model_api_errors
        # Optional explicit query-template choice that overrides the dataset-based routing.
        # Useful for controlled template experiments (e.g. running the same FRAMES text under
        # PLAIN vs NATURAL vs the structured QUERY_TEMPLATE while holding phrasing fixed).
        self.query_template_override = query_template_override
        # Optional conversation prefix prepended before the actual question. `history_path` is
        # either a single fixed conversation (list of turn dicts, e.g. an unrelated chit-chat
        # exchange -- same history used for every example) or a POOL of conversations (list of
        # lists, e.g. mocked-search exchanges) -- one is picked per example via a seed keyed on
        # that example's id, so the choice is stable across --resume.
        self.history_pool = None
        if history_path:
            loaded = json.load(open(history_path))
            self.history_pool = loaded if (loaded and isinstance(loaded[0], list)) else [loaded]

        # Load Dataset
        self.examples = self._load_dataset(dataset_name, dataset_path, split, relations)

        if num_examples:
            assert n_repeats == 1, "n_repeats only supported when max_examples = None"
            sample_size = min(num_examples, len(self.examples))
            if dataset_name.lower() in ENTITY_STYLE_DATASETS:
                self.examples = stratified_sample(self.examples, sample_size, seed=seed)
            else:
                rng = random.Random(seed)
                self.examples = rng.sample(self.examples, sample_size)

        self.examples = self.examples * n_repeats
        
        # Load existing results if output_path exists
        self.existing_results = []
        self.completed_problems = set()
        self._load_existing_results()

    def _load_dataset(self, dataset_name: str, dataset_path: str | None,
                      split: str = "dev", relations: list[str] | None = None) -> list[dict]:
        """Loads the dataset based on name or path."""
        print(f"Loading dataset: {dataset_name}...")

        # Datasets whose own branch already honors `dataset_path or <default>` (and applies the
        # right column rename) must NOT be intercepted by the generic path branch below — let them
        # fall through to their loader so the JSONL schema is handled correctly.
        _self_path_datasets = {"frames-benchmark", "frames-cues", "musique-natural", "musique-natural2", "facts-open", "medqa-500", "medqa-terse"}
        if dataset_path and dataset_name.lower() not in _self_path_datasets:
            path = dataset_path
        elif dataset_name.lower() == "facts-param":
            path = "data/facts/FACTS-Parametric-public.csv"
            df = pd.read_csv(path)
            df = df.rename(columns={"query": "problem", "answer": "gold answer"})
        elif dataset_name.lower() == "medqa":
            # MedQA-USMLE (GBaker/MedQA-USMLE-4-options), test split. We deliberately
            # present ONLY the question — the four options are dropped so the model must
            # answer open-ended with search — and grade the free-form response against the
            # gold answer TEXT (the `answer` field, not the letter). No local file; load
            # from HuggingFace and return early to bypass the path-existence check.
            from datasets import load_dataset
            mdf = load_dataset("GBaker/MedQA-USMLE-4-options", split="test").to_pandas()
            mdf = mdf.rename(columns={"question": "problem", "answer": "gold answer"})
            mdf["example_id"] = ["medqa_test_%04d" % i for i in range(len(mdf))]
            # Keep options + answer_idx OUT of the prompt (only `problem` is shown to the
            # model) but carry them through so the MedQA-specific grader can map the model's
            # free-form answer onto the keyed option vs. the distractors.
            return mdf[["example_id", "problem", "gold answer", "options", "answer_idx"]].to_dict("records")
        elif dataset_name.lower() == "medqa-500":
            # Fixed 500-question MedQA-USMLE set (materialized by build_medqa_500,
            # data/medqa_500.jsonl) — a strict superset of the 200 example_ids used by the
            # completed plain/elaborate cue runs (results/medqa_cue_200/). "Verbose"/original
            # phrasing condition of the MedQA verbose-vs-terse phrasing axis; paired by
            # example_id with medqa-terse below. Already in canonical
            # {example_id, problem, gold answer, options, answer_idx} schema — no rename.
            path = dataset_path or "data/medqa_500.jsonl"
            df = pd.read_json(path, lines=True)
        elif dataset_name.lower() == "medqa-terse":
            # Terse paraphrase of the MedQA-500 vignettes (built by
            # scripts/paraphrase_medqa_terse.py) — the "terse" condition of the MedQA
            # verbose-vs-terse phrasing axis, paired by example_id with medqa-500. Keep all
            # rows (incl. flagged validation failures) so the example_id set stays paired;
            # filter on validation_status downstream. Same options-aware grading as medqa/medqa-500.
            path = dataset_path or "data/medqa_terse.jsonl"
            df = pd.read_json(path, lines=True)
            df = df.rename(columns={"text": "problem", "answer": "gold answer"})
        elif dataset_name.lower() == "facts-open":
            # Open-ended multi-hop facts questions (data/facts/facts_open_filtered.csv),
            # already in canonical {example_id, problem, gold answer} schema — no rename.
            # Used for the PLAIN-vs-cue query-template sensitivity smoke test.
            path = dataset_path or "data/facts/facts_open_filtered.csv"
            df = pd.read_csv(path)
        elif dataset_name.lower() == "facts-search":
            path = "data/facts/facts_classified.csv"
            df = pd.read_csv(path)
            df = df[df["classification"] == "one-hop"].reset_index(drop=True)
        elif dataset_name.lower() == "nq":
            path = "data/NQ-open.train.jsonl"
            df = pd.read_json(path, lines=True)
            df = df.rename(columns={"question": "problem", "answer": "gold answer"})
        elif dataset_name.lower() == "entity-questions":
            return load_entity_questions(split=split, relations=relations)
        elif dataset_name.lower() == "popqa":
            return load_popqa()
        elif dataset_name.lower() == "sharechat":
            path = "data/sharechat_info_seeking_v3.jsonl"
            df = pd.read_json(path, lines=True)
            df = df[df['is_info_seeking'].astype(bool) & (df["reasoning_hops"] > 1) & ~df['is_time_dependent'].astype(bool)].reset_index(drop=True)
            df = df.rename(columns={"text": "problem"})
        elif dataset_name.lower() == "sharechat-benchmark":
            path = "data/sharechat_benchmark.jsonl"
            df = pd.read_json(path, lines=True)
            df = df[df['is_info_seeking'].astype(bool) & (df["reasoning_hops"] > 1) & ~df['is_time_dependent'].astype(bool)].reset_index(drop=True)
            df = df.rename(columns={"benchmark_question": "problem"})
        elif dataset_name.lower() == "curated-sharechat":
            path = "data/curated_sharechat_wildchat.csv"
            df = pd.read_csv(path)
            df = df.rename(columns={"text": "problem"})
        elif dataset_name.lower() == "curated-sharechat-benchmark":
            path = "data/curated_sharechat_wildchat_benchmark.csv"
            df = pd.read_csv(path)
            df = df.rename(columns={"benchmark_question": "problem"})
        elif dataset_name.lower() == "musique-natural":
            path = dataset_path or "data/musique_val_natural.jsonl"
            df = pd.read_json(path, lines=True)
            df = df.rename(columns={"text": "problem", "answer": "gold answer"})
        elif dataset_name.lower() == "musique-natural2":
            # 2nd MuSiQue phrasing (gpt-oss:120b paraphraser) over the same 600
            # val ids — for the phrasing-invariance experiment. Same schema as
            # musique-natural.
            path = dataset_path or "data/musique_val_natural2.jsonl"
            df = pd.read_json(path, lines=True)
            df = df.rename(columns={"text": "problem", "answer": "gold answer"})
        elif dataset_name.lower() == "expertqa":
            path = "data/expertqa_sample.csv"
            df = pd.read_csv(path)
            df = df.rename(columns={"question": "problem"})
        elif dataset_name.lower() == "frames":
            # FRAMES multi-hop benchmark (google/frames-benchmark): 824-question
            # test split with naturalistic prose questions. No gold hop
            # decomposition, so it supports baseline accuracy + search-volume
            # experiments only (not the per-hop E/CP/PR/M taxonomy). Loaded from
            # HuggingFace; return early to bypass the local-path existence check.
            from datasets import load_dataset
            df = load_dataset("google/frames-benchmark", split="test").to_pandas()
            df = df.rename(columns={"Prompt": "problem", "Answer": "gold answer"})
            return df.to_dict("records")
        elif dataset_name.lower() == "frames-benchmark":
            # Formal/benchmark-style paraphrases of FRAMES (built by
            # scripts/paraphrase_frames_benchmark.py) — the "formal" condition,
            # paired by example_id with the natural `frames` prompts. Keep all rows
            # (incl. flagged validation failures) so the example_id set stays paired;
            # filter on validation_status downstream.
            path = dataset_path or "data/frames_benchmark.jsonl"
            df = pd.read_json(path, lines=True)
            df = df.rename(columns={"text": "problem", "answer": "gold answer"})
        elif dataset_name.lower() == "frames-cues":
            # Prompt-cue sensitivity experiment: one of the per-condition files produced by
            # scripts/paraphrase_frames_cues.py (data/frames_cues/<slug>.jsonl). Select the
            # specific condition via dataset_path. Same {text, answer, example_id} schema as
            # frames-benchmark; uses the PLAIN passthrough template (see PLAIN_QUERY_DATASETS).
            path = dataset_path or "data/frames_cues/neutral.jsonl"
            df = pd.read_json(path, lines=True)
            df = df.rename(columns={"text": "problem", "answer": "gold answer"})
        elif dataset_name.lower() == "hotpotqa":
            # HotpotQA distractor config (hotpotqa/hotpot_qa), validation split (7405
            # examples -- the test split's answers are hidden). Each question already ships
            # its own 2 gold + 8 distractor Wikipedia paragraphs; the pooled corpus for local
            # search is built separately by scripts/build_hotpotqa_index.py. `id` is already
            # a stable unique string, reused as example_id. Loaded from HuggingFace; return
            # early to bypass the local-path existence check.
            from datasets import load_dataset
            df = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation").to_pandas()
            df = df.rename(columns={"question": "problem", "answer": "gold answer", "id": "example_id"})
            return df[["example_id", "problem", "gold answer"]].to_dict("records")
        elif dataset_name.lower() == "bioasq":
            # rag-datasets/rag-mini-bioasq, question-answer-passages config, test split
            # (4719 rows) -- freely loadable, unlike the official bigbio/bioasq_task_b which
            # requires BioASQ registration credentials. Free-text answers (not multiple
            # choice), so the standard LLM-judge grader applies directly, no custom grader
            # needed. The matching passage corpus (text-corpus config) is built separately by
            # scripts/build_bioasq_index.py. Loaded from HuggingFace; return early to bypass
            # the local-path existence check.
            from datasets import load_dataset
            df = load_dataset("rag-datasets/rag-mini-bioasq", "question-answer-passages", split="test").to_pandas()
            df = df.rename(columns={"question": "problem", "answer": "gold answer", "id": "example_id"})
            return df[["example_id", "problem", "gold answer"]].to_dict("records")

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
                            if stop_reason == 'max_steps' or (stop_reason is None and search_calls >= 200):
                                is_completed = False
                        
                        if is_completed:
                            self.completed_problems.add(problem)
                            
                    print(f"Loaded {len(self.existing_results)} existing results from {self.output_path}")
                    print(f"Considering {len(self.completed_problems)} problems as completed.")
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load existing results from {self.output_path}: {e}")
                self.existing_results = []
                self.completed_problems = set()

    async def grade_sample(self, question: str, correct_answer: str, response: str,
                           options_block: str | None = None) -> str:
        if options_block is not None:
            grader_prompt = MEDQA_GRADER_TEMPLATE.format(
                question=question,
                options=options_block,
                correct_answer=correct_answer,
                response=response,
            )
        else:
            grader_prompt = self.grader_template.format(
                question=question,
                correct_answer=correct_answer,
                response=response
            )

        prompt_messages = [
            self.grader_model._pack_message(content=grader_prompt, role="user")
        ]
        sampler_response = await self.grader_model.acall(prompt_messages)
        grading_response = sampler_response.response_text

        match = re.search(r"correct:\s*(yes|no)", grading_response.output, re.IGNORECASE)
        return match.group(1).lower() if match else "no"

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        return asyncio.run(self._run_async(sampler))

    async def _run_async(self, sampler: SamplerBase) -> EvalResult:
        results = []
        lock = asyncio.Lock()
        sem = asyncio.Semaphore(self.num_workers)

        final_json_results_map = {}
        for r in self.existing_results:
            if r['problem'] in self.completed_problems:
                final_json_results_map[r['problem']] = r

        async def _process_single(row):
            problem = row.get("problem", "")

            max_retries = 5
            for retry_attempt in range(max_retries):
                try:
                    gold_answer = row.get("gold answer", "")
                    is_entity_q = self.dataset_name.lower() in ENTITY_STYLE_DATASETS

                    # Select prompt template. An explicit override wins over dataset routing.
                    _tmpl_by_name = {
                        "plain": PLAIN_QUERY_TEMPLATE, "natural": NATURAL_QUERY_TEMPLATE,
                        "elaborate": ELABORATE_QUERY_TEMPLATE, "polite": POLITE_QUERY_TEMPLATE,
                        "direct": DIRECT_QUERY_TEMPLATE,
                        "confident_parametric": CONFIDENT_PARAMETRIC_QUERY_TEMPLATE,
                        "query": QUERY_TEMPLATE, "entity": ENTITY_QUESTIONS_QUERY_TEMPLATE,
                    }
                    if self.query_template_override:
                        query = _tmpl_by_name[self.query_template_override].format(Question=problem)
                    elif is_entity_q:
                        query = ENTITY_QUESTIONS_QUERY_TEMPLATE.format(Question=problem)
                    elif self.dataset_name.lower() in PLAIN_QUERY_DATASETS:
                        query = PLAIN_QUERY_TEMPLATE.format(Question=problem)
                    elif self.dataset_name.lower() in NATURAL_QUERY_DATASETS:
                        query = NATURAL_QUERY_TEMPLATE.format(Question=problem)
                    else:
                        query = QUERY_TEMPLATE.format(Question=problem)

                    prompt_messages = []
                    if self.history_pool:
                        rng = random.Random(row.get("example_id") or problem)
                        prompt_messages = list(rng.choice(self.history_pool))
                    prompt_messages.append(sampler._pack_message(content=query, role="user"))

                    # Sampler
                    sampler_response = await sampler.acall(prompt_messages)
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
                        if gold_answer and self.grader_model is not None:
                            options_block = _format_medqa_options(row.get("options"), row.get("answer_idx"))
                            grade_result = await self.grade_sample(problem, str(gold_answer), answer_text,
                                                                   options_block=options_block)
                            is_correct = grade_result == "yes"
                        else:
                            # No grader (e.g. --no_grader): leave correctness to offline regex grading.
                            is_correct = None

                    score = 1.0 if is_correct else (None if is_correct is None else 0.0)
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

                    result_entry = {
                        "problem": problem,
                        "correct_answer": gold_answer,
                        "sampler_response": extract_answer_text(response1_text.output) if is_entity_q else response1_text.output,
                        "sampler_correct": is_correct,
                        "sampler_search_calls": metadata.get("search_calls", 0),
                        "stop_reason": metadata.get("stop_reason"),
                        "metrics": metrics,
                    }
                    # Persist the dataset's stable id (when present) so paired cross-condition
                    # analysis can join on a UNIQUE key instead of the gold answer (which can
                    # repeat across questions and silently collapse rows in a dict-keyed join).
                    if row.get("example_id") is not None:
                        result_entry["example_id"] = row.get("example_id")
                    if is_entity_q:
                        result_entry["sampler_explanation"] = extract_explanation(response1_text.output)
                        source_file = row.get("source_file")
                        if source_file:
                            result_entry["source_file"] = source_file

                    async with lock:
                        final_json_results_map[problem] = result_entry
                        if self.output_path:
                            with open(self.output_path, "w") as f:
                                json.dump(list(final_json_results_map.values()), f, indent=4)

                    return result

                except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.TimeoutException, ConnectionError, ModelHTTPError, ModelAPIError) as e:
                    # ModelAPIError is ModelHTTPError's PARENT class (a bare timeout raises
                    # ModelAPIError directly, not the narrower ModelHTTPError) -- so a raw
                    # ModelAPIError falls through to the generic transient-retry path below.
                    # self.retry_model_api_errors=False reverts to the pre-fix behavior: let it
                    # propagate and crash the batch (see constructor docstring for why).
                    if isinstance(e, ModelAPIError) and not isinstance(e, ModelHTTPError) and not self.retry_model_api_errors:
                        raise
                    # Provider 429/5xx are transient (rate-limit / temporary unavailability) and
                    # should be retried.
                    if isinstance(e, ModelHTTPError) and e.status_code not in (429, 500, 502, 503, 504):
                        # Non-transient model error. 401/403/404 (auth / bad model name) will
                        # never succeed for ANY example, so fail fast. Any other non-transient
                        # 4xx (e.g. a 400 "invalid message content" that Ollama raises on a
                        # single bad trajectory) is example-specific: skip just this example
                        # rather than aborting the whole run over one row.
                        if e.status_code in (401, 403, 404):
                            raise
                        print(f"\nNon-retryable model error ({e.status_code}) on this example; skipping it: {e}")
                        break
                    if retry_attempt < max_retries - 1:
                        wait_time = 2 ** retry_attempt
                        print(f"\nTransient error on attempt {retry_attempt + 1}/{max_retries}: {e}")
                        print(f"Retrying in {wait_time} seconds...")
                        await asyncio.sleep(wait_time)
                    else:
                        print(f"\nFailed after {max_retries} attempts, skipping this example")

                except UnexpectedModelBehavior as e:
                    # The model exhausted pydantic-ai's internal output/tool-validation
                    # retries on THIS example (e.g. kept emitting tool-only turns and never
                    # produced a final answer). This is example-specific, not a run-wide
                    # failure — retry a couple of times (temperature is stochastic, so a
                    # rerun may produce a valid final answer) and then SKIP the example
                    # rather than letting the exception propagate out of asyncio.gather and
                    # abort the entire run (which would also poison --resume: the row never
                    # gets marked complete, so every resumed run re-hits and re-aborts here).
                    if retry_attempt < 2:
                        print(f"\nUnexpectedModelBehavior on attempt {retry_attempt + 1}: {e}. Retrying this example...")
                        continue
                    print(f"\nUnexpectedModelBehavior persisted; skipping this example: {e}")
                    break

            return None

        async def _process_with_semaphore(row, pbar):
            async with sem:
                result = await _process_single(row)
                pbar.update(1)
                return result

        pending = [row for row in self.examples if row.get("problem") not in self.completed_problems]

        with tqdm(total=len(pending), desc="Evaluating") as pbar:
            tasks = [_process_with_semaphore(row, pbar) for row in pending]
            task_results = await asyncio.gather(*tasks)

        async with lock:
            for r in task_results:
                if r is not None:
                    results.append(r)

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