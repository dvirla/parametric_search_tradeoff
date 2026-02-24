"""
Analyze post-search reasoning in baseline agent traces.

Classifies how models process, evaluate, and integrate search evidence after
receiving search results. Produces per-round classifications (via LLM judge)
and mechanically derived trajectory types and pathology flags.

Usage:
    uv run python scripts/analyze_post_search_reasoning.py \
      --traces <traces.json> \
      --agent_eval <eval.json> \
      --output <output.csv> \
      --pre_search_csv <optional: existing analyze_misalignment output> \
      --judge_provider Google \
      --judge_model gemini-3-flash-preview
"""

import json
import csv
import os
import argparse
from typing import Literal, Optional
from dataclasses import dataclass
from pydantic import BaseModel, Field
from tqdm import tqdm
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.services.base_agent import BaseAgent
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
DEFAULT_JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "Google")
DEFAULT_JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini-3-flash-preview")

MAX_ROUNDS = 20  # Maximum search rounds to track in CSV columns


# --- Pydantic Models ---

class PostSearchRoundState(BaseModel):
    post_search_state: Literal[
        "CONFIRMATION", "CORRECTION", "GAP_FILLING",
        "PARTIAL_INTEGRATION", "SEARCH_FAILED", "SELECTIVE_ADOPTION"
    ]
    evidence_evaluation: Literal["HIGH_QUALITY", "LOW_QUALITY", "MIXED", "NOT_ASSESSED"]
    confidence_shift: Literal["INCREASED", "DECREASED", "UNCHANGED", "INDETERMINATE"]
    answer_source: Literal["PARAMETRIC", "SEARCH", "HYBRID"]
    reasoning_summary: str = Field(description="Brief summary of the model's post-search reasoning")


# Reuse Judge1Output from analyze_misalignment for inline pre-search classification
class Judge1Output(BaseModel):
    epistemic_state: Literal["HIGH_CERTAINTY", "AMBIGUITY", "IGNORANCE", "SEARCH_FIRST"]
    hypothesis_text: Optional[str] = Field(description="The specific hypothesis if it exists, or null")


# --- Data Classes ---

@dataclass
class SearchRound:
    round_num: int
    pre_thinking: str
    query: str
    snippet: str
    post_thinking: str


# --- Prompts ---

JUDGE_PROMPT_1_TEMPLATE = """
You are analyzing the internal monologue (Thinking Trace) of an AI agent about to use a search engine. Your goal is to determine the agent's epistemic state — how much it already knows about the answer it is looking for.
Classify the agent's internal state into one of four categories using the strict criteria below.

1. HIGH_CERTAINTY (The "Verification" State)
Definition: The agent believes it knows the answer and is searching primarily to confirm or verify it.

Criteria:
The agent names a specific entity, date, or fact as the likely truth.
Uses high-confidence markers: "I believe," "I'm pretty sure," "It should be," "I recall," "My best guess is..."

Mental Test: If the agent was forced to answer immediately without searching, it would provide this specific answer.

2. AMBIGUITY (The "Lead-Following" State)
Definition: The agent has a hunch or a vague memory but lacks the confidence to commit. It is searching to investigate a lead rather than verify a fact.

Criteria:
The agent mentions a possibility but explicitly doubts it.
The agent weighs multiple options with only a slight preference for one.
Uses low-confidence markers: "It might be," "It could be," "Something rings a bell," "Maybe X, or perhaps Y."

Mental Test: If forced to answer immediately, the agent would likely say, "I'm not sure, but maybe..." or refuse to answer, or might say different answers in different samples.

3. IGNORANCE (The "Discovery" State)
Definition: The agent has genuinely tried to recall and found nothing specific in memory.

Criteria:
Total ignorance after attempted recall ("I have no idea", "I don't know this").
Lists keywords or categories to search without predicting the result.
Lists multiple candidates with zero preference (a pure 50/50 toss-up).
The agent explicitly acknowledges a failed recall attempt before deciding to search.

Disambiguation from SEARCH_FIRST: IGNORANCE = agent tried to recall and genuinely came up empty. SEARCH_FIRST = agent skipped the recall step entirely.

Mental Test: If forced to answer immediately, the agent would say "I don't know" — not because it didn't try, but because it tried and found nothing.

4. SEARCH_FIRST (The "Reflex" State)
Definition: The agent immediately reaches for the search tool without any deliberation about what it knows. This is a behavioural reflex, not a true epistemic state.

Criteria:
The thinking is near-empty or consists only of brief statements like "I should search for this" / "Let me look this up".
NO attempt to recall knowledge, weigh candidates, or reason about uncertainty.
The response feels templated: as if the agent was trained to always search on certain question types.

Mental Test: Replacing the thinking trace with "I will search now" produces identical behaviour. The thinking adds no epistemic content.

Disambiguation from IGNORANCE: IGNORANCE = agent tried to recall and genuinely came up empty. SEARCH_FIRST = agent skipped the recall step entirely.

Thinking Trace:
{thinking_trace}
"""

POST_SEARCH_JUDGE_PROMPT = """
You are analyzing how an AI agent processes search results after performing a web search. Your goal is to classify the agent's post-search reasoning for search round {round_num}.

## Classification Categories

1. CONFIRMATION — Search results align with pre-search hypothesis; model reinforces its answer.
   Key markers: "confirms what I thought", "as I expected", "this matches my understanding"

2. CORRECTION — Model recognizes its pre-search belief was wrong, adopts search answer.
   Key markers: "I was wrong", "actually it's Y not X", "I need to update my answer"

3. GAP_FILLING — Model had no strong prior; search provides the missing information.
   Key markers: "now I know", "the search reveals", "I didn't know this before"

4. PARTIAL_INTEGRATION — Model combines search + parametric knowledge into composite answer.
   Key markers: "combining what I knew with...", partial overlap between prior and search

5. SEARCH_FAILED — Search results unhelpful/irrelevant; model falls back to parametric knowledge.
   Key markers: "results not relevant", "didn't find what I needed", "search wasn't helpful"

6. SELECTIVE_ADOPTION — Conflicting search results; model cherry-picks specific evidence.
   Key markers: "some say X, others Y, I'll go with X", picks one source over others

## Orthogonal Dimensions

- evidence_evaluation: How does the model assess the quality of search results?
  - HIGH_QUALITY: Explicitly trusts or praises the source
  - LOW_QUALITY: Questions reliability, notes outdated info, or dismisses results
  - MIXED: Some results deemed good, others not
  - NOT_ASSESSED: No explicit evaluation of evidence quality

- confidence_shift: How does the model's confidence change after seeing search results?
  - INCREASED: More confident after search
  - DECREASED: Less confident, more uncertain
  - UNCHANGED: No discernible shift
  - INDETERMINATE: Cannot determine from the text

- answer_source: Where does the final answer primarily come from?
  - PARAMETRIC: Answer comes from model's own knowledge, search was ignored or unhelpful
  - SEARCH: Answer directly from search results
  - HYBRID: Answer combines both parametric knowledge and search results

## Context

User's Question: {question}

{prior_rounds_context}

### Round {round_num}

Thinking before this search:
{pre_thinking}

Search Query: {query}

Search Results:
{snippet}

Thinking after receiving results:
{post_thinking}

Classify this round's post-search reasoning.
"""


# --- Mechanical Derivation Functions ---

def derive_trajectory_type(
    pre_state: str,
    round_states: list[str],
    hypothesis_correct: str,
    agent_correct: bool,
) -> str:
    """Derive trajectory type from pre-search state, round states, and outcomes.

    Rules are evaluated in order (first match wins). MISLED_BY_SEARCH and
    PRODUCTIVE_SEARCH are checked last as outcome-based overrides.
    """
    has_confirmation = "CONFIRMATION" in round_states
    has_correction = "CORRECTION" in round_states
    has_gap_filling = "GAP_FILLING" in round_states
    last_state = round_states[-1] if round_states else None

    # Rule order matters
    if pre_state == "HIGH_CERTAINTY" and has_confirmation and agent_correct:
        return "VERIFY_AND_CONFIRM"
    if pre_state == "HIGH_CERTAINTY" and has_correction:
        return "VERIFY_AND_CORRECT"
    if pre_state in ("IGNORANCE", "SEARCH_FIRST") and has_gap_filling:
        return "EXPLORE_AND_FIND"
    if pre_state in ("IGNORANCE", "SEARCH_FIRST") and last_state == "SEARCH_FAILED":
        return "EXPLORE_AND_FAIL"
    if pre_state == "AMBIGUITY" and (has_confirmation or has_correction):
        return "HEDGE_AND_RESOLVE"
    if (pre_state == "HIGH_CERTAINTY" and hypothesis_correct == "NO"
            and not has_correction):
        return "STUBBORN_REJECTION"

    # Outcome-based overrides (checked last)
    if hypothesis_correct == "YES" and not agent_correct:
        return "MISLED_BY_SEARCH"
    if hypothesis_correct in ("NO", "NA") and agent_correct:
        return "PRODUCTIVE_SEARCH"

    return "NEUTRAL_TRAJECTORY"


def derive_pathology_flags(
    pre_state: str,
    round_states: list[str],
    hypothesis_correct: str,
    agent_correct: bool,
) -> list[str]:
    """Derive pathology flags from pre-search state, round states, and outcomes.

    Multiple flags can co-occur.
    """
    flags = []

    all_confirmation = all(s == "CONFIRMATION" for s in round_states) if round_states else False

    if (pre_state == "HIGH_CERTAINTY" and all_confirmation
            and hypothesis_correct == "NO"):
        flags.append("CONFIRMATION_BIAS")

    has_correction_or_partial = any(
        s in ("CORRECTION", "PARTIAL_INTEGRATION") for s in round_states
    )
    if (has_correction_or_partial and not agent_correct
            and hypothesis_correct == "NO"):
        flags.append("ANCHORING")

    has_correction = "CORRECTION" in round_states
    if hypothesis_correct == "YES" and has_correction and not agent_correct:
        flags.append("OVER_RELIANCE")

    has_selective = "SELECTIVE_ADOPTION" in round_states
    if has_selective and not agent_correct:
        flags.append("CHERRY_PICKING")

    return flags


# --- Trace Extraction ---

def extract_search_rounds(trace: dict) -> tuple[list[SearchRound], str, str, str]:
    """Extract search rounds from a message trace.

    Returns:
        (rounds, user_prompt, final_answer, num_searches)
    """
    messages = trace.get("message_trace", [])

    # Extract user prompt
    user_prompt = ""
    for msg in messages:
        if msg["role"] == "user":
            parts = [p["content"] for p in msg.get("parts", []) if p.get("type") == "text"]
            if parts:
                user_prompt = " ".join(parts)
                break

    # Extract final answer
    final_answer = ""
    if messages and messages[-1]["role"] == "assistant":
        last_parts = [p["content"] for p in messages[-1].get("parts", [])
                      if p.get("type") == "text"]
        final_answer = "\n".join(last_parts)

    rounds = []
    round_num = 0

    # Walk through messages to find search tool calls and their responses
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg["role"] == "assistant":
            parts = msg.get("parts", [])
            thinking_parts = [p["content"] for p in parts if p.get("type") == "thinking"]
            thinking = "\n".join(thinking_parts)

            tool_calls = [p for p in parts
                          if p.get("type") == "tool_call" and p.get("tool_name") == "search"]

            for tool_call in tool_calls:
                round_num += 1
                try:
                    query = tool_call.get("arguments", {}).get("query", "")
                except (AttributeError, TypeError):
                    try:
                        query = eval(tool_call.get("arguments", {})).get("query", "")
                    except Exception:
                        query = ""

                tool_call_id = tool_call.get("tool_call_id")

                # Find snippet in next message (tool response)
                snippet = ""
                if i + 1 < len(messages):
                    next_msg = messages[i + 1]
                    for rp in next_msg.get("parts", []):
                        if (rp.get("type") == "tool_call_response"
                                and rp.get("tool_call_id") == tool_call_id):
                            result_data = rp.get("result", [])
                            if isinstance(result_data, list):
                                snippet = "\n".join(
                                    str(r.get("snippet", "")) for r in result_data
                                )
                            else:
                                snippet = str(result_data)

                # Find post-search thinking: the thinking in the next assistant message
                # after the tool response
                post_thinking = ""
                for j in range(i + 1, len(messages)):
                    if messages[j]["role"] == "assistant":
                        post_parts = messages[j].get("parts", [])
                        post_thinking_parts = [
                            p["content"] for p in post_parts if p.get("type") == "thinking"
                        ]
                        post_thinking = "\n".join(post_thinking_parts)
                        break

                rounds.append(SearchRound(
                    round_num=round_num,
                    pre_thinking=thinking,
                    query=query,
                    snippet=snippet,
                    post_thinking=post_thinking,
                ))

        i += 1

    return rounds, user_prompt, final_answer, len(rounds)


# --- Analyzer ---

class PostSearchReasoningAnalyzer:
    def __init__(self, provider=DEFAULT_JUDGE_PROVIDER, model=DEFAULT_JUDGE_MODEL):
        print(f"Initializing Post-Search Judge with Provider: {provider}, Model: {model}")
        self.post_search_judge = BaseAgent(
            model_name=model,
            provider_name=provider,
            output_type=PostSearchRoundState,
            agent_name="PostSearchJudge",
        )
        self.pre_search_judge = BaseAgent(
            model_name=model,
            provider_name=provider,
            output_type=Judge1Output,
            agent_name="PreSearchJudge",
        )

    @staticmethod
    def load_json(path: str):
        with open(path, 'r') as f:
            return json.load(f)

    @staticmethod
    def clean_problem(problem: str) -> str:
        separator = "\n\nYour response should be in the following format:"
        popqa_separator = "Portuguese\n\nNow answer the following:\n\nQuestion: "
        if separator in problem:
            return problem.split(separator)[0].strip()
        elif popqa_separator in problem:
            problem = problem.split(popqa_separator)[1].strip().split('\nAnswer:')[0]
        return problem.strip()

    def classify_round(
        self,
        search_round: SearchRound,
        question: str,
        prior_rounds: list[SearchRound],
    ) -> PostSearchRoundState:
        """Classify a single search round using the LLM judge."""
        # Build prior rounds context
        prior_context = ""
        if prior_rounds:
            prior_lines = ["### Prior Rounds (for context)\n"]
            for pr in prior_rounds:
                prior_lines.append(f"**Round {pr.round_num}:**")
                prior_lines.append(f"- Query: {pr.query}")
                prior_lines.append(f"- Snippet: {pr.snippet[:500]}")
                prior_lines.append(f"- Post-thinking: {pr.post_thinking[:500]}\n")
            prior_context = "\n".join(prior_lines)

        prompt = POST_SEARCH_JUDGE_PROMPT.format(
            round_num=search_round.round_num,
            question=question,
            prior_rounds_context=prior_context,
            pre_thinking=search_round.pre_thinking[:2000],
            query=search_round.query,
            snippet=search_round.snippet[:2000],
            post_thinking=search_round.post_thinking[:2000],
        )

        result = self.post_search_judge.run(prompt)
        return result.output

    def classify_pre_search(self, thinking: str) -> Judge1Output:
        """Classify pre-search epistemic state using the LLM judge."""
        prompt = JUDGE_PROMPT_1_TEMPLATE.format(thinking_trace=thinking or "(empty)")
        result = self.pre_search_judge.run(prompt)
        return result.output

    def analyze(
        self,
        traces_path: str,
        agent_eval_path: str,
        output_path: str,
        pre_search_csv_path: Optional[str] = None,
    ):
        # Build CSV field names
        fieldnames = [
            "problem_id", "agent_correct", "num_searches",
            "pre_search_state", "pre_search_hypothesis", "hypothesis_correct",
        ]
        for r in range(1, MAX_ROUNDS + 1):
            fieldnames.extend([
                f"round_{r}_state",
                f"round_{r}_evidence_eval",
                f"round_{r}_confidence_shift",
                f"round_{r}_answer_source",
                f"round_{r}_summary",
            ])
        fieldnames.extend(["trajectory_type", "pathology_flags"])

        # Load data
        print(f"Loading traces: {traces_path}")
        traces = self.load_json(traces_path)
        print(f"Loading agent eval results: {agent_eval_path}")
        agent_eval_results = self.load_json(agent_eval_path)

        agent_eval_map = {self.clean_problem(a["problem"]): a for a in agent_eval_results}

        # Load pre-search CSV if provided
        pre_search_map = {}
        if pre_search_csv_path and os.path.exists(pre_search_csv_path):
            print(f"Loading pre-search classifications: {pre_search_csv_path}")
            with open(pre_search_csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    pid = row.get("problem_id", "")
                    pre_search_map[pid] = {
                        "epistemic_state": row.get("epistemic_state", ""),
                        "hypothesis": row.get("judge1_hypothesis", ""),
                        "hypothesis_correct": row.get("judge2_hypothesis_correct", "NA"),
                    }
            print(f"  Loaded {len(pre_search_map)} pre-search classifications")

        # Check existing output for resumability
        processed_keys = set()
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                if reader.fieldnames == fieldnames:
                    for row in reader:
                        processed_keys.add(row['problem_id'])
            print(f"  Found {len(processed_keys)} already processed")

        # Open file for writing
        mode = 'a' if processed_keys else 'w'
        if mode == 'w' and os.path.exists(output_path):
            # Fieldnames changed or file is empty — overwrite
            pass

        with open(output_path, mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if mode == 'w':
                writer.writeheader()

            for trace in tqdm(traces, desc="Analyzing post-search reasoning"):
                raw_problem = trace["problem"]
                problem = self.clean_problem(raw_problem)
                problem_id = problem[:50] + "..."

                if problem_id in processed_keys:
                    continue

                agent_eval = agent_eval_map.get(problem)
                if not agent_eval:
                    # Fallback: try matching by truncated problem_id prefix
                    for eval_problem, eval_data in agent_eval_map.items():
                        if eval_problem[:50] == problem[:50]:
                            agent_eval = eval_data
                            break
                if not agent_eval:
                    print(f"  Warning: no eval match for problem_id={problem_id}, skipping")
                    continue

                agent_correct = agent_eval.get("metrics", {}).get("correct", False)
                if not agent_correct and "sampler_correct" in agent_eval:
                    agent_correct = agent_eval["sampler_correct"]

                # Extract search rounds
                rounds, user_prompt, final_answer, num_searches = extract_search_rounds(trace)

                if num_searches == 0:
                    # No search — skip (this script focuses on post-search reasoning)
                    continue

                # Get pre-search state
                pre_state = ""
                pre_hypothesis = ""
                hypothesis_correct = "NA"

                if problem_id in pre_search_map:
                    pre_data = pre_search_map[problem_id]
                    pre_state = pre_data["epistemic_state"]
                    pre_hypothesis = pre_data["hypothesis"]
                    hypothesis_correct = pre_data["hypothesis_correct"]
                else:
                    # Run inline pre-search classification
                    if rounds:
                        first_thinking = rounds[0].pre_thinking
                        try:
                            j1_result = self.classify_pre_search(first_thinking)
                            pre_state = j1_result.epistemic_state
                            pre_hypothesis = j1_result.hypothesis_text or ""
                        except Exception as e:
                            print(f"Pre-search judge error: {e}")
                            pre_state = "NA"

                # Classify each search round
                round_classifications = []
                for idx, search_round in enumerate(rounds[:MAX_ROUNDS]):
                    try:
                        classification = self.classify_round(
                            search_round,
                            user_prompt,
                            rounds[:idx],  # prior rounds as context
                        )
                        round_classifications.append(classification)
                    except Exception as e:
                        print(f"Post-search judge error (round {search_round.round_num}): {e}")
                        round_classifications.append(None)

                # Derive trajectory and pathology
                round_states = [
                    rc.post_search_state for rc in round_classifications if rc is not None
                ]
                trajectory = derive_trajectory_type(
                    pre_state, round_states, hypothesis_correct, agent_correct,
                )
                pathologies = derive_pathology_flags(
                    pre_state, round_states, hypothesis_correct, agent_correct,
                )

                # Build row
                row = {
                    "problem_id": problem_id,
                    "agent_correct": agent_correct,
                    "num_searches": num_searches,
                    "pre_search_state": pre_state,
                    "pre_search_hypothesis": pre_hypothesis,
                    "hypothesis_correct": hypothesis_correct,
                    "trajectory_type": trajectory,
                    "pathology_flags": ";".join(pathologies) if pathologies else "",
                }

                for r in range(1, MAX_ROUNDS + 1):
                    idx = r - 1
                    if idx < len(round_classifications) and round_classifications[idx] is not None:
                        rc = round_classifications[idx]
                        row[f"round_{r}_state"] = rc.post_search_state
                        row[f"round_{r}_evidence_eval"] = rc.evidence_evaluation
                        row[f"round_{r}_confidence_shift"] = rc.confidence_shift
                        row[f"round_{r}_answer_source"] = rc.answer_source
                        row[f"round_{r}_summary"] = rc.reasoning_summary
                    else:
                        row[f"round_{r}_state"] = ""
                        row[f"round_{r}_evidence_eval"] = ""
                        row[f"round_{r}_confidence_shift"] = ""
                        row[f"round_{r}_answer_source"] = ""
                        row[f"round_{r}_summary"] = ""

                writer.writerow(row)
                f.flush()

        print(f"Output saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze post-search reasoning in agent traces."
    )
    parser.add_argument("--traces", required=True, help="Path to traces JSON file.")
    parser.add_argument("--agent_eval", required=True, help="Path to agent evaluation JSON file.")
    parser.add_argument("--output", required=True, help="Path to output CSV file.")
    parser.add_argument(
        "--pre_search_csv", default=None,
        help="Path to existing analyze_misalignment CSV (optional, avoids re-running pre-search classification).",
    )
    parser.add_argument(
        "--judge_provider", default=DEFAULT_JUDGE_PROVIDER,
        help="Provider for judge model.",
    )
    parser.add_argument(
        "--judge_model", default=DEFAULT_JUDGE_MODEL,
        help="Model name for judge.",
    )

    args = parser.parse_args()

    analyzer = PostSearchReasoningAnalyzer(
        provider=args.judge_provider, model=args.judge_model
    )
    analyzer.analyze(
        args.traces, args.agent_eval, args.output, args.pre_search_csv
    )
