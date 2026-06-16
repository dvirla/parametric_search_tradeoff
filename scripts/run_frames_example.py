"""
Quick single-example FRAMES smoke test.

Runs ONE FRAMES question through the baseline agent using the free live-Wikipedia
search backend, and prints the question, gold answer, the search queries the agent
issued, and the model's final answer.

No grader is used (so no Ollama / heavy local model needed) — this only verifies the
agent + free search plumbing end-to-end.

Usage:
    uv run python scripts/run_frames_example.py
    uv run python scripts/run_frames_example.py --index 5 --model gemini-flash-latest
"""

import os
import sys
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from datasets import load_dataset
from pydantic_ai import Tool
from pydantic_ai.messages import ModelResponse, ToolCallPart

from src.services.base_agent import BaseAgent
from src.services.agent_sampler import AgentAsSampler
from src.services.wiki_search import WikipediaSearchService
from src.services.local_index_search import LocalIndexSearchService


def setup_args():
    parser = argparse.ArgumentParser(description="Run a single FRAMES example (no grader).")
    parser.add_argument("--index", type=int, default=0, help="Row index in the FRAMES test split.")
    parser.add_argument("--model", type=str, default="gemini-flash-latest", help="Model name (cheap flash by default).")
    parser.add_argument("--provider", type=str, default="Google", help="Provider name.")
    parser.add_argument("--search-backend", dest="search_backend", default="wiki", choices=["wiki", "local"], help="Search backend: 'wiki' (live MediaWiki) or 'local' (offline index).")
    parser.add_argument("--index-dir", default="data/frames_index", help="Corpus dir for --search-backend local.")
    parser.add_argument("--local-backend", default="bm25", choices=["bm25", "dense"], help="Retrieval method for the local index.")
    return parser.parse_args()


def main():
    args = setup_args()

    # 1. Load one FRAMES example via the same mapping used in qa_eval._load_dataset.
    df = load_dataset("google/frames-benchmark", split="test").to_pandas()
    df = df.rename(columns={"Prompt": "problem", "Answer": "gold answer"})
    row = df.to_dict("records")[args.index]
    question = row["problem"]
    gold = row["gold answer"]

    # 2. Build the baseline agent with the chosen search backend.
    if args.search_backend == "local":
        search_service = LocalIndexSearchService(args.index_dir, backend=args.local_backend)
    else:
        search_service = WikipediaSearchService()
    raw_agent = BaseAgent(
        provider_name=args.provider,
        model_name=args.model,
        tools=[Tool(search_service.search)],
        agent_name="frames_smoke_baseline",
        output_type=str,
    )
    sampler = AgentAsSampler(raw_agent)

    print("=" * 80)
    backend_label = f"local:{args.local_backend}" if args.search_backend == "local" else "wiki"
    print(f"FRAMES example #{args.index}  (model={args.model}, search={backend_label})")
    print("=" * 80)
    print(f"\nQUESTION:\n{question}\n")
    print(f"GOLD ANSWER:\n{gold}\n")
    print("Running agent...\n")

    # 3. Run it (single user turn).
    result = sampler([{"role": "user", "content": question}])
    run = result.response_text  # raw pydantic-ai result

    # 4. Surface the search queries the agent issued.
    queries = []
    for msg in run.all_messages():
        if isinstance(msg, ModelResponse):
            for p in msg.parts:
                if isinstance(p, ToolCallPart) and p.tool_name == "search":
                    queries.append(p.args)

    print(f"SEARCH CALLS: {result.response_metadata.get('search_calls')}")
    for i, q in enumerate(queries, 1):
        print(f"  {i}. {q}")

    print(f"\nMODEL ANSWER:\n{run.output}\n")
    print("=" * 80)
    print("Smoke test complete (answer not auto-graded — eyeball vs. gold above).")


if __name__ == "__main__":
    main()
