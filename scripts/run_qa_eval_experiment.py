import os
import sys
import argparse
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.services.base_agent import BaseAgent
from src.services.brave_search import BraveSearchService
from src.services.wiki_search import WikipediaSearchService
from src.services.local_index_search import LocalIndexSearchService
from src.services.qa_eval import EvaluationService
from src.services.agent_sampler import AgentAsSampler
from src.services.iterative_search_agent import IterativeSearchAgent
from src.services.generalized_iterative_search_agent import GeneralizedIterativeSearchAgent
from src.services.entity_questions import ConciseAnswer, ENTITY_STYLE_DATASETS
from pydantic_ai import Tool

def setup_args():
    parser = argparse.ArgumentParser(description="Run HLE/FACTS evaluation experiment.")
    parser.add_argument("--test", action='store_true', default=False, help="Run in test mode with fewer examples (10).")
    parser.add_argument("--num_examples", type=int, default=None, help="Specific number of examples to run.")
    parser.add_argument("--resume", action='store_true', default=False, help="Resume incomplete runs.")
    parser.add_argument("--dataset", type=str, default="facts-search", choices=["facts-param", "facts-search", "facts-open", "medqa", "medqa-500", "medqa-terse", "nq", "entity-questions", "popqa", "sharechat", "sharechat-benchmark", "curated-sharechat", "curated-sharechat-benchmark", "musique-natural", "musique-natural2", "frames", "frames-benchmark", "frames-cues"], help="Dataset to evaluate on.")
    parser.add_argument("--dataset_path", type=str, default=None, help="Override path to the dataset file (used by frames-cues to select a condition file, and frames-benchmark/musique-natural/medqa-500/medqa-terse variants).")
    parser.add_argument("--history_path", type=str, default=None, help="Path to a JSON file of {role, content} turns prepended as conversation history before every question (e.g. an unrelated multi-turn chit-chat prefix).")
    parser.add_argument("--search-backend", dest="search_backend", type=str, default="brave", choices=["brave", "wiki", "local"], help="Search tool backend: 'brave' (paid API), 'wiki' (free live MediaWiki), or 'local' (offline index). Default: brave.")
    parser.add_argument("--index-dir", type=str, default="data/frames_index", help="Corpus directory for --search-backend local.")
    parser.add_argument("--local-backend", type=str, default="bm25", choices=["bm25", "dense"], help="Retrieval method for the local index backend.")
    parser.add_argument("--dense-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Query encoder for --local-backend dense (used only if the index manifest doesn't record one).")
    parser.add_argument("--agent_type", type=str, required=True, choices=["baseline", "no_search", "iterative", "generalized"], help="Type of agent to evaluate.")
    parser.add_argument("--model_name", type=str, default="gemini-3-pro-preview", help="Name of the model to use.")
    parser.add_argument("--provider_name", type=str, default="Google", help="Provider name for the model.")
    parser.add_argument("--run_name", type=str, default="run_1", help="Identifier for this run.")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results.")
    parser.add_argument("--baseline_sys_prompt_path", type=str, default=None, help="Path to read baseline system prompt.")
    parser.add_argument("--no_structured_output", action='store_true', default=False, help="Disable structured output (ConciseAnswer) for entity-style datasets.")
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "test", "train"], help="EntityQuestions split to use (default: dev).")
    parser.add_argument("--relations", type=str, nargs='+', default=None, help="Restrict EntityQuestions to these property IDs (e.g. P26 P264).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling (default: 0).")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of parallel threads for evaluation (default: 1).")
    parser.add_argument("--grader_model", type=str, default="gpt-oss:20b", help="Grader LLM model name (default: gpt-oss:20b).")
    parser.add_argument("--grader_provider", type=str, default="ollama", help="Grader LLM provider (default: ollama). Use e.g. 'Google' with a remote grader_model when no local ollama is available.")
    parser.add_argument("--query_template", type=str, default=None, choices=["plain", "natural", "elaborate", "polite", "direct", "confident_parametric", "query", "entity"], help="Override the dataset-based query-template routing (plain=passthrough, natural='answer in 2-4 sentences', elaborate='detailed 8-10 sentence explanation', polite=extreme-politeness wrapper with no length/format directive, direct=maximal answer-directive with no length/politeness/format, confident_parametric=explicit 'you already know this, no need to search' capability-framing instruction, query=structured Exact-Answer). For controlled template experiments.")
    parser.add_argument("--no_grader", action='store_true', default=False, help="Disable the LLM judge entirely (grader=None -> exact-match path, no grader API calls). Use with offline regex grading.")
    return parser.parse_args()

def get_agent(agent_type, model_name, provider_name, search_service, resume=False, baseline_sys_prompt_path=None, run_name="run_1", dataset_name="facts-search", no_structured_output=False):
    # Use structured output for entity-questions with baseline/no_search agents
    if no_structured_output:
        output_type = str
    else:
        output_type = ConciseAnswer if dataset_name.lower() in ENTITY_STYLE_DATASETS else str

    if agent_type == "baseline":
        if baseline_sys_prompt_path:
            with open(baseline_sys_prompt_path, 'r') as f:
                system_prompt = f.read()
        else:
            system_prompt = None
        raw_agent = BaseAgent(
            provider_name=provider_name,
            model_name=model_name,
            tools=[Tool(search_service.search)],
            agent_name=f"baseline_agent_{run_name}",
            system_prompt=system_prompt,
            output_type=output_type
        )
        return AgentAsSampler(raw_agent)

    elif agent_type == "no_search":
        raw_agent = BaseAgent(
            provider_name=provider_name,
            model_name=model_name,
            agent_name="no_search_agent",
            output_type=output_type
        )
        return AgentAsSampler(raw_agent)
    
    elif agent_type == "iterative":
        if dataset_name.lower() in ENTITY_STYLE_DATASETS:
            print(f"Warning: iterative agent does not use ConciseAnswer structured output for {dataset_name}.")
        raw_model = BaseAgent(
            provider_name=provider_name,
            model_name=model_name,
            agent_name="iterative_search_model"
        )
        # IterativeSearchAgent is already a SamplerBase
        return IterativeSearchAgent(
            model=raw_model,
            search_tool=search_service,
            max_steps=8 if resume else 4,
            load_history=resume
        )

    elif agent_type == "generalized":
        if dataset_name.lower() in ENTITY_STYLE_DATASETS:
            print(f"Warning: generalized agent does not use ConciseAnswer structured output for {dataset_name}.")
        # GeneralizedIterativeSearchAgent is already a SamplerBase
        return GeneralizedIterativeSearchAgent(
            search_tool=search_service,
            max_steps=4,
            load_history=resume,
            main_model_name=model_name
        )
    
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

def main():
    args = setup_args()
    
    print(f"--- Starting Evaluation: {args.agent_type.upper()} on {args.dataset.upper()} ---")
    
    # Setup configuration
    num_examples = 10 if args.test and args.num_examples is None else args.num_examples
    
    # Initialize Core Services
    if args.search_backend == "wiki":
        print("Using free live MediaWiki search backend.")
        search_service = WikipediaSearchService()
    elif args.search_backend == "local":
        print(f"Using local index search backend ({args.local_backend}) from {args.index_dir}.")
        search_service = LocalIndexSearchService(args.index_dir, backend=args.local_backend, dense_model=args.dense_model)
    else:
        search_service = BraveSearchService()

    # Initialize Grader (not needed for entity-questions which uses exact match)
    if args.no_grader:
        print("Grader disabled (--no_grader): no judge API calls; grade offline (e.g. regex).")
        grader_agent = None
    elif args.dataset.lower() in ENTITY_STYLE_DATASETS or args.dataset.lower() == "sharechat" or args.dataset.lower() == "sharechat-benchmark":
        print(f"Using exact-match grading for {args.dataset} (no grader LLM needed).")
        grader_agent = None
    else:
        print(f"Initializing Grader Agent ({args.grader_provider}/{args.grader_model})...")
        grader_agent_raw = BaseAgent(provider_name=args.grader_provider, model_name=args.grader_model, agent_name="grader_agent")
        grader_agent = AgentAsSampler(grader_agent_raw)

    # Initialize Evaluated Agent
    print(f"Initializing {args.agent_type} agent with model {args.model_name}...")
    agent = get_agent(args.agent_type, args.model_name, args.provider_name, search_service, args.resume, args.baseline_sys_prompt_path, args.run_name, dataset_name=args.dataset, no_structured_output=args.no_structured_output)
    
    # Setup Output Path
    os.makedirs(args.output_dir, exist_ok=True)
    output_filename = f"{args.dataset}_{args.agent_type}_{args.model_name}_{args.run_name}.json"
    output_path = os.path.join(args.output_dir, output_filename)
    
    # Run Evaluation
    eval_service = EvaluationService(
        grader_model=grader_agent,
        dataset_name=args.dataset,
        dataset_path=args.dataset_path,
        output_path=output_path,
        num_examples=num_examples,
        resume_incomplete=args.resume,
        split=args.split,
        relations=args.relations,
        seed=args.seed,
        num_workers=args.num_workers,
        query_template_override=args.query_template,
        history_path=args.history_path,
    )
    
    print(f"Running evaluation... Results will be saved to {output_path}")
    results = eval_service(sampler=agent)
    
    print(f"\n--- Evaluation Complete ---")
    print(f"Aggregated Metrics: {results.metrics}")

if __name__ == "__main__":
    main()