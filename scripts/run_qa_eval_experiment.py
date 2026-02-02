import os
import sys
import argparse
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.services.base_agent import BaseAgent
from src.services.brave_search import BraveSearchService
from src.services.qa_eval import EvaluationService
from src.services.agent_sampler import AgentAsSampler
from src.services.iterative_search_agent import IterativeSearchAgent
from src.services.generalized_iterative_search_agent import GeneralizedIterativeSearchAgent
from pydantic_ai import Tool

def setup_args():
    parser = argparse.ArgumentParser(description="Run HLE/FACTS evaluation experiment.")
    parser.add_argument("--test", action='store_true', default=False, help="Run in test mode with fewer examples (10).")
    parser.add_argument("--num_examples", type=int, default=None, help="Specific number of examples to run.")
    parser.add_argument("--resume", action='store_true', default=False, help="Resume incomplete runs.")
    parser.add_argument("--dataset", type=str, default="facts-search", choices=["facts-param", "facts-search", "nq"], help="Dataset to evaluate on.")
    parser.add_argument("--agent_type", type=str, required=True, choices=["baseline", "no_search", "iterative", "generalized"], help="Type of agent to evaluate.")
    parser.add_argument("--model_name", type=str, default="gemini-3-pro-preview", help="Name of the model to use.")
    parser.add_argument("--provider_name", type=str, default="Google", help="Provider name for the model.")
    parser.add_argument("--run_name", type=str, default="run_1", help="Identifier for this run.")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results.")
    return parser.parse_args()

def get_agent(agent_type, model_name, provider_name, search_service, resume=False):
    if agent_type == "baseline":
        raw_agent = BaseAgent(
            provider_name=provider_name, 
            model_name=model_name, 
            tools=[Tool(search_service.search)], 
            agent_name="baseline_agent"
        )
        return AgentAsSampler(raw_agent)
    
    elif agent_type == "no_search":
        raw_agent = BaseAgent(
            provider_name=provider_name, 
            model_name=model_name, 
            agent_name="no_search_agent"
        )
        return AgentAsSampler(raw_agent)
    
    elif agent_type == "iterative":
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
    search_service = BraveSearchService()
    
    # Initialize Grader
    print("Initializing Grader Agent...")
    grader_agent_raw = BaseAgent(provider_name="ollama", model_name="gpt-oss:20b", agent_name="grader_agent")
    grader_agent = AgentAsSampler(grader_agent_raw)

    # Initialize Evaluated Agent
    print(f"Initializing {args.agent_type} agent with model {args.model_name}...")
    agent = get_agent(args.agent_type, args.model_name, args.provider_name, search_service, args.resume)
    
    # Setup Output Path
    os.makedirs(args.output_dir, exist_ok=True)
    output_filename = f"{args.dataset}_{args.agent_type}_{args.model_name}_{args.run_name}.json"
    output_path = os.path.join(args.output_dir, output_filename)
    
    # Run Evaluation
    eval_service = EvaluationService(
        grader_model=grader_agent,
        dataset_name=args.dataset,
        output_path=output_path,
        num_examples=num_examples,
        resume_incomplete=args.resume
    )
    
    print(f"Running evaluation... Results will be saved to {output_path}")
    results = eval_service(sampler=agent)
    
    print(f"\n--- Evaluation Complete ---")
    print(f"Aggregated Metrics: {results.metrics}")

if __name__ == "__main__":
    main()