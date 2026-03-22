import os
import json
import requests
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional
from datetime import datetime
import argparse
import tqdm

load_dotenv()

LOGFIRE_API_KEY = os.getenv("LOGFIRE_API_KEY")
LOGFIRE_API_URL = "https://logfire-us.pydantic.dev"


def extract_message_trace(all_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extracts a structured trace from the agent's message chain.
    """
    trace = []
    
    for msg in all_messages:
        message_obj = {
            "role": msg.get("role", "unknown"),
            "timestamp": msg.get("timestamp"),
            "finish_reason": msg.get("finish_reason"),
            "parts": []
        }
        
        for part in msg.get("parts", []):
            part_type = part.get("type")
            
            if part_type == "text":
                message_obj["parts"].append({
                    "type": "text",
                    "content": part.get("content", "")
                })
            
            elif part_type == "thinking":
                message_obj["parts"].append({
                    "type": "thinking",
                    "content": part.get("content", "")
                })
            
            elif part_type == "tool_call":
                tool_call = {
                    "type": "tool_call",
                    "tool_call_id": part.get("id"),
                    "tool_name": part.get("name"),
                    "arguments": part.get("arguments", {})
                }
                message_obj["parts"].append(tool_call)
            
            elif part_type == "tool_call_response":
                tool_return = {
                    "type": "tool_call_response",
                    "tool_call_id": part.get("id"),
                    "tool_name": part.get("name"),
                    "result": part.get("result")
                }
                message_obj["parts"].append(tool_return)
            
            else:
                message_obj["parts"].append(part)
        
        trace.append(message_obj)
    
    return trace


def clean_problem(problem: str) -> str:
    """Removes the instruction suffix from the problem string to get the core question."""
    separator = "\n\nYour response should be in the following format:"
    popqa_separator = "Portuguese\n\nNow answer the following:\n\nQuestion: "
    if separator in problem:
        return problem.split(separator)[0].strip()
    elif popqa_separator in problem:
        problem = problem.split(popqa_separator)[1].strip().split('\nAnswer:')[0]
    return problem.strip()


def load_eval_problems(eval_json_path: str) -> set:
    """
    Loads the set of problem strings from an evaluation JSON file.
    """
    with open(eval_json_path, 'r') as f:
        data = json.load(f)
    problems = {item['problem'].strip() for item in data if 'problem' in item}
    print(f"Loaded {len(problems)} problems from evaluation file.")
    return problems


def get_agent_traces_from_logfire(agent_name: str, model_name: str, limit: Optional[int] = 100, eval_problems: Optional[set] = None, lookback_days: int = 5) -> Dict[str, List[Dict[str, Any]]]:
    """
    Fetches message traces for a specific agent from Logfire.
    """
    if not LOGFIRE_API_KEY:
        raise ValueError("LOGFIRE_API_KEY environment variable not set.")

    headers = {
        "Authorization": f"Bearer {LOGFIRE_API_KEY}",
        "Content-Type": "application/json",
    }

    query = f"""
SELECT
    attributes->>'agent_name' as agent_name,
    attributes->'pydantic_ai.all_messages' as all_messages,
    attributes->'pydantic_ai.result' as result,
    start_timestamp,
    end_timestamp
FROM records
WHERE attributes->>'agent_name' LIKE '%{agent_name}%'
AND attributes->>'model_name' LIKE '%{model_name}%'
AND start_timestamp > NOW() - INTERVAL '{lookback_days} days'
ORDER BY start_timestamp DESC
"""
    
    if limit:
        query += f"\nLIMIT {limit}"

    # The Logfire API has its own row limit (default can be as low as 100).
    # Pass it explicitly to ensure we get all requested rows.
    api_limit = limit if limit else 10000  # 10000 is the API max
    params = {'sql': query, 'limit': api_limit}

    try:
        print(f"Fetching traces for agent '{agent_name}' from Logfire...")
        response = requests.get(f'{LOGFIRE_API_URL}/v1/query', headers=headers, params=params)
        response.raise_for_status()
        
        data = response.json()
        columns = data.get('columns', [])
        
        agent_name_col = next((col for col in columns if col['name'] == 'agent_name'), None)
        all_messages_col = next((col for col in columns if col['name'] == 'all_messages'), None)
        result_col = next((col for col in columns if col['name'] == 'result'), None)
        start_timestamp_col = next((col for col in columns if col['name'] == 'start_timestamp'), None)
        end_timestamp_col = next((col for col in columns if col['name'] == 'end_timestamp'), None)
        
        if not all_messages_col or not agent_name_col:
            print("Warning: Could not find all_messages or agent_name column in response")
            return {}
        
        num_records = len(all_messages_col['values'])
        print(f"Processing {num_records} records...")
        
        traces_by_agent: Dict[str, List[Dict[str, Any]]] = {}
        seen_problems_by_agent: Dict[str, Dict[str, bool]] = {}
        
        for idx in tqdm.tqdm(range(num_records), desc="Processing traces"):
            all_messages = all_messages_col['values'][idx]
            actual_agent_name = agent_name_col['values'][idx]
            
            if not all_messages or len(all_messages) < 2:
                continue
            
            try:
                # Extract problem from the first user message (assumed to be at index 1 after system prompt or 0)
                # Typically index 0 is user if no system, or index 1 if system. 
                # pydantic-ai usually has system prompt first if defined.
                # Let's check for the first message with role 'user'
                problem_content = ""
                for msg in all_messages:
                    if msg.get('role') == 'user':
                        # Get first text part
                        for part in msg.get('parts', []):
                             if part.get('type') == 'text':
                                 problem_content = part.get('content', '')
                                 break
                        if problem_content:
                            break
                
                if not problem_content:
                    continue
                
                problem_content = problem_content.strip()
                
                if actual_agent_name not in seen_problems_by_agent:
                    seen_problems_by_agent[actual_agent_name] = {}
                    traces_by_agent[actual_agent_name] = []

                if problem_content in seen_problems_by_agent[actual_agent_name]:
                    continue

                if eval_problems is not None and clean_problem(problem_content) not in eval_problems:
                    continue

                seen_problems_by_agent[actual_agent_name][problem_content] = True
                
                message_trace = extract_message_trace(all_messages)
                
                trace_obj = {
                    "problem": problem_content.strip(),
                    "agent_name": actual_agent_name,
                    "start_timestamp": start_timestamp_col['values'][idx] if start_timestamp_col else None,
                    "end_timestamp": end_timestamp_col['values'][idx] if end_timestamp_col else None,
                    "result": result_col['values'][idx] if result_col else None,
                    "message_trace": message_trace,
                    "metadata": {
                        "total_messages": len(message_trace),
                        "tool_calls": sum(1 for msg in message_trace for part in msg.get('parts', []) if part.get('type') == 'tool_call'),
                        "tool_responses": sum(1 for msg in message_trace for part in msg.get('parts', []) if part.get('type') == 'tool_call_response'),
                        "thinking_messages": sum(1 for msg in message_trace for part in msg.get('parts', []) if part.get('type') == 'thinking')
                    }
                }
                
                traces_by_agent[actual_agent_name].append(trace_obj)
                
            except (KeyError, IndexError, TypeError) as e:
                # print(f"Error processing record {idx}: {e}")
                continue
        
        for agent, traces in traces_by_agent.items():
            print(f"Successfully extracted {len(traces)} traces for agent '{agent}'")
        return traces_by_agent

    except requests.exceptions.RequestException as e:
        print(f"Error querying Logfire: {e}")
        return {}


def save_traces(traces_by_agent: Dict[str, List[Dict[str, Any]]], output_dir: str = "logs"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for agent_name, traces in traces_by_agent.items():
        if not traces:
            continue

        output_file = os.path.join(output_dir, f"{agent_name}_traces_{timestamp}.json")
        
        with open(output_file, 'w') as f:
            json.dump(traces, f, indent=2)
        
        print(f"\nTraces for agent '{agent_name}' saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Download agent message traces from Logfire")
    parser.add_argument("--agent-name", type=str, required=True, help="Name of the agent (LIKE query)")
    parser.add_argument("--model-name", type=str, default="", help="Name of the model (LIKE query)")
    parser.add_argument("--limit", type=int, default=100, help="Max traces (0 for all)")
    parser.add_argument("--output-dir", type=str, default="logs", help="Output directory")
    parser.add_argument("--eval-json", type=str, default=None, help="Path to evaluation JSON to filter traces by matching problems")
    parser.add_argument("--lookback-days", type=int, default=5, help="How many days back to query Logfire (default: 5)")

    args = parser.parse_args()
    limit = None if args.limit == 0 else args.limit
    eval_problems = load_eval_problems(args.eval_json) if args.eval_json else None

    traces_by_agent = get_agent_traces_from_logfire(args.agent_name, args.model_name, limit=limit, eval_problems=eval_problems, lookback_days=args.lookback_days)
    
    if traces_by_agent:
        save_traces(traces_by_agent, args.output_dir)
    else:
        print("No traces found.")

if __name__ == "__main__":
    main()
