import os
import sys
import json
import argparse
from tqdm import tqdm
from pydantic import BaseModel, Field
from typing import Optional, List

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.services.base_agent import BaseAgent

class Turn(BaseModel):
    role: str = Field(description="Role of the speaker: either 'user' or 'assistant'")
    content: str = Field(description="Content of the message")

class Conversation(BaseModel):
    history: list[Turn] = Field(description="List of turns in the conversation")

# New models for search templates
class SearchPart(BaseModel):
    type: str = Field(description="Must be one of: 'text', 'tool_call', 'tool_response'")
    content: Optional[str] = Field(default=None, description="Text content for 'text' or 'tool_response' parts")
    tool_call_id: Optional[str] = Field(default=None, description="A unique ID for the tool call (e.g. 'call_abc123')")
    tool_name: Optional[str] = Field(default=None, description="Name of the tool, e.g. 'search'")
    arguments: Optional[str] = Field(default=None, description="JSON string of arguments for the tool call, e.g. '{\"query\": \"...\"}'")

class SearchTurn(BaseModel):
    role: str = Field(description="Role of the speaker: 'user', 'assistant', or 'tool'")
    content: Optional[str] = Field(default=None, description="Content of the message (for simple user messages)")
    parts: Optional[List[SearchPart]] = Field(default=None, description="Parts of the message (for assistant tool calls and tool responses)")

class SearchTemplate(BaseModel):
    history: List[SearchTurn] = Field(description="A single multi-turn conversation history")

class SearchTemplatesList(BaseModel):
    templates: List[SearchTemplate] = Field(description="List of 5 conversation templates")

def setup_args():
    parser = argparse.ArgumentParser(description="Synthesize multi-turn distractors for QA evaluation.")
    parser.add_argument("--dataset_path", type=str, required=False, help="Path to input dataset JSONL (e.g., data/frames_benchmark.jsonl)")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSON(L)")
    parser.add_argument("--n", type=int, default=None, help="Test mode: run only N examples")
    parser.add_argument("--model", type=str, default="gemini-3.1-pro-preview", help="Model name for synthesis")
    parser.add_argument("--provider", type=str, default="Google", help="Provider name (e.g. Google, ollama, OpenAI, Anthropic)")
    parser.add_argument("--condition", type=str, choices=["chit-chat", "topical", "tool-heavy", "search-templates"], default="topical", help="Type of distractor to generate")
    parser.add_argument("--num_turns", type=int, default=4, help="Total number of messages in the history (e.g., 4 means 2 user, 2 assistant)")
    return parser.parse_args()

def get_prompt(condition: str, target_question: str, gold_answer: str, num_turns: int) -> str:
    base = f"You are generating a synthetic conversation history between a 'user' and an 'assistant' with exactly {num_turns} messages.\n"
    base += f"This conversation will happen right BEFORE the user pivots and asks the following target question: \"{target_question}\"\n"
    base += f"The true answer to this target question is: \"{gold_answer}\"\n\n"
    
    if condition == "chit-chat":
        base += "Condition: Unrelated Realistic Task (Chit-Chat).\n"
        base += "Do NOT generate simple greetings like 'How are you?'. Instead, generate a realistic, practical interaction where the user asks the assistant to perform a task COMPLETELY UNRELATED to the target question. Examples: drafting a professional email, debugging a short Python script, providing a recipe, or summarizing a general concept. The assistant should provide a helpful, detailed, and realistic response."
    elif condition == "topical":
        base += "Condition: Topical but Strictly Tangential.\n"
        base += "The conversation should involve the general broad topic or main entity of the target question, BUT it must be strictly tangential. \n"
        base += "CRITICAL CONSTRAINTS:\n"
        base += f"1. You MUST NOT include the true answer (\"{gold_answer}\") anywhere in the conversation.\n"
        base += "2. You MUST NOT include intermediate reasoning steps, sub-answers, or specific entities that would make solving the target question easier.\n"
        base += "3. The user should ask about a completely different aspect of the topic, and the assistant should answer that specific tangential question realistically."
    elif condition == "tool-heavy":
        base += "Condition: Tool-Biased History.\n"
        base += "Generate a realistic interaction where the user asks for highly specific, up-to-date, or obscure information that clearly requires web search. The assistant's response MUST simulate having used a search tool (e.g., 'Based on my search results...', 'According to recent sources...'). The topic can be generic or somewhat related, but must NOT reveal the answer to the target question."
        
    base += "\n\nGenerate the conversation history. It must alternate between 'user' and 'assistant', starting with the 'user'. Make the assistant sound like a standard helpful AI."
    return base

def get_search_templates_prompt() -> str:
    return """You are generating exactly 5 distinct synthetic multi-turn conversation templates between a 'user', an 'assistant', and a 'tool'.
Each conversation should represent a realistic interaction where the user asks a question, the assistant uses a search tool, the tool responds with search results, and the assistant provides a final answer. 
The topics should be completely generic and unrelated to each other (e.g., history, science, coding, daily tasks).

For each conversation template, the history should ideally follow this structure:
1. User: Asks a question requiring up-to-date or specific information.
2. Assistant: Outputs a tool_call part for the 'search' tool.
3. Tool: Outputs a tool_response part with the search results (mocked).
4. Assistant: Outputs a text part with the final answer based on the search results.

CRITICAL CONSTRAINTS:
- You must use the `parts` field for assistant tool calls and tool responses.
- For a tool call, the `role` is 'assistant', the `type` of the part is 'tool_call', provide a `tool_call_id`, `tool_name` ('search'), and `arguments` (JSON string like '{"query": "..."}').
- For a tool response, the `role` is 'tool', the `type` of the part is 'tool_response', provide the SAME `tool_call_id`, and `content` (a string mocking the search results, e.g. a JSON list of titles/snippets).
- For text messages, you can either use `role: 'user'` with the `content` field, or `role: 'assistant'` with `parts: [{"type": "text", "content": "..."}]`.
"""

def generate_search_templates(args):
    print(f"Generating 5 search templates using {args.provider}/{args.model}...")
    
    agent = BaseAgent(
        provider_name=args.provider,
        model_name=args.model,
        agent_name="search_templates_generator",
        output_type=SearchTemplatesList
    )
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    prompt = get_search_templates_prompt()
    
    try:
        result = agent.run(prompt)
        
        output_data = getattr(result, "data", None) or getattr(result, "output", None)
        
        if output_data and isinstance(output_data, SearchTemplatesList):
            templates_dicts = []
            for template in output_data.templates:
                history_list = []
                for turn in template.history:
                    turn_dict = {"role": turn.role}
                    if turn.content is not None:
                        turn_dict["content"] = turn.content
                    if turn.parts is not None:
                        turn_dict["parts"] = [p.dict(exclude_none=True) for p in turn.parts]
                    history_list.append(turn_dict)
                templates_dicts.append(history_list)
            
            with open(args.output, "w") as out_f:
                json.dump(templates_dicts, out_f, indent=2)
            print(f"Successfully generated and saved 5 search templates to {args.output}")
        else:
            print(f"Warning: Failed to parse structured output. Result: {result}")
    except Exception as e:
        print(f"Error generating templates:\n{e}")

def main():
    args = setup_args()
    
    if args.condition == "search-templates":
        generate_search_templates(args)
        return
        
    # Load data
    data = []
    if not args.dataset_path or not os.path.exists(args.dataset_path):
        print(f"Error: dataset path {args.dataset_path} does not exist.")
        sys.exit(1)
        
    with open(args.dataset_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
            
    if args.n is not None:
        data = data[:args.n]
        print(f"Test mode: processing {args.n} examples.")
        
    print(f"Loaded {len(data)} examples. Generating '{args.condition}' distractors using {args.provider}/{args.model}...")
    
    # Initialize the agent with Structured Output
    agent = BaseAgent(
        provider_name=args.provider,
        model_name=args.model,
        agent_name="distractor_generator",
        output_type=Conversation
    )
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    out_f = open(args.output, "w")
    
    for row in tqdm(data):
        # Handle different naming conventions in datasets
        target_q = row.get("problem") or row.get("text") or row.get("question") or ""
        gold_ans = str(row.get("gold answer") or row.get("answer") or "")
        
        prompt = get_prompt(args.condition, target_q, gold_ans, args.num_turns)
        
        try:
            result = agent.run(prompt)
            
            # The result from pydantic_ai Agent.run_sync() is a RunResult, 
            # where .data contains the parsed Conversation object.
            # (Note: BaseAgent's run() returns this result or a partial salvage result)
            if hasattr(result, "data") and isinstance(result.data, Conversation):
                history_dicts = [{"role": t.role, "content": t.content} for t in result.data.history]
                row["history"] = history_dicts
                row["condition"] = args.condition
                
                out_f.write(json.dumps(row) + "\n")
                out_f.flush()
            else:
                print(f"\nWarning: Failed to parse structured output for question: {target_q[:30]}")
                print(f"Result type: {type(result)}")
                if hasattr(result, 'data'): print(f"Result data type: {type(result.data)}")
                if hasattr(result, 'output'): print(f"Result output: {result.output}")
                
        except Exception as e:
            print(f"\nError processing question: {target_q[:30]}\n{e}")
            
    out_f.close()
    print(f"Done. Wrote results to {args.output}")

if __name__ == '__main__':
    main()
