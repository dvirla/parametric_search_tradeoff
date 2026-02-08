import os
import sys
import json
import glob
from tqdm import tqdm
from dotenv import load_dotenv

# Ensure the project root is in sys.path
sys.path.append(os.getcwd())

from src.services.base_agent import BaseAgent
from src.services.agent_sampler import AgentAsSampler
from src.services.qa_eval import STANDARD_GRADER_TEMPLATE
import re
from google import genai
from google.genai import types

load_dotenv()

# --- Configuration ---

NQ_DATA_PATH = "data/NQ-open.train.jsonl"
UPDATED_NQ_DATA_PATH = "data/NQ-open.train.updated.jsonl"
LOGS_DIR = "logs/natural_questions"

# --- Helpers ---

def get_unique_questions_from_logs(logs_dir):
    questions = set()
    files = glob.glob(os.path.join(logs_dir, "**/*.json"), recursive=True)
    
    log_files = []
    
    for fpath in files:
        if "trace" in fpath or "analysis" in fpath:
            continue
            
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
                if not isinstance(data, list):
                    continue
                
                # Check if it looks like an eval result
                if len(data) > 0 and "problem" in data[0]:
                    log_files.append(fpath)
                    for entry in data:
                        if "problem" in entry:
                            questions.add(entry["problem"])
        except Exception:
            pass
            
    return questions, log_files

def load_nq_data(path):
    data_map = {} # question -> entry
    all_lines = []
    if os.path.exists(path):
        with open(path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    # NQ dataset uses 'question' and 'answer'
                    q = entry.get("question")
                    if q:
                        data_map[q] = entry
                    all_lines.append(entry)
                except:
                    pass
    return data_map, all_lines

def solve_question(client, question, model_name):
    prompt = f"""
    Question: {question}
    
    Please provide the exact, up-to-date correct answer to this question.
    Use the search tool to verify facts if necessary.
    
    Output strictly the answer string. Do not include "The answer is" or punctuation unless part of the answer.
    """
    try:
        generate_content_config = types.GenerateContentConfig(
            tools=[types.Tool(google_search_retrieval=types.GoogleSearchRetrieval())],
            thinking_config=types.ThinkingConfig(
                thinking_level="HIGH",
            ),
            system_instruction="You are an expert fact-checker. Your goal is to provide the single most accurate, up-to-date, and concise answer to the user's question.",
        )
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=generate_content_config,
        )
        return response.text.strip()
    except Exception as e:
        print(f"Error solving '{question}': {e}")
        return None

def grade_sample(grader_model, question, correct_answer, response_text):
    grader_prompt = STANDARD_GRADER_TEMPLATE.format(
        question=question,
        correct_answer=correct_answer,
        response=response_text
    )

    prompt_messages = [
        grader_model._pack_message(content=grader_prompt, role="user")
    ]
    try:
        sampler_response = grader_model(prompt_messages)
        grading_output = sampler_response.response_text.output
        match = re.search(r"correct:\s*(yes|no)", grading_output, re.IGNORECASE)
        return match.group(1).lower() if match else "no"
    except Exception as e:
        print(f"Grading error: {e}")
        return "no"

def main():
    print("--- Starting NQ Update and Re-eval ---")

    # 1. Initialize Agents
    print(f"Initializing Solver Agent (google-genai client with gemini-3-flash-preview)...")
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    solver_model_name = 'gemini-3-flash-preview'
    
    print(f"Initializing Grader Agent (gpt-oss:20b)...")
    grader_agent_raw = BaseAgent(
        provider_name='ollama',
        model_name='gpt-oss:20b', 
        agent_name="grader_agent"
    )
    grader_agent = AgentAsSampler(grader_agent_raw)

    # 2. Identify Questions
    print("Scanning logs for questions...")
    unique_questions, log_files_to_process = get_unique_questions_from_logs(LOGS_DIR)
    print(f"Found {len(unique_questions)} unique questions in {len(log_files_to_process)} log files.")
    
    # 3. Load Dataset
    print(f"Loading {NQ_DATA_PATH}...")
    nq_map, nq_list = load_nq_data(NQ_DATA_PATH)
    print(f"Loaded {len(nq_list)} entries from dataset.")

    # 4. Generate New Ground Truths
    print("Generating new ground truths...")
    new_ground_truths = {} # question -> new_answer
    
    # We only update questions that are actually in the logs
    for q in tqdm(unique_questions, desc="Solving"):
        # Check if we already have a "good" answer? No, we assume existing are potentially bad.
        # But we can check if the question exists in the original dataset
        if q not in nq_map:
            # Maybe normalization issue?
            # For now, we only process exact matches or if we can map it.
            # If it's not in NQ map, we can't update NQ file for it easily.
            pass
        
        answer = solve_question(client, q, solver_model_name)
        if answer:
            new_ground_truths[q] = answer

    # 5. Update NQ Dataset
    print(f"Writing updated entries to {UPDATED_NQ_DATA_PATH}...")
    solved_entries = []
    for entry in nq_list:
        q = entry.get("question")
        if q in new_ground_truths:
            entry["answer"] = [new_ground_truths[q]] # Replace with new single correct answer
            solved_entries.append(entry)
            
    with open(UPDATED_NQ_DATA_PATH, 'w') as f:
        for entry in solved_entries:
            f.write(json.dumps(entry) + "\n")
    print(f"Saved {len(solved_entries)} updated entries to {UPDATED_NQ_DATA_PATH}.")

    # 6. Re-evaluate Logs
    print("Re-evaluating logs...")
    for log_path in log_files_to_process:
        print(f"Processing {log_path}...")
        try:
            with open(log_path, 'r') as f:
                data = json.load(f)
            
            modified = False
            for entry in tqdm(data, desc="Grading"):
                prob = entry.get("problem")
                if prob in new_ground_truths:
                    new_ans = new_ground_truths[prob]
                    
                    # Update correct answer in log
                    entry["correct_answer"] = str([new_ans]) # Format as list string to match old style if needed
                    
                    # Re-grade
                    response = entry.get("sampler_response", "")
                    is_correct = grade_sample(grader_agent, prob, new_ans, response)
                    
                    entry["sampler_correct"] = (is_correct == "yes")
                    if "metrics" in entry:
                        entry["metrics"]["correct"] = (is_correct == "yes")
                    
                    modified = True
            
            if modified:
                with open(log_path, 'w') as f:
                    json.dump(data, f, indent=4)
                print(f"Updated {log_path}")
        except Exception as e:
            print(f"Failed to process {log_path}: {e}")

    print("Done.")

if __name__ == "__main__":
    main()
