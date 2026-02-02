from typing import List, Any, Dict, Tuple, Optional
import logging
import json
import os
import re
from datetime import datetime
from pydantic import BaseModel, Field

from src.services.base_agent import BaseAgent
from src.services.iterative_search_agent import IterativeSearchAgent, ResponseWrapper
from src.services.service_types import SamplerBase, SamplerResponse, MessageList

logger = logging.getLogger(__name__)

# --- Constants ---
LOG_DIR = "logs/gemini_3_pro_v4_generalized_runs"
DEFAULT_MAIN_MODEL = "gemini-2.5-flash"
DEFAULT_PROVIDER = "Google"

DECOMPOSITION_MODEL_NAME = "gemini-3-pro-preview"
DECOMPOSITION_PROVIDER = "Google"

SYNTHESIS_MODEL_NAME = "gemini-3-flash-preview"
SYNTHESIS_PROVIDER = "Google"

# --- Prompts ---
DECOMPOSITION_PROMPT = """You are an expert Query Decomposition Agent.
Your task is to break down a complex user query into a sequential list of simple, atomic questions.
These questions will be answered by a delegated agent that has access to both internal knowledge and web search tools.

### Instructions
1. **Analyze**: Determine if the user's query is complex (multi-step) or simple.
2. **Decompose**: Break the query into a list of minimal, self-contained questions.
3. **Sequential Order**: Order the questions logically. Earlier questions must provide the context needed for later questions.
4. **Dependencies**:
   - If a question depends on the result of a previous step, use the strict placeholder: `{{Answer to Step X}}` (where X is the 1-based index).
   - Do NOT include the actual answer or hallucinate names/entities that haven't been found yet.
5. **Phrasing**:
   - Write these as **natural language questions** (e.g., "Who is the CEO of Apple?"), NOT search keywords.
   - The delegated agent will decide whether to use its memory or a search engine based on your question.

### Examples

User Query: "Who is the current CEO of the company that created the iPhone, and how old are they?"
Reasoning: I need to first identify the company, then find its current CEO, and finally find that specific person's age.
Queries:
1. "Which company created the iPhone?"
2. "Who is the current CEO of {{Answer to Step 1}}?"
3. "How old is {{Answer to Step 2}}?"

User Query: "{cleaned_query}"
"""

SYNTHESIS_PROMPT = """You are a helpful assistant.
Original User Query: "{cleaned_query}"

Decomposition Reasoning: {decomposition_reasoning}

Results from Atomic Queries:
{context_str}

Your Task:
Synthesize a comprehensive and coherent answer to the original user query based on the results of the atomic queries.

**CRITICAL OUTPUT FORMAT:**
Your response must strictly follow this format:

Explanation: {{your detailed explanation for your final answer, citing the atomic results}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}
"""

SUB_AGENT_SYSTEM_INSTRUCTION = "You are a precise answering agent. Provide ONLY the final exact answer to the user's question. Use the 'explanation' field to provide your reasoning, but ensure the 'exact_answer' field contains ONLY the answer string."

class AtomicQueryList(BaseModel):
    queries: List[str] = Field(description="List of simple, atomic search queries. Each query should be self-contained and have no prerequisites.")
    reasoning: str = Field(description="Explanation of how the complex query was broken down.")

class AtomicResult(BaseModel):
    explanation: str = Field(description="Explanation or reasoning for the answer.")
    exact_answer: str = Field(description="The exact, concise answer to the query.")

class AgentWrapper:
    """
    Wraps a BaseAgent to enforce structured output (AtomicResult).
    Returns a combined string of explanation and exact answer to the caller
    so that clustering and gap analysis can use the explanation.
    """
    def __init__(self, base_agent: BaseAgent):
        self.base_agent = base_agent
        self.model_name = base_agent.model_name # Forward model_name

    def run(self, prompt: str):
        response = self.base_agent.run(prompt)
        # response.output is the typed output (AtomicResult)
        if hasattr(response, 'output') and isinstance(response.output, AtomicResult):
             # Include explanation in the output for better clustering/gap analysis
             combined_output = f"Exact Answer: {response.output.exact_answer}\nExplanation: {response.output.explanation}"
             return ResponseWrapper(combined_output)
        elif hasattr(response, 'output'):
             # Fallback if something weird happens (e.g. str)
             return ResponseWrapper(str(response.output))
        else:
             # Fallback for raw response without .output (older pydantic-ai?)
             # Usually response is RunResult
             return ResponseWrapper(str(response))

class GeneralizedIterativeSearchAgent(SamplerBase):
    """
    A generalized agent that breaks down a complex query into atomic queries,
    runs the IterativeSearchAgent on each atomic query in parallel,
    and aggregates the results.
    """
    def __init__(self, 
                 model: Any = None, # BaseAgent or similar
                 search_tool: Any = None, 
                 max_steps: int = 4,
                 device: str = "cuda",
                 load_history: bool = False,
                 main_model_name: str = DEFAULT_MAIN_MODEL):
        
        self.search_tool = search_tool
        self.max_steps = max_steps
        self.load_history = load_history
        
        # Compatibility with IterativeSearchAgent signature
        if model:
            self.main_model_name = model.model_name
            # We might want to capture provider too if BaseAgent exposes it, 
            # but for now we rely on model_name being sufficient for the internal agents
            # or we assume 'Google'/'gemini' defaults if not strictly passed.
            # Ideally BaseAgent should expose provider.
            self.provider_name = getattr(model, "provider_name", DEFAULT_PROVIDER)
        else:
            self.main_model_name = main_model_name
            self.provider_name = DEFAULT_PROVIDER
        
        # Agent for decomposition
        self.decomposition_agent = BaseAgent(
            agent_name="query_decomposition_agent",
            model_name=DECOMPOSITION_MODEL_NAME,
            provider_name=DECOMPOSITION_PROVIDER,
            output_type=AtomicQueryList
        )
        
        # Agent for final synthesis
        self.synthesis_agent = BaseAgent(
            agent_name="synthesis_agent",
            model_name=SYNTHESIS_MODEL_NAME,
            provider_name=SYNTHESIS_PROVIDER,
            output_type=str
        )

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        # Extract the last user message as the query
        user_query = ""
        for msg in reversed(message_list):
            if msg.get('role') == 'user':
                user_query = msg.get('content', "")
                break
        
        if not user_query:
            return SamplerResponse(
                response_text="No user query found.",
                actual_queried_message_list=message_list,
                response_metadata={"search_calls": 0}
            )

        # Run the generalized process
        final_answer, metadata = self.run(user_query)

        # Construct a synthetic message list
        convo = list(message_list)
        convo.append({"role": "assistant", "content": final_answer})

        # Wrap the answer
        response_wrapper = ResponseWrapper(final_answer)

        return SamplerResponse(
            response_text=response_wrapper,
            actual_queried_message_list=convo,
            response_metadata=metadata
        )

    def run(self, user_query: str) -> Tuple[str, Dict[str, Any]]:
        # 1. Decompose Query
        # Clean the query to remove the "Your response should be in..." part for decomposition
        cleaned_query = user_query.split("\n\nYour response should be in")[0]
        sanitized_original = re.sub(r'[^a-zA-Z0-9]', '_', cleaned_query[:50])
        
        logger.info(f"Decomposing query: {cleaned_query}")
        decomposition_prompt = DECOMPOSITION_PROMPT.format(cleaned_query=cleaned_query)
        try:
            decomp_response = self.decomposition_agent.run(decomposition_prompt)
            atomic_queries = decomp_response.output.queries
            decomposition_reasoning = decomp_response.output.reasoning
        except Exception as e:
            logger.error(f"Decomposition failed: {e}")
            # Fallback to original query
            atomic_queries = [cleaned_query]
            decomposition_reasoning = "Decomposition failed, using original query."

        logger.info(f"Atomic Queries: {atomic_queries}")
        logger.info(f"Reasoning: {decomposition_reasoning}")

        # 2. Run IterativeSearchAgent for each atomic query
        results = {}  # query -> {answer, metadata}
        total_search_calls = 0
        previous_context = ""
        step_answers = {} # 1-based index -> exact answer

        for i, q in enumerate(atomic_queries):
            # Resolve placeholders if possible (simple substitution)
            # We treat {Answer to Step X} where X is 1-based index
            for prev_idx in range(i):
                 step_num = prev_idx + 1
                 placeholder = f"{{Answer to Step {step_num}}}"
                 if placeholder in q and step_num in step_answers:
                     # Replace with the actual exact answer from that step
                     exact_ans = step_answers[step_num]
                     q = q.replace(placeholder, f"{exact_ans}")
            
            # Prepare query with context
            if previous_context:
                effective_query = f"Context from previous steps:\n{previous_context}\n\nCurrent Step Query: {q}"
            else:
                effective_query = q
            
            try:
                # Calculate informative log prefix
                log_prefix = f"{sanitized_original}_step_{i+1}"
                
                # We use the same helper, but pass the effective query and prefix
                ans, meta = self._run_single_atomic_query(effective_query, log_prefix=log_prefix)
                
                # Check for explicit error return
                if ans.startswith("Error"):
                     raise Exception(f"Sub-agent returned error: {ans}")

                results[q] = {
                    "answer": ans,
                    "metadata": meta
                }
                total_search_calls += meta.get("search_calls", 0)

                # Extract only the "Exact Answer" part for future placeholders
                match = re.search(r"Exact Answer:\s*(.*?)(?:\nExplanation:|$)", ans, re.DOTALL)
                exact_ans_for_step = match.group(1).strip() if match else ans
                step_answers[i+1] = exact_ans_for_step
                
                # Update context
                previous_context += f"Q: {q}\nA: {ans}\n\n"
            except Exception as e:
                logger.error(f"Error processing atomic query '{q}': {e}")
                # Critical failure propagation
                return f"Error: Processing failed at step '{q}' with error: {e}", {
                    "search_calls": total_search_calls,
                    "atomic_results": results
                }

        # 3. Synthesize Final Answer
        logger.info("Synthesizing final answer...")
        context_str = "\n".join([f"Q: {q}\nA: {data['answer']}" for q, data in results.items()])
        
        synthesis_prompt = SYNTHESIS_PROMPT.format(
            cleaned_query=cleaned_query,
            decomposition_reasoning=decomposition_reasoning,
            context_str=context_str
        )
        try:
            synthesis_response = self.synthesis_agent.run(synthesis_prompt)
            final_answer = synthesis_response.output
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            final_answer = f"Explanation: Synthesis failed due to error: {e}\nExact Answer: Error\nConfidence: 0%"

        # 4. Log Everything
        os.makedirs(LOG_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sanitized_query = re.sub(r'[^a-zA-Z0-9]', '_', user_query[:50])
        log_file_path = os.path.join(LOG_DIR, f"{timestamp}_{sanitized_query}.json")

        full_log = {
            "user_query": user_query,
            "timestamp": timestamp,
            "decomposition": {
                "atomic_queries": atomic_queries,
                "reasoning": decomposition_reasoning
            },
            "atomic_results": results,
            "final_answer": str(final_answer),
            "total_search_calls": total_search_calls
        }
        
        with open(log_file_path, 'w') as f:
            json.dump(full_log, f, indent=2)
        logger.info(f"Generalized run log saved to {log_file_path}")

        metadata = {
            "search_calls": total_search_calls,
            "atomic_queries_count": len(atomic_queries),
            "atomic_results": results
        }
        
        return str(final_answer), metadata

    def _run_single_atomic_query(self, query: str, log_prefix: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Helper to instantiate and run an IterativeSearchAgent for a single query.
        """
        # Create a dedicated model instance for this thread to ensure safety/isolation
        # although pydantic-ai agents might be safe, having separate 'model' instances 
        # (which wrap the same provider) is cleaner for the IterativeSearchAgent structure.
        
        # We use the same model_name as passed to __init__
        # We need to recreate the BaseAgent for the IterativeSearchAgent
        
        # Note: IterativeSearchAgent expects a 'model' that is a BaseAgent instance.
        model_instance = BaseAgent(
            model_name=self.main_model_name,
            provider_name=self.provider_name,
            agent_name=f"sub_agent_{re.sub(r'[^a-zA-Z0-9]', '_', query[:10])}",
            output_type=AtomicResult, # Use structured output
            system_prompt=SUB_AGENT_SYSTEM_INSTRUCTION
        )
        
        # Wrap the model to be compatible with IterativeSearchAgent's string expectation
        wrapped_model = AgentWrapper(model_instance)
        
        agent = IterativeSearchAgent(
            model=wrapped_model,
            search_tool=self.search_tool,
            max_steps=self.max_steps,
            load_history=self.load_history,
        )
        
        # We need to wrap the response because IterativeSearchAgent returns (str, dict)
        # But wait, IterativeSearchAgent.run returns (str, metadata).
        return agent.run(query, log_prefix=log_prefix)
    
    def _pack_message(self, content, role="user"):
        return {"content": content, "role": role}
