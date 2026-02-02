from typing import List, Dict, Tuple, Any, Optional
import math
import re
import logging
import json
import os
import random
from datetime import datetime
from pydantic import BaseModel, Field
from tqdm import tqdm

from src.services.base_agent import BaseAgent
from src.services.service_types import SamplerBase, SamplerResponse, MessageList

# Configure logger
logger = logging.getLogger(__name__)

# --- Constants ---
LOG_DIR = "logs/gemini_3_pro_v5_trajectory_analysis"
CLUSTERING_MODEL_NAME = "gpt-oss:20b"
CLUSTERING_PROVIDER = "ollama"
NEGATIVE_CONSENSUS_MODEL_NAME = "gpt-oss:20b"
NEGATIVE_CONSENSUS_PROVIDER = "ollama"
GAP_AGENT_MODEL_NAME = "gemini-3-pro-preview"
GAP_AGENT_PROVIDER = "Google"
DISTILLATION_MODEL_NAME = "gemini-3-flash-preview"
DISTILLATION_PROVIDER = "Google"
UNCERTAINTY_THRESHOLD = 0.55

# --- Prompts ---
NEGATIVE_CONSENSUS_PROMPT = """User Question: \"{cleaned_question}\"\nDraft Answer: \"{draft}\"\n\nDoes this draft answer claim that the requested information does not exist, is not known, or that there is no answer matching the criteria?\nTrue if it claims non-existence or \"I don't know\".\nFalse if it provides a specific affirmative answer.\n"""

DRAFT_GENERATION_INSTRUCTION = (
    "\nIMPORTANT: Provide a concise, self-contained answer. "
    "Do NOT use pronouns like 'he', 'she', 'it', or 'they' without explicit antecedents. "
    "Use full names and specific entities (e.g., say 'Apple's revenue' instead of 'its revenue')."
)

CLUSTERING_PROMPT = """User Question: \"{cleaned_question}\"\n\nI have generated {num_drafts} draft answers to this question.\nDrafts:\n{drafts_text}\n\nYour Task:\nGroup these drafts into clusters based on semantic equivalence.\n- If the exact answers are the same, it is a strong signal they belong in the same cluster.\n- Drafts in the same cluster must mean the same thing and contain the same key facts.\n- If drafts contradict each other or have different key facts, they must be in different clusters.\n- Minor wording differences or level of detail (if facts are consistent) should be ignored (same cluster).\n\nReturn the clusters as a list of lists of indices (1-based).\nExample: [[1, 3], [2, 4], [5]]\n"""

DISTILLATION_PROMPT = """You are a Knowledge Distillation Engine.\nYour goal is to maintain a concise, noise-free list of facts gathered so far.\n\nCurrent Question: \"{cleaned_question}\"\n\nExisting Knowledge State:\n{current_knowledge}\n\nNew Search Snippets (Raw Data):\n{new_snippets}\n\n### Instructions\n1. **Integration**: Read the `New Search Snippets`. Do they contain *new* information that answers the Current Question?\n   - If YES: Merge it into the Existing Knowledge.\n   - If NO (irrelevant, ads, or duplicate): Ignore it.\n2. **Conflict Resolution**:\n   - If a new snippet contradicts existing knowledge, NOTE the conflict explicitly (e.g., \"Conflict: Source A says X, Source B says Y\").\n3. **Noise Removal**:\n   - Remove any \"meta\" text (e.g., \"cookies\", \"sign up\", \"404 error\").\n   - Remove facts that are clearly irrelevant to the specific User Question.\n4. **Output Format**:\n   - Return ONLY the updated list of facts as a bulleted list.\n   - Do not explain your process. Just give the clean data.\n\n### Example\nExisting: \"- Apple CEO is Tim Cook.\"\nNew Snippet: \"Tim Cook (born 1960) is an executive... click to read more.\"\nOutput:\n- Apple CEO is Tim Cook.\n- Tim Cook was born in 1960.\n"""

GAP_ANALYSIS_INITIAL_PROMPT = """You are a Search Query Specialist.\nUser Question: \"{cleaned_question}\"\n\nYour Task:\n1. **Consult Internal Knowledge**: Before searching, ask yourself: \"Do I suspect I know the answer?\"\n   - If **YES** (e.g., you recall the CEO is Tim Cook): Formulate a **Verification Query** to confirm this specific fact (e.g., \"Tim Cook Apple CEO current status\"). This is much more effective than a generic query.\n   - If **NO**: Formulate a **Discovery Query** using broad, high-value keywords.\n2. **Snippet Optimization**: Design the query so the answer appears in a short snippet (approx. 200 chars).\n   - Use specific names/dates if your internal knowledge provides them.\n"""

GAP_ANALYSIS_NEGATIVE_CONSENSUS_PROMPT = """You are an expert Researcher solving a \"Missing Information\" problem.\nUser Question: \"{cleaned_question}\"\nPrevious Queries:\n{history_str}\n\nFailed Queries (No Results):\n{failed_queries_str}\n\nFailed Snippets:\n{distilled_knowledge}\n\nYour Task:\n1. **Leverage Internal Associations**: The previous queries failed to retrieve relevant information. Use your internal training data to find:\n   - **Synonyms/Technical Terms**: (e.g., if \"heart attack\" failed, try \"myocardial infarction\").\n   - **Parent Concepts**: (e.g., if \"specific python error\" failed, try the library name or \"exception hierarchy\").\n   - **Related Entities**: (e.g., if a person is unsearchable, search for their known employer or spouse).\n2. **Formulate a Pivot Query**: Create a NEW query based on these internal associations.\n3. **Constraint**: Do not repeat the semantic meaning of failed queries.\n"""

GAP_ANALYSIS_CONFLICT_PROMPT = """You are a Fact-Checking Editor resolving a data conflict.\nUser Question: \"{cleaned_question}\"\nConflicting Drafts:\n{clusters_str}\n\nPrevious Queries:\n{history_str}\n\nContext:\n{distilled_knowledge}\n\nYour Task:\n1. **Consult Internal Knowledge**: Look at the conflicting options.\n   - Does one align with your training data? (Likely the \"conventional wisdom\").\n   - Does one seem like a recent update or a common misconception?\n2. **Formulate a Verification Query**: \n   - Do NOT just search the topic again.\n   - Target the **discriminator**. \n   - Example: If Draft A says \"Released in 2020\" (your internal knowledge matches this) and Draft B says \"Delayed to 2024\", your query should be: \"Product X release date delay 2024 confirmation\".\n   - **Goal**: Prove or disprove the *change* from your internal baseline.\n"""

VERIFICATION_QUERY_PROMPT = """You are a meticulous Fact-Checking Assistant.\nYour goal is to generate a highly specific search query to verify a single, potentially controversial claim.\n\nUser Question: \"{cleaned_question}\"\nClaim to Verify (from a minority of drafts): \"{claim}\"\n\nPrevious Successful Queries:\n{history_str}\n\nFailed Queries (No Results):\n{failed_queries_str}\n\nInstructions:\n1.  **Analyze Failed Queries**: Carefully review the \"Failed Queries\" list. These queries yielded NO results. You MUST NOT generate a query that is semantically similar to any of them. If a direct approach failed, you must pivot to a different angle (e.g., check related entities, parent concepts, or different specific keywords).\n2.  **Isolate the Core Fact**: Identify the single most important and verifiable fact in the \"Claim to Verify\". This might be a date, a name, a number, a specific event, etc.\n3.  **Formulate a Targeted Query**: Create a search query that directly asks about that core fact. The query should be designed to get a definitive \"yes\" or \"no\" answer or a specific correcting value from a search result snippet.\n4.  **Avoid Broad Keywords**: Do not use generic terms from the original user question. Focus only on the entities in the specific claim.\n5.  **Productivity Check**: Ask yourself: \"Is this query likely to return a result given that the previous ones failed?\" If not, broaden it slightly or check a related fact that implies the answer.\n"""

VALIDATION_GATE_PROMPT = """You are a strict Validation Gate. Your job is to determine if new evidence explicitly confirms a specific hypothesis.\n\nOriginal User Question: \"{cleaned_question}\"\nHypothesis to Validate: \"{hypothesis}\"\n\nSearch Result Snippets:\n---\n{snippets}\n---\n\nInstructions:\n1.  **Strictly Compare**: Read the \"Hypothesis to Validate\" and the \"Search Result Snippets\".\n2.  **Confirm or Refute**: Does any snippet **directly and explicitly** support the core facts of the hypothesis?\n    - **CONFIRMED**: Only if the snippet contains the same key entities, relationships, and values.\n    - **REFUTED**: If the snippet contradicts the hypothesis or provides no relevant information.\n3.  **Return a Boolean**: Your answer must be either `true` (CONFIRMED) or `false` (REFUTED). Do not provide any explanation.\n\nExample:\nHypothesis: \"The capital of Australia is Sydney.\"\nSnippet: \"Canberra has been the capital of Australia since 1913...\"\nYour Judgement: `false`\n"""

CONSENSUS_VERIFICATION_PROMPT = """You are a meticulous Fact-Checking Assistant.\nYour goal is to generate a highly specific search query to verify a strong consensus answer.\n\nUser Question: \"{cleaned_question}\"\nClaim to Verify: \"{claim}\"\n\nPrevious Successful Queries:\n{history_str}\n\nFailed Queries (No Results):\n{failed_queries_str}\n\nInstructions:\n1.  **Formulate a Targeted Query**: Create a search query that directly asks about the core facts in the \"Claim to Verify\".\n2.  **Verify**: The query should aim to confirm the specific details (dates, names, numbers) in the claim.\n3.  **Avoid Redundancy**: If previous queries already covered this, try a slightly different angle or specific detail to double-check.\n"""

# --- Pydantic Models ---
class ClusteringResult(BaseModel):
    clusters: List[List[int]] = Field(description="List of clusters (1-based indices).")

class GapAnalysisResult(BaseModel):
    query: str = Field(description="The search query to resolve the conflict or missing info.")

class NegativeConsensusResult(BaseModel):
    is_negative: bool = Field(description="True if the answer claims info is missing/unknown.")

class DistillationResult(BaseModel):
    facts: List[str] = Field(description="Updated list of concise, noise-free facts.")

class VerificationQueryResult(BaseModel):
    query: str = Field(description="Specific search query to verify the claim.")

class ValidationResult(BaseModel):
    is_confirmed: bool = Field(description="True if snippets confirm the hypothesis.")

# --- Response Wrapper ---
class ResponseWrapper:
    def __init__(self, output: str):
        self.output = output
    def __str__(self):
        return self.output

class IterativeSearchAgent(SamplerBase):
    """
    A cleaner, refactored implementation of IterativeSearchAgent.
    """

    def __init__(self, model: BaseAgent, search_tool: Any, max_steps: int = 4, device: str = "cuda", load_history: bool = False):
        self.model = model
        self.search_tool = search_tool
        self.max_steps = max_steps
        # load_history is ignored in this clean version for simplicity, unless requested.
        self.history: List[Tuple[str, str]] = []  # (Query, Result)
        self.search_call_count = 0
        self.distilled_facts: List[str] = []

        # Sub-agents
        self.clustering_agent = BaseAgent(agent_name="clustering", provider_name=CLUSTERING_PROVIDER, model_name=CLUSTERING_MODEL_NAME, output_type=ClusteringResult)
        self.negative_consensus_agent = BaseAgent(agent_name="negative_consensus", provider_name=NEGATIVE_CONSENSUS_PROVIDER, model_name=NEGATIVE_CONSENSUS_MODEL_NAME, output_type=NegativeConsensusResult)
        self.gap_agent = BaseAgent(agent_name="gap_analysis", provider_name=GAP_AGENT_PROVIDER, model_name=GAP_AGENT_MODEL_NAME, output_type=GapAnalysisResult)
        self.distillation_agent = BaseAgent(agent_name="distillation", provider_name=DISTILLATION_PROVIDER, model_name=DISTILLATION_MODEL_NAME, output_type=DistillationResult)
        self.verification_query_agent = BaseAgent(agent_name="verification", provider_name=GAP_AGENT_PROVIDER, model_name=GAP_AGENT_MODEL_NAME, output_type=VerificationQueryResult)
        self.validation_gate_agent = BaseAgent(agent_name="validation", provider_name=DISTILLATION_PROVIDER, model_name=DISTILLATION_MODEL_NAME, output_type=ValidationResult)

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        user_query = ""
        for msg in reversed(message_list):
            if msg.get('role') == 'user':
                user_query = msg.get('content', "")
                break
        
        if not user_query:
            return SamplerResponse("No user query found.", message_list, {"search_calls": 0})

        final_answer, metadata = self.run(user_query)
        
        convo = list(message_list)
        convo.append({"role": "assistant", "content": final_answer})
        
        response_metadata = {"search_calls": self.search_call_count}
        response_metadata.update(metadata)
        
        return SamplerResponse(ResponseWrapper(final_answer), convo, response_metadata)

    def run(self, user_query: str, log_prefix: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        self.history = []
        self.search_call_count = 0
        self.distilled_facts = []
        context = ""
        trajectory_logs = []
        
        cleaned_question = user_query[:user_query.find("\n\nYour response should be in")]
        if cleaned_question == "" or "\n\nYour response should be in" not in user_query:
            cleaned_question = user_query

        logger.info(f"Starting CleanIterativeSearchAgent for: {cleaned_question[:50]}...")

        for step in range(self.max_steps):
            logger.info(f"--- Step {step + 1}/{self.max_steps} ---")
            step_log = {"step": step + 1, "timestamp": datetime.now().isoformat()}

            # 1. Generate Drafts
            drafts = self._generate_drafts(user_query, context)
            step_log["drafts"] = drafts
            
            # 2. Calculate Uncertainty
            uncertainty_score, clusters = self._calculate_uncertainty(drafts, cleaned_question)
            step_log["uncertainty_score"] = uncertainty_score
            step_log["clusters"] = clusters
            
            # 3. Check Stopping Criteria
            effective_threshold = 0.1 if step == 0 else UNCERTAINTY_THRESHOLD
            
            should_stop, stop_reason, final_answer, is_negative_consensus = self._check_stopping_criteria(
                uncertainty_score, effective_threshold, clusters, cleaned_question, step
            )
            
            if should_stop:
                logger.info(f"Stopping criterion met: {stop_reason}")
                step_log["action"] = stop_reason
                trajectory_logs.append(step_log)
                self._save_logs(user_query, final_answer, stop_reason, trajectory_logs, log_prefix)
                return final_answer, {"stop_reason": stop_reason, "trajectory": trajectory_logs, "search_calls": self.search_call_count}

            # 4. Perform Search (Dissenter or Broad)
            logger.info("High uncertainty. Initiating search...")
            new_context_segment, search_log = self._perform_search(
                step, cleaned_question, clusters, context, uncertainty_score, effective_threshold, is_negative_consensus
            )
            step_log["search_log"] = search_log
            
            # 5. Distill & Update Context
            if new_context_segment:
                context += new_context_segment
                self._update_distilled_facts(cleaned_question, new_context_segment)
                step_log["distilled_facts"] = self.distilled_facts
            
            trajectory_logs.append(step_log)

        # Max steps reached
        final_answer = self._select_random_from_largest_cluster(clusters) if clusters else "No answer derived."
        self._save_logs(user_query, final_answer, "max_steps", trajectory_logs, log_prefix)
        return final_answer, {"stop_reason": "max_steps", "trajectory": trajectory_logs, "search_calls": self.search_call_count}

    def _save_logs(self, user_query: str, final_answer: str, stop_reason: str, trajectory_logs: list, log_prefix: Optional[str] = None):
        os.makedirs(LOG_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if log_prefix:
            log_filename = f"{timestamp}_{log_prefix}.json"
        else:
            sanitized_query = re.sub(r'[^a-zA-Z0-9]', '_', user_query[:50])
            log_filename = f"{timestamp}_{sanitized_query}.json"
        
        log_file_path = os.path.join(LOG_DIR, log_filename)
        full_log = {
            "user_query": user_query,
            "timestamp": timestamp,
            "final_answer": final_answer,
            "stop_reason": stop_reason,
            "trajectory": trajectory_logs
        }
        try:
            with open(log_file_path, 'w') as f:
                json.dump(full_log, f, indent=2)
            logger.info(f"Trajectory logs saved to {log_file_path}")
        except Exception as e:
            logger.error(f"Failed to save trajectory logs: {e}")

    def _generate_drafts(self, user_query: str, context: str, n: int = 5) -> List[str]:
        base_prompt = f"Answer the question to the best of your knowledge.\n\nQuestion: {user_query}"
        if context:
            base_prompt += "\n\nThe following are search results that may or may not be useful:"
            base_prompt += context
        
        drafts = []
        for _ in range(n):
            response = self.model.run(base_prompt + DRAFT_GENERATION_INSTRUCTION)
            drafts.append(str(response.output))
        return drafts

    def _calculate_uncertainty(self, drafts: List[str], cleaned_question: str) -> Tuple[float, Dict[int, List[str]]]:
        if not drafts: return 0.0, {}
        if len(drafts) == 1: return 0.0, {0: [drafts[0]]}
        
        drafts_text = "\n".join([f"Draft {i+1}:\n{d}" for i, d in enumerate(drafts)])
        prompt = CLUSTERING_PROMPT.format(cleaned_question=cleaned_question, num_drafts=len(drafts), drafts_text=drafts_text)
        
        try:
            res = self.clustering_agent.run(prompt)
            # Map 1-based indices to 0-based drafts
            clusters_map = {}
            seen = set()
            for i, cluster in enumerate(res.output.clusters):
                c_drafts = []
                for idx in cluster:
                    if 0 <= idx - 1 < len(drafts):
                        c_drafts.append(drafts[idx - 1])
                        seen.add(idx - 1)
                if c_drafts: clusters_map[i] = c_drafts
            
            # Add leftovers
            next_id = len(clusters_map)
            for i, d in enumerate(drafts):
                if i not in seen:
                    clusters_map[next_id] = [d]
                    next_id += 1
            
            # Entropy
            entropy = 0.0
            n = len(drafts)
            for c_drafts in clusters_map.values():
                p = len(c_drafts) / n
                entropy -= p * math.log(p + 1e-10)
            
            return entropy, clusters_map
        except Exception as e:
            logger.error(f"Clustering error: {e}")
            return 1.0, {i: [d] for i, d in enumerate(drafts)}

    def _check_stopping_criteria(self, uncertainty: float, threshold: float, clusters: Dict[int, List[str]], cleaned_question: str, step: int) -> Tuple[bool, str, str, bool]:
        if uncertainty <= threshold:
            largest_cluster = max(clusters.values(), key=len)
            is_negative = self._check_negative_consensus(largest_cluster[0], cleaned_question)
            if not is_negative:
                # If we are at step 0, we want to VERIFY this consensus instead of stopping immediately.
                if step == 0:
                    return False, "", "", False
                return True, "low_uncertainty", self._select_random_from_largest_cluster(clusters), False
            else:
                return False, "", "", True # Continue to search (Broad)
        
        if step == 1: # Check if we have results after 2 steps (0 and 1)
            has_results = any(r.strip() for _, r in self.history)
            if not has_results:
                return True, "no_results_after_2_steps", self._select_random_from_largest_cluster(clusters), False

        return False, "", "", False

    def _perform_search(self, step: int, cleaned_question: str, clusters: Dict[int, List[str]], context: str, uncertainty: float, threshold: float, is_negative_consensus: bool) -> Tuple[str, Dict[str, Any]]:
        # 0. Consensus Verification Flow (only if step == 0 and low uncertainty)
        if step == 0 and uncertainty <= threshold and not is_negative_consensus:
            logger.info("Step 0 Consensus detected. Initiating Consensus Verification Flow.")
            largest_cluster = max(clusters.values(), key=len)
            context_add, log = self._run_consensus_verification_flow(cleaned_question, largest_cluster)
            if context_add:
                return context_add, {"type": "consensus_verification", "details": log}
            else:
                 logger.info("Consensus verification returned no results. Falling back to Broad flow.")

        # 1. Try Dissenter Flow (only if step > 0)
        if step > 0:
            min_size = min(len(c) for c in clusters.values())
            minority_clusters = [c for c in clusters.values() if len(c) == min_size]
            
            if minority_clusters:
                context_add, log = self._run_dissenter_flow(cleaned_question, minority_clusters[0])
                if context_add:
                    return context_add, {"type": "dissenter", "details": log}
                else:
                    logger.info("Dissenter flow returned no results. Falling back to Broad flow.")

        # 2. Broad Context Flow (if Dissenter/Verification failed or at step 0 with high uncertainty)
        context_add, log = self._run_broad_flow(step, cleaned_question, context, clusters, uncertainty, threshold, is_negative_consensus)
        return context_add, {"type": "broad", "details": log}

    def _run_consensus_verification_flow(self, cleaned_question: str, cluster: List[str]) -> Tuple[str, Any]:
        hypothesis = cluster[0]
        try:
            history_str, failed_str = self._get_history_strings()
            prompt = CONSENSUS_VERIFICATION_PROMPT.format(
                cleaned_question=cleaned_question, claim=hypothesis, history_str=history_str, failed_queries_str=failed_str
            )
            res = self.verification_query_agent.run(prompt)
            query = res.output.query.strip()
            
            if not query or self._is_redundant(query):
                self.history.append((query or "EMPTY", ""))
                return "", None
            
            self.search_call_count += 1
            snippets = self._search(query)
            self.history.append((query, snippets))
            
            if snippets.strip():
                return f"\n### [Consensus Verification Search: '{query}']\n{snippets}\n", {"query": query, "verified": True}
        except Exception as e:
            logger.error(f"Consensus verification error: {e}")
        return "", None

    def _run_dissenter_flow(self, cleaned_question: str, cluster: List[str]) -> Tuple[str, Any]:
        hypothesis = cluster[0]
        # Only one trial
        try:
            history_str, failed_str = self._get_history_strings()
            prompt = VERIFICATION_QUERY_PROMPT.format(
                cleaned_question=cleaned_question, claim=hypothesis, history_str=history_str, failed_queries_str=failed_str
            )
            res = self.verification_query_agent.run(prompt)
            query = res.output.query.strip()
            
            if not query or self._is_redundant(query):
                self.history.append((query or "EMPTY", ""))
                return "", None
            
            self.search_call_count += 1
            snippets = self._search(query)
            self.history.append((query, snippets))
            
            if snippets.strip():
                # Validation gate commented out per request
                # val_res = self.validation_gate_agent.run(VALIDATION_GATE_PROMPT.format(
                #     cleaned_question=cleaned_question, hypothesis=hypothesis, snippets=snippets
                # ))
                # if val_res.output.is_confirmed:
                return f"\n### Search Results for Query: '{query}'\nSnippets:\n{snippets}\n", {"query": query, "confirmed": True}
        except Exception as e:
            logger.error(f"Dissenter error: {e}")
        return "", None

    def _run_broad_flow(self, step: int, cleaned_question: str, context: str, clusters: Dict[int, List[str]], uncertainty: float, threshold: float, is_negative_consensus: bool) -> Tuple[str, Any]:
        for retry in range(5):
            try:
                history_str, failed_str = self._get_history_strings()
                distilled_str = "\n".join([f"- {f}" for f in self.distilled_facts]) or "None"
                
                if step == 0 and uncertainty > threshold and retry == 0:
                    prompt = GAP_ANALYSIS_INITIAL_PROMPT.format(cleaned_question=cleaned_question)
                elif is_negative_consensus or retry > 0:
                    prompt = GAP_ANALYSIS_NEGATIVE_CONSENSUS_PROMPT.format(
                        cleaned_question=cleaned_question,
                        history_str=history_str,
                        failed_queries_str=failed_str,
                        distilled_knowledge=distilled_str
                    )
                else:
                    clusters_str = ""
                    for cid, clist in clusters.items():
                        clusters_str += f"Cluster {cid}: - {clist[0]} ({len(clist)} drafts)\n"
                    
                    prompt = GAP_ANALYSIS_CONFLICT_PROMPT.format(
                        cleaned_question=cleaned_question, clusters_str=clusters_str, context=context,
                        history_str=history_str, failed_queries_str=failed_str, distilled_knowledge=distilled_str
                    )

                res = self.gap_agent.run(prompt)
                query = res.output.query.strip()
                
                if not query or self._is_redundant(query):
                    self.history.append((query or "EMPTY", ""))
                    continue
                
                self.search_call_count += 1
                snippets = self._search(query)
                self.history.append((query, snippets))
                
                if snippets.strip():
                    return f"\n### [Broad Search: '{query}']\n{snippets}\n", {"query": query, "success": True}
            except Exception as e:
                logger.error(f"Broad flow error: {e}")
        return "", None

    def _update_distilled_facts(self, cleaned_question: str, new_snippets: str):
        current = "\n".join([f"- {f}" for f in self.distilled_facts]) or "None"
        prompt = DISTILLATION_PROMPT.format(cleaned_question=cleaned_question, current_knowledge=current, new_snippets=new_snippets)
        try:
            res = self.distillation_agent.run(prompt)
            self.distilled_facts = res.output.facts
        except Exception:
            pass

    def _search(self, query: str) -> str:
        res = self.search_tool.search(query, 5)
        # Format result
        if isinstance(res, list):
            return "\n".join([f"Title: {i.get('title')}\nContent: {i.get('content') or i.get('snippet')}\n---" for i in res if isinstance(i, dict)])
        return str(res)

    def _check_negative_consensus(self, draft: str, question: str) -> bool:
        try:
            return self.negative_consensus_agent.run(NEGATIVE_CONSENSUS_PROMPT.format(cleaned_question=question, draft=draft)).output.is_negative
        except:
            return False

    def _is_redundant(self, query: str) -> bool:
        return any(query.lower() in old.lower() or old.lower() in query.lower() for old, _ in self.history)

    def _select_random_from_largest_cluster(self, clusters: Dict[int, List[str]]) -> str:
        if not clusters: return ""
        largest = max(clusters.values(), key=len)
        return random.choice(largest)

    def _get_history_strings(self) -> Tuple[str, str]:
        succ = [q for q, r in self.history if r.strip()]
        fail = [q for q, r in self.history if not r.strip()]
        return ("\n".join([f"- {q}" for q in succ]) or "None", "\n".join([f"- {q}" for q in fail]) or "None")

    def _pack_message(self, content, role="user"):
        return {"content": content, "role": role}
