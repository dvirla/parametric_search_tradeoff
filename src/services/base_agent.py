import asyncio
import os
import time
from dotenv import load_dotenv
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.agent import Agent
from pydantic_ai.usage import UsageLimits
from pydantic_ai import capture_run_messages
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import ModelResponse, TextPart
import openai
import logfire
import httpx

load_dotenv()

# Per-question agent-loop budget. Kept at the original 100: legitimate multi-hop FRAMES questions
# genuinely use a wide range of search calls (observed: median 3, p99 ~50, max ~92 on completed
# runs), and search_calls is itself a measured outcome — so lowering this would truncate real
# questions and corrupt the metric. The change here is NOT the cap, it's what happens when the cap
# (or a timeout) is hit: salvage the best-effort answer instead of crashing the whole eval batch.
DEFAULT_TOOL_CALLS_LIMIT = 100
DEFAULT_REQUEST_LIMIT = 100


class _PartialRunResult:
    """Lightweight stand-in for a pydantic-ai run result, used to salvage a best-effort answer when
    a run hits the usage limit or times out instead of discarding all work on that question. Exposes
    just what AgentAsSampler/qa_eval read: ``all_messages()`` and ``output`` (plus ``stop_reason``)."""

    def __init__(self, messages, output, stop_reason):
        self._messages = list(messages)
        self.output = output
        self.stop_reason = stop_reason

    def all_messages(self):
        return self._messages


def _salvage_partial(messages, exc):
    """Build a _PartialRunResult from whatever messages were captured before ``exc`` was raised,
    using the last non-empty assistant text as the best-effort answer."""
    output = ""
    for msg in reversed(messages):
        if isinstance(msg, ModelResponse):
            txt = "\n".join(p.content for p in msg.parts if isinstance(p, TextPart) and p.content)
            if txt:
                output = txt
                break
    return _PartialRunResult(messages, output, type(exc).__name__)

# Graceful handling of Logfire
logfire_key = os.getenv("LOGFIRE_API_KEY")
if logfire_key:
    logfire.configure(send_to_logfire=True, token=logfire_key)
    logfire.instrument_pydantic_ai()
else:
    # If no key, we can try to configure it to be no-op or just skip
    # logfire.configure(send_to_logfire=False) # This might still trigger auth check if defaults are strict
    pass

class BaseAgent:
    def __init__(self, prompt_path: str = None, provider_name: str = "Google", model_name: str = "gemini-flash-latest",
                 output_type = str, tools: list = [], agent_name: str = None, use_thinking: bool = True,
                 temperature: float = 1, top_p: float = None, top_k: int = None,
                 system_prompt: str = None, timeout: float = None, ollama_base_url: str = None,
                 tool_calls_limit: int = DEFAULT_TOOL_CALLS_LIMIT, request_limit: int = DEFAULT_REQUEST_LIMIT):
        self.provider_name = provider_name
        self.model_name = model_name
        self.tool_calls_limit = tool_calls_limit
        self.request_limit = request_limit

        self.agent_name = agent_name or f"{provider_name}_{model_name}"

        timeout_kwargs = {"timeout": timeout} if timeout is not None else {}
        settings = None
        if provider_name == "Google":
            settings = GoogleModelSettings(google_thinking_config={'include_thoughts': use_thinking}, temperature=temperature, **timeout_kwargs)
            provider = GoogleProvider(api_key=os.getenv("GOOGLE_API_KEY"))
            self.model = GoogleModel(model_name, provider=provider)
        elif provider_name == "OpenAI":
            provider = OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = OpenAIChatModel(model_name=model_name, provider=provider)
        elif provider_name == "Anthropic":
            settings = AnthropicModelSettings(anthropic_thinking={'type': 'enabled' if use_thinking else 'disabled',
                                                                  'budget_tokens': 16000 if use_thinking else 0},
                                                                  max_tokens=20000, **timeout_kwargs)
            provider = AnthropicProvider(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self.model = AnthropicModel(model_name=model_name, provider=provider, settings=settings)
        elif provider_name == "ollama":
            extra = {}
            if top_p is not None:
                extra["top_p"] = top_p
            if top_k is not None:
                extra["extra_body"] = {"top_k": top_k}
            settings = OpenAIChatModelSettings(temperature=temperature, **extra, **timeout_kwargs)
            _ollama_url = ollama_base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1/")
            provider = OllamaProvider(base_url=_ollama_url)
            self.model = OpenAIChatModel(
                model_name=model_name,
                provider=provider
            )
        else:
            raise ValueError(f"Invalid provider: {provider_name}")

        if prompt_path:
            self.system_prompt = self._load_prompt(prompt_path)
        elif system_prompt:
            self.system_prompt = system_prompt
        else:
            self.system_prompt = ""

        self.agent = Agent(
            model=self.model,
            tools=tools,
            system_prompt=self.system_prompt,
            model_settings=settings,
            output_type=output_type,
            retries=3,
            name=self.agent_name
        )

    @staticmethod
    def _load_prompt(path: str) -> str:
        with open(path, 'r') as f:
            return f.read()

    def run(self, user_input: str, max_retries: int = 5):
        """Run the agent with retry logic for network failures.

        A usage-limit overrun or request timeout does NOT raise: instead the last best-effort
        assistant text is salvaged and returned, so one runaway/stalled question can't discard all
        the work done on it (and, upstream, can't abort the rest of the eval batch)."""
        for attempt in range(max_retries):
            try:
                custom_limits = UsageLimits(tool_calls_limit=self.tool_calls_limit, request_limit=self.request_limit)
                with capture_run_messages() as messages:
                    try:
                        return self.agent.run_sync(user_input, usage_limits=custom_limits)
                    except (UsageLimitExceeded, openai.APITimeoutError) as e:
                        print(f"[salvage] {type(e).__name__}: returning best-effort partial answer")
                        return _salvage_partial(messages, e)
            except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.TimeoutException, ConnectionError) as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4, 8, 16 seconds
                    print(f"Network error on attempt {attempt + 1}/{max_retries}: {e}")
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"Failed after {max_retries} attempts")
                    raise

    async def arun(self, user_input: str, max_retries: int = 5):
        """Async version of run() for use in async evaluation contexts."""
        for attempt in range(max_retries):
            try:
                custom_limits = UsageLimits(tool_calls_limit=self.tool_calls_limit, request_limit=self.request_limit)
                with capture_run_messages() as messages:
                    try:
                        return await self.agent.run(user_input, usage_limits=custom_limits)
                    except (UsageLimitExceeded, openai.APITimeoutError) as e:
                        print(f"[salvage] {type(e).__name__}: returning best-effort partial answer")
                        return _salvage_partial(messages, e)
            except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.TimeoutException, ConnectionError) as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Network error on attempt {attempt + 1}/{max_retries}: {e}")
                    print(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"Failed after {max_retries} attempts")
                    raise

def test_agent():
    agent = BaseAgent(
        provider_name="Google",
        model_name="gemini-flash-latest",
    )
    user_query = "What is the 'Hydra' protocol in the context of decentralized AI compute?"
    result = agent.run(user_query)
    print(result.all_messages())
