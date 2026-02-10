import logging
import time

import httpx
import ollama
from ollama import ChatResponse

logger = logging.getLogger(__name__)

# Default generation parameters per model family.
# Keys are matched as prefixes against the model_name.
MODEL_DEFAULTS: dict[str, dict] = {
    "deepseek-r1": {
        "temperature": 0.6,
        "top_p": 0.95,
        "num_predict": 8192,
    },
    "qwq": {
        "temperature": 0.6,
        "top_p": 0.8,
        "top_k": 20,
        "num_predict": 32768,
    },
    "gpt-oss": {
        "temperature": 0.7,
        "top_p": 0.9,
        "num_predict": 4096,
    },
}

FALLBACK_DEFAULTS: dict = {
    "temperature": 0.7,
    "top_p": 0.9,
    "num_predict": 4096,
}


def _get_model_defaults(model_name: str) -> dict:
    """Return generation defaults for a model, matched by longest prefix."""
    for prefix in sorted(MODEL_DEFAULTS, key=len, reverse=True):
        if model_name.startswith(prefix):
            return MODEL_DEFAULTS[prefix].copy()
    return FALLBACK_DEFAULTS.copy()


def _is_retriable(e: Exception) -> bool:
    return isinstance(e, (httpx.ConnectError, httpx.RemoteProtocolError, ConnectionError))


def _format_messages_for_log(messages: list[dict]) -> str:
    lines = []
    for msg in messages:
        role = msg.get("role", "?").upper()
        content = msg.get("content", "")
        thinking = msg.get("thinking", "")
        if thinking:
            lines.append(f"  [{role}] <thinking> {thinking[:200]}{'...' if len(thinking) > 200 else ''}")
        if content:
            lines.append(f"  [{role}] {content[:300]}{'...' if len(content) > 300 else ''}")
        if not thinking and not content:
            lines.append(f"  [{role}] (empty)")
    return "\n".join(lines)


class OllamaThinkingAgent:
    """
    An agent service using the native ollama-python library that supports
    injecting modified thinking sections into conversations and prompting
    the model to continue answering based on them.

    Unlike BaseAgent (which wraps ollama via pydantic-ai's OpenAI compatibility layer),
    this uses ollama's native Message.thinking field for first-class thinking support.
    """

    def __init__(self, prompt_path: str = None, model_name: str = "deepseek-r1",
                 agent_name: str = None, system_prompt: str = None,
                 base_url: str = "http://localhost:11434",
                 **generation_overrides):
        """
        Args:
            prompt_path: Path to a text file containing the system prompt.
            model_name: Ollama model identifier (e.g. "deepseek-r1", "qwq", "gpt-oss:20b").
            agent_name: Human-readable name for logging.
            system_prompt: Inline system prompt (ignored if prompt_path is set).
            base_url: Ollama server URL.
            **generation_overrides: Override any default generation parameter
                (temperature, top_p, top_k, num_predict, etc.).
        """
        self.model_name = model_name
        self.agent_name = agent_name or f"ollama_{model_name}"
        self.client = ollama.Client(host=base_url)

        # Resolve generation options: model defaults ← overrides
        self.options = _get_model_defaults(model_name)
        self.options.update({k: v for k, v in generation_overrides.items() if v is not None})

        if prompt_path:
            self.system_prompt = self._load_prompt(prompt_path)
        elif system_prompt:
            self.system_prompt = system_prompt
        else:
            self.system_prompt = ""

        logger.info("Initialized %s  model=%s  options=%s", self.agent_name, model_name, self.options)

    @staticmethod
    def _load_prompt(path: str) -> str:
        with open(path, 'r') as f:
            return f.read()

    def _build_messages(self, user_input: str, message_history: list = None) -> list[dict]:
        messages = []
        if self.system_prompt:
            messages.append({'role': 'system', 'content': self.system_prompt})
        if message_history:
            messages.extend(message_history)
        messages.append({'role': 'user', 'content': user_input})
        return messages

    def run(self, user_input: str, think: bool = True,
            message_history: list = None, max_retries: int = 5) -> ChatResponse:
        """Run the agent with retry logic, optionally with thinking enabled."""
        messages = self._build_messages(user_input, message_history)

        logger.debug("run()  think=%s  messages:\n%s", think, _format_messages_for_log(messages))

        for attempt in range(max_retries):
            try:
                response = self.client.chat(
                    model=self.model_name,
                    messages=messages,
                    think=think,
                    options=self.options,
                )
                logger.info("run() response  thinking=%s  content=%s",
                            bool(response.message.thinking),
                            response.message.content[:200] if response.message.content else "(empty)")
                logger.debug("run() full response thinking:\n%s", response.message.thinking or "(none)")
                return response
            except Exception as e:
                if attempt < max_retries - 1 and _is_retriable(e):
                    wait_time = 2 ** attempt
                    logger.warning("Network error on attempt %d/%d: %s — retrying in %ds",
                                   attempt + 1, max_retries, e, wait_time)
                    time.sleep(wait_time)
                else:
                    raise

    def run_with_injected_thinking(self, user_input: str, injected_thinking: str,
                                   continuation_prompt: str = "Please continue based on the thought process above.",
                                   max_retries: int = 5) -> dict:
        """
        Inject a modified thinking section and prompt the model to continue answering.

        Builds a conversation:
          1. [system prompt]
          2. User query
          3. Assistant message with the injected thinking (via Message.thinking field)
          4. Continuation user prompt

        Then calls the model with think=False so it relies on the injected thought
        rather than starting a new thinking process.

        Returns a dict with:
          - injected_thinking: the thinking that was injected
          - response: the ChatResponse from ollama
          - messages: the full message list sent to the model
        """
        messages = []
        if self.system_prompt:
            messages.append({'role': 'system', 'content': self.system_prompt})

        messages.append({'role': 'user', 'content': user_input})

        # Inject thinking as a native assistant message with the thinking field
        messages.append({
            'role': 'assistant',
            'content': '',
            'thinking': injected_thinking,
        })

        messages.append({'role': 'user', 'content': continuation_prompt})

        logger.debug("run_with_injected_thinking()  messages:\n%s", _format_messages_for_log(messages))

        for attempt in range(max_retries):
            try:
                response = self.client.chat(
                    model=self.model_name,
                    messages=messages,
                    think=False,
                    options=self.options,
                )
                logger.info("injected_thinking response  content=%s",
                            response.message.content[:200] if response.message.content else "(empty)")
                return {
                    'injected_thinking': injected_thinking,
                    'response': response,
                    'messages': messages,
                }
            except Exception as e:
                if attempt < max_retries - 1 and _is_retriable(e):
                    wait_time = 2 ** attempt
                    logger.warning("Network error on attempt %d/%d: %s — retrying in %ds",
                                   attempt + 1, max_retries, e, wait_time)
                    time.sleep(wait_time)
                else:
                    raise


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    agent = OllamaThinkingAgent(
        model_name="deepseek-r1",
        system_prompt="You are a helpful trivia assistant. Answer concisely.",
    )

    # --- 1. Normal run with thinking ---
    print("=" * 60)
    print("TEST 1: Normal run (think=True)")
    print("=" * 60)
    query = "What was the original name of the insurance company also known as Nationwide Mutual Group?"
    response = agent.run(query, think=True)
    print(f"Thinking: {response.message.thinking}")
    print(f"Answer:   {response.message.content}")

    # --- 2. Run with injected thinking ---
    print("\n" + "=" * 60)
    print("TEST 2: Run with injected thinking")
    print("=" * 60)
    injected = (
        "The user is asking about the original name of Nationwide Mutual Group. "
        "I recall that Nationwide was originally founded as the Farm Bureau Mutual "
        "Automobile Insurance Company in 1926. I'm fairly confident about this."
    )
    result = agent.run_with_injected_thinking(
        user_input=query,
        injected_thinking=injected,
    )
    print(f"Injected thinking: {result['injected_thinking']}")
    print(f"Answer:            {result['response'].message.content}")

    # --- 3. Normal run without thinking ---
    print("\n" + "=" * 60)
    print("TEST 3: Normal run (think=False)")
    print("=" * 60)
    response = agent.run(query, think=False)
    print(f"Thinking: {response.message.thinking}")
    print(f"Answer:   {response.message.content}")
