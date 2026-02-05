import os
import time
from dotenv import load_dotenv
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.agent import Agent
import logfire
import httpx

load_dotenv()

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
                 temperature: float = 1, system_prompt: str = None):
        self.provider_name = provider_name
        self.model_name = model_name

        self.agent_name = agent_name or f"{provider_name}_{model_name}"

        settings = None
        if provider_name == "Google":
            settings = GoogleModelSettings(google_thinking_config={'include_thoughts': use_thinking}, temperature=temperature)
            provider = GoogleProvider(api_key=os.getenv("GOOGLE_API_KEY"))
            self.model = GoogleModel(model_name, provider=provider)
        elif provider_name == "OpenAI":
            provider = OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = OpenAIChatModel(model_name=model_name, provider=provider)
        elif provider_name == "Anthropic":
            settings = AnthropicModelSettings(anthropic_thinking={'type': 'enabled' if use_thinking else 'disabled',
                                                                  'budget_tokens': 16000 if use_thinking else 0},
                                                                  max_tokens=20000)
            provider = AnthropicProvider(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self.model = AnthropicModel(model_name=model_name, provider=provider, settings=settings)            
        elif provider_name == "ollama":
            provider = OllamaProvider(base_url='http://localhost:11434/v1/')
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
        """Run the agent with retry logic for network failures."""
        for attempt in range(max_retries):
            try:
                response = self.agent.run_sync(user_input)
                return response
            except (httpx.ConnectError, httpx.RemoteProtocolError, ConnectionError) as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4, 8, 16 seconds
                    print(f"Network error on attempt {attempt + 1}/{max_retries}: {e}")
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
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