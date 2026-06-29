"""
Base Hermes-style agent class.

Hermes Agent framework (https://github.com/nousresearch/hermes-agent) follows
a ReAct / function-calling pattern.  This base class mimics that interface so
each specialised agent only needs to implement `tools` and `system_prompt`.
"""

from abc import ABC, abstractmethod
from utils.llm_client import chat_completion
from utils.logger import logger


class BaseAgent(ABC):
    """
    Abstract base for all CrowdWisdom agents.

    Subclasses must define:
        name          – human-readable agent name
        system_prompt – str fed to the LLM as the system message
        tools         – list of tool definitions (OpenAI function-calling format)
        run(**kwargs) – entry-point that returns a structured result dict
    """

    name: str = "BaseAgent"

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        ...

    @property
    def tools(self) -> list[dict]:
        return []

    def _llm(self, user_message: str, extra_history: list[dict] | None = None) -> str:
        """Send a message to the LLM with this agent's system prompt."""
        messages = [{"role": "system", "content": self.system_prompt}]
        if extra_history:
            messages.extend(extra_history)
        messages.append({"role": "user", "content": user_message})

        logger.info(f"[{self.name}] Calling LLM...")
        response = chat_completion(messages)
        logger.info(f"[{self.name}] LLM responded.")
        return response

    @abstractmethod
    def run(self, **kwargs) -> dict:
        """Execute the agent's main task and return a result dict."""
        ...
