import asyncio
import time
from typing import Literal

from llama_index.core.base.llms.types import ChatMessage, ChatResponse
from llama_index.core.llms.llm import LLM


class MockLLM(LLM):
    """
    Mock LLM class that simulates response latency for performance testing.

    Args:
        agent_type: Type of agent - "codeact", "manager", or "executor"
        wait_time: Fixed time to wait before responding (in seconds)
    """

    # Configure Pydantic to allow extra fields
    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def __init__(
        self,
        agent_type: Literal["codeact", "manager", "executor"] = "codeact",
        wait_time: float = 3.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.agent_type = agent_type
        self.wait_time = wait_time
        self._responses = {
            "codeact": """assistant: The user wants to open the Settings app and check the battery level. The current screen is the Pixel Launcher, so the first step is to open the Settings app. I can see a "Settings" icon at index 11, which I will click.

```python
click(11)
```""",
            "manager": """<thought>
I need to help the user search for news in Chrome. Let me break this down into steps: first open Chrome, then search for news.
</thought>

<plan>
1. Open the Chrome app
2. Type "news" in the search bar
3. Press enter to search
</plan>""",
            "executor": """assistant: ### Thought ###
The user wants me to execute the subgoal "Open the Chrome app.". This directly translates to the `open_app` atomic action. The parameter for this action is the name of the app, which is "Chrome".

### Action ###
```json
{
  "action": "open_app",
  "text": "Chrome"
}
```
### Description ###
Open the Chrome application.""",
        }

    @classmethod
    def class_name(cls) -> str:
        """Get class name for llama_index compatibility."""
        return "MockLLM"

    @property
    def metadata(self) -> dict:
        """Get metadata for llama_index compatibility."""
        return {
            "agent_type": self.agent_type,
            "wait_time": self.wait_time,
            "model_name": "mock",
        }

    def _complete(self, prompt: str, **kwargs) -> str:
        """Synchronous completion (not used in async workflows)."""
        time.sleep(self.wait_time)
        return self._responses[self.agent_type]

    async def _acomplete(self, prompt: str, **kwargs) -> str:
        """Async completion (not used in chat workflows)."""
        await asyncio.sleep(self.wait_time)
        return self._responses[self.agent_type]

    def _chat(self, messages: list, **kwargs) -> ChatResponse:
        """Synchronous chat (not used in async workflows)."""
        time.sleep(self.wait_time)
        content = self._responses[self.agent_type]
        return ChatResponse(
            message=ChatMessage(role="assistant", content=content),
            raw={"mock": True},
        )

    async def _achat(self, messages: list, **kwargs) -> ChatResponse:
        """
        Async chat implementation - this is what DroidAgent uses.

        Args:
            messages: List of ChatMessage objects

        Returns:
            ChatResponse with mocked content
        """
        await asyncio.sleep(self.wait_time)
        content = self._responses[self.agent_type]
        return ChatResponse(
            message=ChatMessage(role="assistant", content=content),
            raw={"mock": True, "agent_type": self.agent_type},
        )

    # Public methods required by LLM base class
    def complete(self, prompt: str, **kwargs) -> str:
        """Synchronous completion - calls _complete."""
        return self._complete(prompt, **kwargs)

    async def acomplete(self, prompt: str, **kwargs) -> str:
        """Async completion - calls _acomplete."""
        return await self._acomplete(prompt, **kwargs)

    def chat(self, messages: list, **kwargs) -> ChatResponse:
        """Synchronous chat - calls _chat."""
        return self._chat(messages, **kwargs)

    async def achat(self, messages: list, **kwargs) -> ChatResponse:
        """Async chat - calls _achat."""
        return await self._achat(messages, **kwargs)

    def stream_complete(self, prompt: str, **kwargs):
        """Stream completion - not implemented for mock."""
        yield self._complete(prompt, **kwargs)

    async def astream_complete(self, prompt: str, **kwargs):
        """Async stream completion - not implemented for mock."""
        yield await self._acomplete(prompt, **kwargs)

    def stream_chat(self, messages: list, **kwargs):
        """Stream chat - not implemented for mock."""
        yield self._chat(messages, **kwargs)

    async def astream_chat(self, messages: list, **kwargs):
        """Async stream chat - not implemented for mock."""
        yield await self._achat(messages, **kwargs)

    # Backward compatibility methods
    async def generate(self, prompt: str = "") -> str:
        """
        Simulate LLM generation with fixed latency.

        Args:
            prompt: Input prompt (unused in mock, but included for interface compatibility)

        Returns:
            Hard-coded response based on agent type
        """
        await asyncio.sleep(self.wait_time)
        return self._responses[self.agent_type]

    def __call__(self, prompt: str = "") -> str:
        """
        Synchronous interface for LLM generation.

        Args:
            prompt: Input prompt (unused in mock)

        Returns:
            Hard-coded response based on agent type
        """
        time.sleep(self.wait_time)
        return self._responses[self.agent_type]
