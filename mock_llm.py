import asyncio
import time
from typing import Literal, List, Dict

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
)
from llama_index.core.llms.llm import LLM


class MockLLM(LLM):
    """
    Mock LLM class that simulates response latency for performance testing.

    Args:
        agent_type: Type of agent - "codeact", "manager", "executor", "app_opener", "text_manipulator"
        wait_time: Fixed time to wait before responding (in seconds)
    """

    # Configure Pydantic to allow extra fields
    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def __init__(
        self,
        agent_type: Literal[
            "codeact", "manager", "executor", "app_opener", "text_manipulator"
        ] = "codeact",
        wait_time: float = 3.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.agent_type = agent_type
        self.wait_time = wait_time
        self.call_counter = 0

        # Multiple responses for each agent type
        self._responses = self._initialize_responses()

    def _initialize_responses(self) -> Dict[str, List[str]]:
        """Initialize response sequences for each agent type."""
        return {
            "codeact": [
                # Step 1: Tap on Settings icon
                """assistant: The user wants to open the Settings app. I can see a "Settings" icon at index 10 on the home screen. I will tap on it.

```python
click(10)
```""",
                # Step 2: Multiple actions - scroll and tap
                """assistant: Now I need to find the battery settings. I'll scroll down to see more options and then tap on the Battery menu item at index 15.

```python
swipe([540, 1500], [540, 500], 1.0)
click(15)
```""",
                # Step 3: Finish
                """assistant: I have successfully opened the Settings app and navigated to the Battery section. The task is complete.

```python
complete(success=True, reason="Successfully accessed battery settings")
```""",
            ],
            "manager": [
                # Planning step 1
                """<thought>
I need to help the user search for news in Chrome. Let me break this down into steps: first open Chrome, then search for news.
</thought>

<plan>
1. Open the Chrome app
2. Tap on the search bar
3. Type "news" in the search bar
4. Press enter to search
</plan>""",
                # Planning step 2
                """<thought>
The Chrome app has been opened. Now I need to interact with the search bar and perform the search.
</thought>

<plan>
1. Tap on the address/search bar at the top
2. Type "news" 
3. Submit the search
</plan>""",
                # Finish - using request_accomplished format
                """<add_memory>
The Chrome app was opened successfully and the search for "news" was completed. The search results are now displayed in the browser.
</add_memory>
<plan>
</plan>
<request_accomplished success="true">
Successfully searched for news in Chrome browser. The search results are now visible.
</request_accomplished>""",
            ],
            "executor": [
                # Action 1: open_app
                """assistant: ### Thought ###
The user wants me to execute the subgoal "Open the Chrome app". This directly translates to the `open_app` atomic action. The parameter for this action is the name of the app, which is "Chrome".

### Action ###
```json
{
  "action": "open_app",
  "text": "Chrome"
}
```
### Description ###
Open the Chrome application.""",
                # Action 2: click
                """assistant: ### Thought ###
I need to tap on the search bar to activate it. The search bar is at index 5 based on the current UI state.

### Action ###
```json
{
  "action": "click",
  "index": 5
}
```
### Description ###
Tap on the search bar to enable text input.""",
                # Action 3: type
                """assistant: ### Thought ###
Now I need to type "news" into the search bar. The focused element is the search input field.

### Action ###
```json
{
  "action": "type",
  "text": "news",
  "index": 5,
  "clear": false
}
```
### Description ###
Type "news" into the search bar.""",
                # Action 4: system_button (enter)
                """assistant: ### Thought ###
To submit the search, I need to press the enter key. I'll use the system_button action with "enter".

### Action ###
```json
{
  "action": "system_button",
  "button": "enter"
}
```
### Description ###
Press enter to submit the search query.""",
                # Action 5: swipe
                """assistant: ### Thought ###
I need to scroll down to see more results. I'll perform a swipe gesture from bottom to top.

### Action ###
```json
{
  "action": "swipe",
  "coordinate": [540, 1500],
  "coordinate2": [540, 500],
  "duration": 1.0
}
```
### Description ###
Scroll down to view more search results.""",
                # Action 6: system_button (back)
                """assistant: ### Thought ###
The user wants to go back to the previous screen. I'll use the system_button action with "back".

### Action ###
```json
{
  "action": "system_button",
  "button": "back"
}
```
### Description ###
Navigate back to the previous screen.""",
            ],
            "app_opener": [
                # Single response for app opener - returns JSON with package name
                """{{
  "package": "com.android.chrome"
}}""",
            ],
            "text_manipulator": [
                # Single response for text manipulation
                """assistant: I'll input the text as requested.

```python
input_text("news", clear=True)
```""",
            ],
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
            "call_counter": self.call_counter,
        }

    def _get_current_response(self) -> str:
        """Get the current response based on the call counter."""
        responses = self._responses[self.agent_type]
        # Cycle through responses, stay on last one if counter exceeds length
        index = min(self.call_counter, len(responses) - 1)
        return responses[index]

    def _complete(self, prompt: str, **kwargs) -> str:
        """Synchronous completion (not used in async workflows)."""
        time.sleep(self.wait_time)
        response = self._get_current_response()
        self.call_counter += 1
        return response

    async def _acomplete(self, prompt: str, **kwargs) -> str:
        """Async completion (not used in chat workflows)."""
        await asyncio.sleep(self.wait_time)
        response = self._get_current_response()
        self.call_counter += 1
        return response

    def _chat(self, messages: list, **kwargs) -> ChatResponse:
        """Synchronous chat (not used in async workflows)."""
        time.sleep(self.wait_time)
        content = self._get_current_response()
        self.call_counter += 1
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
        content = self._get_current_response()
        self.call_counter += 1

        return ChatResponse(
            message=ChatMessage(role="assistant", content=content),
            raw={
                "mock": True,
                "agent_type": self.agent_type,
                "call_number": self.call_counter,
            },
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
        response = self._get_current_response()
        self.call_counter += 1
        yield response

    async def _astream_complete_gen(self, prompt: str, **kwargs):
        """Internal async generator for stream completion."""
        response = self._get_current_response()
        self.call_counter += 1
        yield response

    async def astream_complete(self, prompt: str, **kwargs):
        """Async stream completion - returns async generator wrapped in coroutine."""

        async def gen():
            await asyncio.sleep(self.wait_time)
            content = self._get_current_response()
            self.call_counter += 1

            # For streaming, we need to set delta (the chunk content)
            yield CompletionResponse(
                text=content,
                delta=content,  # This is what the streaming code reads
                raw={
                    "mock": True,
                    "agent_type": self.agent_type,
                    "call_number": self.call_counter,
                },
            )

        return gen()

    def stream_chat(self, messages: list, **kwargs):
        """Stream chat - not implemented for mock."""
        response = self._get_current_response()
        self.call_counter += 1
        yield self._chat(messages, **kwargs)

    async def _astream_chat_gen(self, messages: list, **kwargs):
        """Internal async generator for stream chat."""
        response = self._get_current_response()
        self.call_counter += 1
        yield await self._achat(messages, **kwargs)

    async def astream_chat(self, messages: list, **kwargs):
        """Async stream chat - returns async generator wrapped in coroutine."""

        async def gen():
            await asyncio.sleep(self.wait_time)
            content = self._get_current_response()
            self.call_counter += 1

            # For streaming, we need to set delta (the chunk content)
            yield ChatResponse(
                message=ChatMessage(role="assistant", content=content),
                delta=content,  # This is what the streaming code reads
                raw={
                    "mock": True,
                    "agent_type": self.agent_type,
                    "call_number": self.call_counter,
                },
            )

        return gen()

    # Backward compatibility methods
    async def generate(self, prompt: str = "") -> str:
        """
        Simulate LLM generation with fixed latency.

        Args:
            prompt: Input prompt (unused in mock, but included for interface compatibility)

        Returns:
            Hard-coded response based on agent type and call counter
        """
        await asyncio.sleep(self.wait_time)
        response = self._get_current_response()
        self.call_counter += 1
        return response

    def __call__(self, prompt: str = "") -> str:
        """
        Synchronous interface for LLM generation.

        Args:
            prompt: Input prompt (unused in mock)

        Returns:
            Hard-coded response based on agent type and call counter
        """
        time.sleep(self.wait_time)
        response = self._get_current_response()
        self.call_counter += 1
        return response

    def reset_counter(self) -> None:
        """Reset the call counter to 0."""
        self.call_counter = 0
