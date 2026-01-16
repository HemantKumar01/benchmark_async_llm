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
                # Step 4: long_press example
                """assistant: I need to long press on an element to bring up additional options. I'll use long_press to hold on the icon at index 12.

```python
long_press(12)
```""",
                # Step 5: wait example
                """assistant: I need to wait for the animation to complete before proceeding. I'll wait for 2 seconds.

```python
wait(2.0)
```""",
                # Step 6: system_button with home
                """assistant: I need to go back to the home screen. I'll press the home button.

```python
system_button("home")
```""",
                # Step 7: open_app
                """assistant: I need to open the Chrome browser app.

```python
open_app("Chrome")
```""",
                # Step 8: remember example
                """assistant: I found important information - the battery is at 85%. Let me remember this for later use.

```python
remember("Battery level is at 85%")
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
                # Planning step 2 - after open_app
                """<thought>
Chrome has been opened successfully. Now I need to activate the search bar so we can type the query.
</thought>

<plan>
1. Tap on the address/search bar at the top
2. Type "news" into the search field
3. Submit the search by pressing enter
</plan>""",
                # Planning step 3 - after click
                """<thought>
The search bar is now focused. I need to type the search query "news" into it.
</thought>

<plan>
1. Type "news" in the active search field
2. Press enter to submit the search
3. Wait for results to load
</plan>""",
                # Planning step 4 - after type
                """<thought>
The text has been typed. Now I need to submit the search query by pressing enter.
</thought>

<plan>
1. Press the enter key to submit the search
2. Wait for the search results to load
3. Verify the results are displayed
</plan>""",
                # Planning step 5 - after system_button (enter)
                """<thought>
The search has been submitted and results should be loading. Let me wait a moment for the page to fully load.
</thought>

<plan>
1. Wait for the page to complete loading
2. Scroll down to view more results
3. Check if additional actions are needed
</plan>""",
                # Planning step 6 - after wait
                """<thought>
The page has loaded. Now I should scroll down to see more search results.
</thought>

<plan>
1. Swipe up to scroll down the page
2. Review the additional search results
3. Complete the task if sufficient results are visible
</plan>""",
                # Planning step 7 - after swipe
                """<thought>
Good, I've scrolled through the search results. Let me check if there are any specific items that need interaction or if we need to go back.
</thought>

<plan>
1. Review the visible search results
2. If needed, perform a long press to access additional options
3. Otherwise, navigate back or complete the task
</plan>""",
                # Planning step 8 - after long_press
                """<thought>
I've demonstrated the long press action. Now let me navigate back to show the back button functionality.
</thought>

<plan>
1. Press the back button to return to the previous screen
2. Verify we're back at the search results
</plan>""",
                # Planning step 9 - after back button
                """<thought>
I've successfully demonstrated various actions including opening an app, clicking, typing, submitting, waiting, scrolling, long pressing, and navigating back. Let me try clicking at specific coordinates.
</thought>

<plan>
1. Click at specific screen coordinates to demonstrate click_at
2. Demonstrate click_area for clicking within a bounded region
3. Show long_press_at for coordinate-based long press
</plan>""",
                # Planning step 10 - after click_at
                """<thought>
The click_at action has been executed. Now let me demonstrate clicking within a specific area.
</thought>

<plan>
1. Use click_area to click the center of a region
2. Follow up with long_press_at at specific coordinates
</plan>""",
                # Planning step 11 - after click_area
                """<thought>
The click_area action worked. Let me show the long_press_at functionality.
</thought>

<plan>
1. Perform long_press_at at specific coordinates
2. Complete the demonstration
</plan>""",
                # Finish - after all actions
                """<add_memory>
Successfully demonstrated all available actions: open_app, click, type, system_button (enter and back), swipe, long_press, wait, click_at, click_area, and long_press_at. All actions executed without errors.
</add_memory>
<plan>
</plan>
<request_accomplished success="true">
Successfully completed the comprehensive action demonstration. All atomic actions have been tested and verified.
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
                # Action 7: long_press
                """assistant: ### Thought ###
I need to long press on element at index 8 to bring up a context menu or additional options.

### Action ###
```json
{
  "action": "long_press",
  "index": 8
}
```
### Description ###
Long press on the element to show context menu.""",
                # Action 8: wait
                """assistant: ### Thought ###
I need to wait for the page to load completely before proceeding with the next action.

### Action ###
```json
{
  "action": "wait",
  "duration": 2.0
}
```
### Description ###
Wait for 2 seconds for the page to load.""",
                # Action 9: click_at
                """assistant: ### Thought ###
I need to click at a specific coordinate (400, 600) on the screen where no indexed element is available.

### Action ###
```json
{
  "action": "click_at",
  "x": 400,
  "y": 600
}
```
### Description ###
Click at the specific screen coordinates.""",
                # Action 10: click_area
                """assistant: ### Thought ###
I need to click in the center of a specific rectangular area defined by bounds (100, 200) to (500, 400).

### Action ###
```json
{
  "action": "click_area",
  "x1": 100,
  "y1": 200,
  "x2": 500,
  "y2": 400
}
```
### Description ###
Click the center of the specified area.""",
                # Action 11: long_press_at
                """assistant: ### Thought ###
I need to long press at specific coordinates (300, 800) to trigger a custom action.

### Action ###
```json
{
  "action": "long_press_at",
  "x": 300,
  "y": 800
}
```
### Description ###
Long press at the specified screen coordinates.""",
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
