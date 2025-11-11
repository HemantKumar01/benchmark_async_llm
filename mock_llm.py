import asyncio
import time
from typing import Literal


class MockLLM:
    """
    Mock LLM class that simulates response latency for performance testing.

    Args:
        agent_type: Type of agent - "codeact", "manager", or "executor"
        wait_time: Fixed time to wait before responding (in seconds)
    """

    def __init__(
        self,
        agent_type: Literal["codeact", "manager", "executor"],
        wait_time: float = 3.0,
    ):
        self.agent_type = agent_type
        self.wait_time = wait_time
        self._responses = {
            "codeact": """assistant: The user wants to open the Settings app and check the battery level. The current screen is the Pixel Launcher, so the first step is to open the Settings app. I can see a "Settings" icon at index 11, which I will click.

```python
click(11)
```""",
            "manager": """assistant: <plan>

Open the Chrome app.
Type "news" in the search bar.
Press the enter button on the keyboard to search.
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


class MockAgent:
    """
    Mock agent that simulates LLM calls for performance testing.

    Args:
        wait_time: Fixed time to wait for each LLM call (in seconds)
        reasoning: If True, alternates between manager and executor agents.
                  If False, uses codeact agent.
        num_iter: Number of iterations (LLM calls) to perform
    """

    def __init__(
        self, wait_time: float = 3.0, reasoning: bool = False, num_iter: int = 1
    ):
        self.wait_time = wait_time
        self.reasoning = reasoning
        self.num_iter = num_iter

        if reasoning:
            # Create manager and executor LLMs for reasoning mode
            self.manager_llm = MockLLM(agent_type="manager", wait_time=wait_time)
            self.executor_llm = MockLLM(agent_type="executor", wait_time=wait_time)
        else:
            # Create codeact LLM for non-reasoning mode
            self.codeact_llm = MockLLM(agent_type="codeact", wait_time=wait_time)

    async def run(self, device_id: str = "device_0") -> dict:
        """
        Execute the agent workflow asynchronously.

        Args:
            device_id: Identifier for the device (used for logging/tracking)

        Returns:
            Dictionary containing execution results and metadata
        """
        start_time = time.time()
        responses = []

        if self.reasoning:
            # Reasoning mode: alternate between manager and executor
            for i in range(self.num_iter):
                # Manager call
                manager_response = await self.manager_llm.generate(
                    f"Step {i+1} - Planning"
                )
                responses.append(
                    {
                        "type": "manager",
                        "iteration": i + 1,
                        "response": manager_response,
                    }
                )

                # Executor call
                executor_response = await self.executor_llm.generate(
                    f"Step {i+1} - Execution"
                )
                responses.append(
                    {
                        "type": "executor",
                        "iteration": i + 1,
                        "response": executor_response,
                    }
                )
        else:
            # Non-reasoning mode: use codeact agent
            for i in range(self.num_iter):
                codeact_response = await self.codeact_llm.generate(f"Step {i+1}")
                responses.append(
                    {
                        "type": "codeact",
                        "iteration": i + 1,
                        "response": codeact_response,
                    }
                )

        end_time = time.time()
        execution_time = end_time - start_time

        return {
            "device_id": device_id,
            "reasoning_mode": self.reasoning,
            "num_iterations": self.num_iter,
            "total_llm_calls": len(responses),
            "execution_time": execution_time,
            "responses": responses,
        }
