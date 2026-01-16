import asyncio

from mock_llm import MockLLM

from droidrun.agent.droid import DroidAgent
from droidrun.config_manager.config_manager import AgentConfig, DroidrunConfig


async def benchmark():
    # Single LLM (non-reasoning mode) - codeact with multiple steps
    print("=== Testing Non-Reasoning Mode (CodeAct) ===")
    config_no_reasoning = DroidrunConfig(
        agent=AgentConfig(reasoning=False, max_steps=50)
    )
    mock_llm = MockLLM(agent_type="codeact", wait_time=3.0)
    print(f"Initial call counter: {mock_llm.call_counter}")

    agent = DroidAgent(goal="Open Settings", config=config_no_reasoning, llms=mock_llm)
    result = await agent.run()

    print(f"Final call counter: {mock_llm.call_counter}")

    print("=" * 100)

    # Multiple LLMs (reasoning mode)
    print("=== Testing Reasoning Mode (Manager + Executor) ===")
    config_reasoning = DroidrunConfig(agent=AgentConfig(reasoning=True, max_steps=50))
    llms = {
        "manager": MockLLM(agent_type="manager", wait_time=2.0),
        "executor": MockLLM(agent_type="executor", wait_time=2.0),
        "codeact": MockLLM(agent_type="codeact", wait_time=2.0),
        "text_manipulator": MockLLM(agent_type="text_manipulator", wait_time=1.0),
        "app_opener": MockLLM(agent_type="app_opener", wait_time=1.0),
        "scripter": MockLLM(agent_type="codeact", wait_time=2.0),
    }

    print(f"Manager initial counter: {llms['manager'].call_counter}")
    print(f"Executor initial counter: {llms['executor'].call_counter}")

    agent = DroidAgent(goal="Search in Chrome", config=config_reasoning, llms=llms)
    result = await agent.run()

    print(f"Manager final counter: {llms['manager'].call_counter}")
    print(f"Executor final counter: {llms['executor'].call_counter}")

    print("=" * 100)

    # Test individual LLM responses
    print("=== Testing Individual Response Sequences ===")
    test_llm = MockLLM(agent_type="executor", wait_time=0.1)

    for i in range(7):
        response = await test_llm.acomplete("test prompt")
        print(f"\nCall {i+1} (counter={test_llm.call_counter}):")
        print(response[:100] + "..." if len(response) > 100 else response)


if __name__ == "__main__":
    asyncio.run(benchmark())
