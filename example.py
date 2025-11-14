import asyncio

from mock_llm import MockLLM

from droidrun.agent.droid import DroidAgent
from droidrun.config_manager.config_manager import AgentConfig, DroidrunConfig


async def benchmark():
    config = DroidrunConfig(agent=AgentConfig(reasoning=True, max_steps=2))
    # Single LLM (non-reasoning mode)
    mock_llm = MockLLM(agent_type="codeact", wait_time=3.0)
    agent = DroidAgent(goal="Open Settings", config=config, llms=mock_llm)
    result = await agent.run()

    # Multiple LLMs (reasoning mode)
    llms = {
        "manager": MockLLM(agent_type="manager", wait_time=3.0),
        "executor": MockLLM(agent_type="executor", wait_time=3.0),
        "codeact": MockLLM(agent_type="codeact", wait_time=3.0),
        "text_manipulator": MockLLM(agent_type="codeact", wait_time=3.0),
        "app_opener": MockLLM(agent_type="codeact", wait_time=3.0),
        "scripter": MockLLM(agent_type="codeact", wait_time=3.0),
    }
    agent = DroidAgent(goal="Search in Chrome", config=config, llms=llms)
    result = await agent.run()


if __name__ == "__main__":
    asyncio.run(benchmark())
