"""
Quick demo script for testing the benchmark system with faster execution.
Uses shorter wait times and fewer iterations for rapid testing.
"""

import asyncio
from benchmark_llm import BenchmarkRunner


async def main():
    print("\n" + "=" * 70)
    print("Running Benchmark")
    print("=" * 70)

    # Create runner with faster settings
    runner = BenchmarkRunner(
        wait_time=3.0, num_iter=2  # 1 second instead of 3  # 2 iterations instead of 5
    )

    # Run a subset of benchmarks
    configs = [
        (1, False, "1 agent (Non-Reasoning)"),
        (10, False, "10 agents (Non-Reasoning)"),
        (100, False, "100 agents (Non-Reasoning)"),
        (1, True, "1 agent (Reasoning)"),
        (10, True, "10 agents (Reasoning)"),
        (100, True, "100 agents (Reasoning)"),
    ]

    for num_agents, reasoning, description in configs:
        await runner.run_benchmark(num_agents, reasoning, description)
        await asyncio.sleep(0.5)  # Small delay between benchmarks

    # Print comparison
    runner.print_comparison_table()

    # Save results
    runner.save_results()

    # Visualize
    runner.visualize_results()

    print("\n✓ Demo complete!")


if __name__ == "__main__":
    asyncio.run(main())
