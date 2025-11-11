import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
from mock_llm import MockAgent


class BenchmarkRunner:
    """
    Runs benchmark tests for mock LLM agents with various configurations.
    """

    def __init__(self, wait_time: float = 3.0, num_iter: int = 5):
        """
        Initialize the benchmark runner.

        Args:
            wait_time: Fixed time to wait for each LLM call (in seconds)
            num_iter: Number of iterations per agent
        """
        self.wait_time = wait_time
        self.num_iter = num_iter
        self.results = []

    async def run_benchmark(
        self, num_agents: int, reasoning: bool = False, description: str = ""
    ) -> Dict[str, Any]:

        mode = "Reasoning" if reasoning else "Non-Reasoning"
        print(f"\n{'='*70}")
        print(f"Starting benchmark: {description or f'{num_agents} agents ({mode})'}")
        print(f"{'='*70}")
        print("Configuration:")
        print(f"  - Number of agents: {num_agents}")
        print(f"  - Mode: {mode}")
        print(f"  - Iterations per agent: {self.num_iter}")
        print(f"  - Wait time per LLM call: {self.wait_time}s")
        print(f"{'='*70}\n")

        start_time = time.time()

        # Create agents and tasks
        agents = [
            MockAgent(
                wait_time=self.wait_time, reasoning=reasoning, num_iter=self.num_iter
            )
            for _ in range(num_agents)
        ]

        # Create tasks using asyncio.create_task
        tasks = [
            asyncio.create_task(agent.run(device_id=f"device_{i}"))
            for i, agent in enumerate(agents)
        ]

        # Run all agents concurrently
        agent_results = await asyncio.gather(*tasks)

        end_time = time.time()
        total_time = end_time - start_time

        # Calculate statistics
        total_llm_calls = sum(r["total_llm_calls"] for r in agent_results)
        avg_agent_time = sum(r["execution_time"] for r in agent_results) / len(
            agent_results
        )
        max_agent_time = max(r["execution_time"] for r in agent_results)
        min_agent_time = min(r["execution_time"] for r in agent_results)

        # Expected time if run sequentially
        expected_sequential_time = total_llm_calls * self.wait_time

        # Speedup calculation
        speedup = expected_sequential_time / total_time if total_time > 0 else 0

        # Efficiency (how well we utilize concurrency)
        efficiency = (speedup / num_agents) * 100 if num_agents > 0 else 0

        result = {
            "timestamp": datetime.now().isoformat(),
            "description": description or f"{num_agents} agents ({mode})",
            "num_agents": num_agents,
            "reasoning_mode": reasoning,
            "num_iterations": self.num_iter,
            "wait_time": self.wait_time,
            "total_execution_time": total_time,
            "avg_agent_time": avg_agent_time,
            "max_agent_time": max_agent_time,
            "min_agent_time": min_agent_time,
            "total_llm_calls": total_llm_calls,
            "expected_sequential_time": expected_sequential_time,
            "speedup": speedup,
            "efficiency": efficiency,
            "agent_results": agent_results,
        }

        self.results.append(result)

        # Print summary
        print(f"\n{'='*70}")
        print("Benchmark Results:")
        print(f"{'='*70}")
        print(f"Total execution time:        {total_time:.2f}s")
        print(f"Average agent time:          {avg_agent_time:.2f}s")
        print(
            f"Min/Max agent time:          {min_agent_time:.2f}s / {max_agent_time:.2f}s"
        )
        print(f"Total LLM calls:             {total_llm_calls}")
        print(f"Expected sequential time:    {expected_sequential_time:.2f}s")
        print(f"Speedup (vs sequential):     {speedup:.2f}x")
        print(f"Parallel efficiency:         {efficiency:.1f}%")
        print(f"{'='*70}\n")

        return result

    def save_results(self, output_dir: str = "benchmark_results"):
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_path / f"benchmark_results_{timestamp}.json"

        # Remove agent_results from saved data (too verbose)
        results_to_save = [
            {k: v for k, v in result.items() if k != "agent_results"}
            for result in self.results
        ]

        with open(filename, "w") as f:
            json.dump(results_to_save, f, indent=2)

        print(f"\n✓ Results saved to: {filename}")
        return filename

    def print_comparison_table(self):
        """
        Print a comparison table of all benchmark results.
        """
        print("\n" + "=" * 100)
        print("BENCHMARK COMPARISON TABLE")
        print("=" * 100)
        print(
            f"{'Config':<35} {'Agents':<8} {'Mode':<15} {'Time (s)':<12} {'Speedup':<12} {'Efficiency'}"
        )
        print("-" * 100)

        for result in self.results:
            config = result["description"]
            agents = result["num_agents"]
            mode = "Reasoning" if result["reasoning_mode"] else "Non-Reasoning"
            time_taken = result["total_execution_time"]
            speedup = result["speedup"]
            efficiency = result["efficiency"]

            print(
                f"{config:<35} {agents:<8} {mode:<15} {time_taken:<12.2f} {speedup:<12.2f} {efficiency:.1f}%"
            )

        print("=" * 100 + "\n")

    def visualize_results(self, output_dir: str = "benchmark_results"):
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Separate results by reasoning mode
        non_reasoning = [r for r in self.results if not r["reasoning_mode"]]
        reasoning = [r for r in self.results if r["reasoning_mode"]]

        # Create figure with multiple subplots
        fig = plt.figure(figsize=(16, 10))

        # 1. Execution Time Comparison
        ax1 = plt.subplot(2, 3, 1)
        if non_reasoning:
            agents_nr = [r["num_agents"] for r in non_reasoning]
            times_nr = [r["total_execution_time"] for r in non_reasoning]
            ax1.plot(
                agents_nr,
                times_nr,
                "bo-",
                label="Non-Reasoning",
                linewidth=2,
                markersize=8,
            )

        if reasoning:
            agents_r = [r["num_agents"] for r in reasoning]
            times_r = [r["total_execution_time"] for r in reasoning]
            ax1.plot(
                agents_r, times_r, "ro-", label="Reasoning", linewidth=2, markersize=8
            )

        ax1.set_xlabel("Number of Agents", fontsize=11)
        ax1.set_ylabel("Execution Time (seconds)", fontsize=11)
        ax1.set_title(
            "Total Execution Time vs Number of Agents", fontsize=12, fontweight="bold"
        )
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale("log")

        # 2. Speedup Comparison
        ax2 = plt.subplot(2, 3, 2)
        if non_reasoning:
            speedups_nr = [r["speedup"] for r in non_reasoning]
            ax2.plot(
                agents_nr,
                speedups_nr,
                "bo-",
                label="Non-Reasoning",
                linewidth=2,
                markersize=8,
            )

        if reasoning:
            speedups_r = [r["speedup"] for r in reasoning]
            ax2.plot(
                agents_r,
                speedups_r,
                "ro-",
                label="Reasoning",
                linewidth=2,
                markersize=8,
            )

        # Add ideal linear speedup line
        max_agents = max([r["num_agents"] for r in self.results])
        ax2.plot(
            [1, max_agents],
            [1, max_agents],
            "g--",
            alpha=0.5,
            label="Ideal Linear",
            linewidth=2,
        )

        ax2.set_xlabel("Number of Agents", fontsize=11)
        ax2.set_ylabel("Speedup Factor", fontsize=11)
        ax2.set_title("Speedup vs Number of Agents", fontsize=12, fontweight="bold")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale("log")
        ax2.set_yscale("log")

        # 3. Efficiency Comparison
        ax3 = plt.subplot(2, 3, 3)
        if non_reasoning:
            efficiency_nr = [r["efficiency"] for r in non_reasoning]
            ax3.plot(
                agents_nr,
                efficiency_nr,
                "bo-",
                label="Non-Reasoning",
                linewidth=2,
                markersize=8,
            )

        if reasoning:
            efficiency_r = [r["efficiency"] for r in reasoning]
            ax3.plot(
                agents_r,
                efficiency_r,
                "ro-",
                label="Reasoning",
                linewidth=2,
                markersize=8,
            )

        ax3.axhline(
            y=100,
            color="g",
            linestyle="--",
            alpha=0.5,
            label="100% Efficient",
            linewidth=2,
        )
        ax3.set_xlabel("Number of Agents", fontsize=11)
        ax3.set_ylabel("Efficiency (%)", fontsize=11)
        ax3.set_title(
            "Parallel Efficiency vs Number of Agents", fontsize=12, fontweight="bold"
        )
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale("log")

        # 4. Bar chart: Execution time comparison
        ax4 = plt.subplot(2, 3, 4)
        labels = [f"{r['num_agents']}" for r in self.results]
        times = [r["total_execution_time"] for r in self.results]
        colors = ["blue" if not r["reasoning_mode"] else "red" for r in self.results]

        bars = ax4.bar(range(len(labels)), times, color=colors, alpha=0.7)
        ax4.set_xlabel("Configuration", fontsize=11)
        ax4.set_ylabel("Execution Time (seconds)", fontsize=11)
        ax4.set_title("Execution Time by Configuration", fontsize=12, fontweight="bold")
        ax4.set_xticks(range(len(labels)))
        ax4.set_xticklabels(labels, rotation=45, ha="right")
        ax4.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for i, (bar, time_val) in enumerate(zip(bars, times)):
            height = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{time_val:.1f}s",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # 5. LLM Calls Throughput
        ax5 = plt.subplot(2, 3, 5)
        if non_reasoning:
            throughput_nr = [
                r["total_llm_calls"] / r["total_execution_time"] for r in non_reasoning
            ]
            ax5.plot(
                agents_nr,
                throughput_nr,
                "bo-",
                label="Non-Reasoning",
                linewidth=2,
                markersize=8,
            )

        if reasoning:
            throughput_r = [
                r["total_llm_calls"] / r["total_execution_time"] for r in reasoning
            ]
            ax5.plot(
                agents_r,
                throughput_r,
                "ro-",
                label="Reasoning",
                linewidth=2,
                markersize=8,
            )

        ax5.set_xlabel("Number of Agents", fontsize=11)
        ax5.set_ylabel("LLM Calls per Second", fontsize=11)
        ax5.set_title("Throughput vs Number of Agents", fontsize=12, fontweight="bold")
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_xscale("log")

        # 6. Summary Statistics Table
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis("tight")
        ax6.axis("off")

        summary_data = []
        for r in self.results:
            mode = "R" if r["reasoning_mode"] else "NR"
            summary_data.append(
                [
                    f"{r['num_agents']} ({mode})",
                    f"{r['total_execution_time']:.2f}s",
                    f"{r['speedup']:.2f}x",
                    f"{r['efficiency']:.1f}%",
                ]
            )

        table = ax6.table(
            cellText=summary_data,
            colLabels=["Config", "Time", "Speedup", "Efficiency"],
            cellLoc="center",
            loc="center",
            colWidths=[0.3, 0.25, 0.2, 0.25],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Style header row
        for i in range(4):
            table[(0, i)].set_facecolor("#4472C4")
            table[(0, i)].set_text_props(weight="bold", color="white")

        ax6.set_title("Summary Statistics", fontsize=12, fontweight="bold", pad=20)

        plt.tight_layout()

        # Save the figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_path / f"benchmark_visualization_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"✓ Visualization saved to: {filename}")

        # Create a second figure for detailed comparison
        self._create_detailed_comparison(output_path, timestamp)

        plt.close("all")

    def _create_detailed_comparison(self, output_path: Path, timestamp: str):

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Get data
        non_reasoning = [r for r in self.results if not r["reasoning_mode"]]
        reasoning = [r for r in self.results if r["reasoning_mode"]]

        # 1. Side-by-side execution time comparison
        ax1 = axes[0, 0]
        x = range(len(non_reasoning))
        width = 0.35
        if non_reasoning:
            agents_nr = [r["num_agents"] for r in non_reasoning]
            times_nr = [r["total_execution_time"] for r in non_reasoning]
            ax1.bar(
                [i - width / 2 for i in x],
                times_nr,
                width,
                label="Non-Reasoning",
                color="blue",
                alpha=0.7,
            )

        if reasoning:
            times_r = [r["total_execution_time"] for r in reasoning]
            ax1.bar(
                [i + width / 2 for i in x],
                times_r,
                width,
                label="Reasoning",
                color="red",
                alpha=0.7,
            )

        ax1.set_xlabel("Number of Agents", fontsize=11)
        ax1.set_ylabel("Execution Time (seconds)", fontsize=11)
        ax1.set_title("Execution Time Comparison", fontsize=12, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels([str(a) for a in agents_nr])
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis="y")

        # 2. Scalability analysis
        ax2 = axes[0, 1]
        if non_reasoning and len(non_reasoning) > 1:
            # Calculate relative time (normalized to 1 agent)
            base_time_nr = non_reasoning[0]["total_execution_time"]
            relative_times_nr = [
                r["total_execution_time"] / base_time_nr for r in non_reasoning
            ]
            ax2.plot(
                agents_nr,
                relative_times_nr,
                "bo-",
                label="Non-Reasoning",
                linewidth=2,
                markersize=8,
            )

        if reasoning and len(reasoning) > 1:
            base_time_r = reasoning[0]["total_execution_time"]
            relative_times_r = [
                r["total_execution_time"] / base_time_r for r in reasoning
            ]
            agents_r = [r["num_agents"] for r in reasoning]
            ax2.plot(
                agents_r,
                relative_times_r,
                "ro-",
                label="Reasoning",
                linewidth=2,
                markersize=8,
            )

        ax2.set_xlabel("Number of Agents", fontsize=11)
        ax2.set_ylabel("Relative Time (normalized to 1 agent)", fontsize=11)
        ax2.set_title("Scalability Analysis", fontsize=12, fontweight="bold")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale("log")

        # 3. Average agent time
        ax3 = axes[1, 0]
        if non_reasoning:
            avg_times_nr = [r["avg_agent_time"] for r in non_reasoning]
            ax3.plot(
                agents_nr,
                avg_times_nr,
                "bo-",
                label="Non-Reasoning",
                linewidth=2,
                markersize=8,
            )

        if reasoning:
            avg_times_r = [r["avg_agent_time"] for r in reasoning]
            ax3.plot(
                agents_r,
                avg_times_r,
                "ro-",
                label="Reasoning",
                linewidth=2,
                markersize=8,
            )

        ax3.set_xlabel("Number of Agents", fontsize=11)
        ax3.set_ylabel("Average Agent Time (seconds)", fontsize=11)
        ax3.set_title("Average Agent Execution Time", fontsize=12, fontweight="bold")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale("log")

        # 4. Key findings text
        ax4 = axes[1, 1]
        ax4.axis("off")

        # Calculate key findings
        if non_reasoning and reasoning:
            max_speedup_nr = max([r["speedup"] for r in non_reasoning])
            max_speedup_r = max([r["speedup"] for r in reasoning])
            best_efficiency_nr = max([r["efficiency"] for r in non_reasoning])
            best_efficiency_r = max([r["efficiency"] for r in reasoning])

            findings_text = f"""
KEY FINDINGS

Non-Reasoning Mode:
• Maximum speedup: {max_speedup_nr:.2f}x
• Best efficiency: {best_efficiency_nr:.1f}%
• Configuration: {non_reasoning[-1]['num_agents']} agents

Reasoning Mode:
• Maximum speedup: {max_speedup_r:.2f}x
• Best efficiency: {best_efficiency_r:.1f}%
• Configuration: {reasoning[-1]['num_agents']} agents

Observations:
• Asyncio enables near-linear scaling
  up to {max([r['num_agents'] for r in self.results])} concurrent agents
• Reasoning mode has {reasoning[0]['total_llm_calls'] / non_reasoning[0]['total_llm_calls']:.1f}x more
  LLM calls per iteration
• Both modes benefit equally from
  concurrent execution
            """
            ax4.text(
                0.1,
                0.5,
                findings_text,
                fontsize=10,
                verticalalignment="center",
                family="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
            )

        plt.tight_layout()
        filename = output_path / f"benchmark_detailed_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"✓ Detailed visualization saved to: {filename}")


async def main():
    """
    Main function to run the benchmark suite.
    """
    # Configuration
    WAIT_TIME = 3.0  # seconds per LLM call
    NUM_ITERATIONS = 5  # iterations per agent

    print("\n" + "=" * 70)
    print("DROIDRUN PERFORMANCE BENCHMARK")
    print("=" * 70)
    print(f"Wait time per LLM call: {WAIT_TIME}s")
    print(f"Iterations per agent: {NUM_ITERATIONS}")
    print("=" * 70)

    # Create benchmark runner
    runner = BenchmarkRunner(wait_time=WAIT_TIME, num_iter=NUM_ITERATIONS)

    # Run full benchmark suite
    await runner.run_full_benchmark_suite()

    # Print comparison table
    runner.print_comparison_table()

    # Save results
    runner.save_results()

    # Create visualizations
    runner.visualize_results()

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
