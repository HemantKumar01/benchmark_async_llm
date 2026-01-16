import asyncio
import time
from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import json
import os

from mock_llm import MockLLM
from droidrun.agent.droid import DroidAgent
from droidrun.config_manager.config_manager import AgentConfig, DroidrunConfig


class ParallelBenchmark:
    """Benchmark parallel execution of DroidAgent instances."""

    def __init__(self):
        self.results = {"non_reasoning": [], "reasoning": []}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    async def run_single_agent(
        self, agent_id: int, goal: str, reasoning: bool = False
    ) -> Dict:
        """Run a single DroidAgent and measure its execution time."""
        start_time = time.time()

        # Configure agent
        config = DroidrunConfig(agent=AgentConfig(reasoning=reasoning, max_steps=10))

        if reasoning:
            # Reasoning mode uses multiple LLMs
            llms = {
                "manager": MockLLM(agent_type="manager", wait_time=0.3),
                "executor": MockLLM(agent_type="executor", wait_time=0.3),
                "codeact": MockLLM(agent_type="codeact", wait_time=0.3),
                "text_manipulator": MockLLM(
                    agent_type="text_manipulator", wait_time=0.2
                ),
                "app_opener": MockLLM(agent_type="app_opener", wait_time=0.2),
                "scripter": MockLLM(agent_type="codeact", wait_time=0.3),
            }
            mock_llm = llms
            total_calls_func = lambda: sum(llm.call_counter for llm in llms.values())
        else:
            # Non-reasoning mode uses single LLM
            mock_llm = MockLLM(agent_type="codeact", wait_time=0.5)
            total_calls_func = lambda: mock_llm.call_counter

        try:
            agent = DroidAgent(
                goal=f"{goal} (Agent {agent_id})", config=config, llms=mock_llm
            )
            result = await agent.run()

            execution_time = time.time() - start_time

            return {
                "agent_id": agent_id,
                "execution_time": execution_time,
                "success": True,
                "llm_calls": total_calls_func(),
                "result": result,
                "reasoning": reasoning,
            }
        except Exception as e:
            execution_time = time.time() - start_time
            return {
                "agent_id": agent_id,
                "execution_time": execution_time,
                "success": False,
                "error": str(e),
                "llm_calls": 0,
                "reasoning": reasoning,
            }

    async def run_parallel_agents(
        self, num_agents: int, goal: str = "Open Settings", reasoning: bool = False
    ) -> Dict:
        """Run multiple agents in parallel and measure total time."""
        mode_name = "Reasoning" if reasoning else "Non-Reasoning"
        print(f"\n{'='*80}")
        print(f"Running {num_agents} agents in parallel ({mode_name} Mode)...")
        print(f"{'='*80}")

        start_time = time.time()

        # Create tasks for all agents
        tasks = [self.run_single_agent(i, goal, reasoning) for i in range(num_agents)]

        # Run all agents in parallel
        agent_results = await asyncio.gather(*tasks)

        total_time = time.time() - start_time

        # Calculate statistics
        successful_agents = sum(1 for r in agent_results if r["success"])
        failed_agents = num_agents - successful_agents
        avg_agent_time = np.mean([r["execution_time"] for r in agent_results])
        median_agent_time = np.median([r["execution_time"] for r in agent_results])
        total_llm_calls = sum(r["llm_calls"] for r in agent_results)

        result = {
            "num_agents": num_agents,
            "total_time": total_time,
            "avg_agent_time": avg_agent_time,
            "median_agent_time": median_agent_time,
            "successful_agents": successful_agents,
            "failed_agents": failed_agents,
            "total_llm_calls": total_llm_calls,
            "agent_results": agent_results,
            "reasoning": reasoning,
        }

        print(f"\nResults for {num_agents} agents ({mode_name} Mode):")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Avg Agent Time: {avg_agent_time:.2f}s")
        print(f"  Median Agent Time: {median_agent_time:.2f}s")
        print(f"  Successful: {successful_agents}/{num_agents}")
        print(f"  Total LLM Calls: {total_llm_calls}")
        print(f"  Throughput: {num_agents/total_time:.2f} agents/second")

        return result

    async def run_benchmark_suite(self, agent_counts: List[int]):
        """Run benchmarks for different numbers of parallel agents in both modes."""
        print(f"\n{'#'*80}")
        print(f"# PARALLEL DROIDAGENT BENCHMARK SUITE")
        print(f"# Timestamp: {self.timestamp}")
        print(f"# Testing agent counts: {agent_counts}")
        print(
            f"# Testing modes: Non-Reasoning (CodeAct) and Reasoning (Manager + Executor)"
        )
        print(f"{'#'*80}\n")

        # Run Non-Reasoning benchmarks
        print(f"\n{'*'*80}")
        print(f"* PHASE 1: NON-REASONING MODE (Single LLM - CodeAct)")
        print(f"{'*'*80}")
        for count in agent_counts:
            result = await self.run_parallel_agents(count, reasoning=False)
            self.results["non_reasoning"].append(result)
            await asyncio.sleep(0.5)

        # Run Reasoning benchmarks
        print(f"\n{'*'*80}")
        print(f"* PHASE 2: REASONING MODE (Multiple LLMs - Manager + Executor)")
        print(f"{'*'*80}")
        for count in agent_counts:
            result = await self.run_parallel_agents(count, reasoning=True)
            self.results["reasoning"].append(result)
            await asyncio.sleep(0.5)

        return self.results

    def generate_consolidated_report(self):
        """Generate a consolidated report with all benchmark results."""
        print(f"\n{'#'*80}")
        print(f"# CONSOLIDATED BENCHMARK RESULTS")
        print(f"{'#'*80}\n")

        for mode_name, mode_key in [
            ("NON-REASONING", "non_reasoning"),
            ("REASONING", "reasoning"),
        ]:
            results = self.results[mode_key]
            if not results:
                continue

            print(f"\n{'='*80}")
            print(f" {mode_name} MODE RESULTS")
            print(f"{'='*80}\n")

            # Create table
            print(
                f"{'Agents':<10} {'Total Time':<12} {'Avg Time':<12} {'Median Time':<12} {'Success Rate':<13} {'Throughput':<15} {'LLM Calls':<12}"
            )
            print(f"{'-'*95}")

            for result in results:
                num_agents = result["num_agents"]
                total_time = result["total_time"]
                avg_time = result["avg_agent_time"]
                median_time = result["median_agent_time"]
                success_rate = result["successful_agents"] / num_agents * 100
                throughput = num_agents / total_time
                llm_calls = result["total_llm_calls"]

                print(
                    f"{num_agents:<10} {total_time:<12.2f} {avg_time:<12.2f} {median_time:<12.2f} {success_rate:<12.1f}% {throughput:<15.2f} {llm_calls:<12}"
                )

            # Calculate speedup vs single agent
            if len(results) > 0:
                print(f"\n{'-'*80}")
                print(f"SPEEDUP ANALYSIS ({mode_name} MODE):")
                print(f"{'-'*80}")

                single_agent_time = (
                    results[0]["avg_agent_time"] if len(results) > 0 else 0
                )

                for result in results:
                    num_agents = result["num_agents"]
                    total_time = result["total_time"]

                    # Theoretical time if run sequentially
                    theoretical_sequential_time = single_agent_time * num_agents
                    speedup = (
                        theoretical_sequential_time / total_time
                        if total_time > 0
                        else 0
                    )
                    efficiency = (speedup / num_agents * 100) if num_agents > 0 else 0

                    print(f"  {num_agents} agents:")
                    print(
                        f"    Theoretical Sequential Time: {theoretical_sequential_time:.2f}s"
                    )
                    print(f"    Actual Parallel Time: {total_time:.2f}s")
                    print(f"    Speedup: {speedup:.2f}x")
                    print(f"    Parallel Efficiency: {efficiency:.1f}%")
                    print()

        # Mode comparison
        if self.results["non_reasoning"] and self.results["reasoning"]:
            print(f"\n{'#'*80}")
            print(f"# MODE COMPARISON (Non-Reasoning vs Reasoning)")
            print(f"{'#'*80}\n")

            print(
                f"{'Agents':<10} {'NR Time':<12} {'R Time':<12} {'NR Throughput':<15} {'R Throughput':<15} {'Overhead':<12}"
            )
            print(f"{'-'*90}")

            for nr_result, r_result in zip(
                self.results["non_reasoning"], self.results["reasoning"]
            ):
                num_agents = nr_result["num_agents"]
                nr_time = nr_result["total_time"]
                r_time = r_result["total_time"]
                nr_throughput = num_agents / nr_time
                r_throughput = num_agents / r_time
                overhead = ((r_time - nr_time) / nr_time * 100) if nr_time > 0 else 0

                print(
                    f"{num_agents:<10} {nr_time:<12.2f} {r_time:<12.2f} {nr_throughput:<15.2f} {r_throughput:<15.2f} {overhead:+.1f}%"
                )

    def save_results_json(self):
        """Save detailed results to JSON file."""
        output_dir = "benchmark_results"
        os.makedirs(output_dir, exist_ok=True)

        filepath = f"{output_dir}/benchmark_{self.timestamp}.json"

        # Prepare data (remove agent_results details for cleaner JSON)
        summary_results = {"non_reasoning": [], "reasoning": []}
        for mode in ["non_reasoning", "reasoning"]:
            for result in self.results[mode]:
                summary = {k: v for k, v in result.items() if k != "agent_results"}
                summary_results[mode].append(summary)

        with open(filepath, "w") as f:
            json.dump(
                {"timestamp": self.timestamp, "results": summary_results}, f, indent=2
            )

        print(f"\nDetailed results saved to: {filepath}")
        return filepath

    def generate_graphs(self):
        """Generate visualization graphs for the benchmark results."""
        output_dir = "benchmark_results"
        os.makedirs(output_dir, exist_ok=True)

        nr_results = self.results["non_reasoning"]
        r_results = self.results["reasoning"]

        if not nr_results and not r_results:
            print("No results to plot")
            return

        # Extract data for both modes
        agent_counts = (
            [r["num_agents"] for r in nr_results]
            if nr_results
            else [r["num_agents"] for r in r_results]
        )

        nr_total_times = [r["total_time"] for r in nr_results] if nr_results else []
        nr_avg_times = [r["avg_agent_time"] for r in nr_results] if nr_results else []
        nr_throughputs = (
            [r["num_agents"] / r["total_time"] for r in nr_results]
            if nr_results
            else []
        )

        r_total_times = [r["total_time"] for r in r_results] if r_results else []
        r_avg_times = [r["avg_agent_time"] for r in r_results] if r_results else []
        r_throughputs = (
            [r["num_agents"] / r["total_time"] for r in r_results] if r_results else []
        )

        # Create figure with subplots (2x2 for main metrics)
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle(
            f"DroidAgent Parallel Benchmark: Non-Reasoning vs Reasoning\n{self.timestamp}",
            fontsize=16,
            fontweight="bold",
        )

        # Colors for modes
        nr_color = "#2E86AB"  # Blue for non-reasoning
        r_color = "#E94F37"  # Red for reasoning

        # 1. Total Execution Time vs Number of Agents
        ax1 = axes[0, 0]
        if nr_total_times:
            ax1.plot(
                agent_counts,
                nr_total_times,
                marker="o",
                linewidth=2,
                markersize=8,
                color=nr_color,
                label="Non-Reasoning",
            )
        if r_total_times:
            ax1.plot(
                agent_counts,
                r_total_times,
                marker="s",
                linewidth=2,
                markersize=8,
                color=r_color,
                label="Reasoning",
            )
        ax1.set_xlabel("Number of Agents", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Total Execution Time (seconds)", fontsize=12, fontweight="bold")
        ax1.set_title(
            "Total Execution Time vs Number of Agents", fontsize=13, fontweight="bold"
        )
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale("log")

        # 2. Average Agent Execution Time
        ax2 = axes[0, 1]
        if nr_avg_times:
            ax2.plot(
                agent_counts,
                nr_avg_times,
                marker="o",
                linewidth=2,
                markersize=8,
                color=nr_color,
                label="Non-Reasoning",
            )
        if r_avg_times:
            ax2.plot(
                agent_counts,
                r_avg_times,
                marker="s",
                linewidth=2,
                markersize=8,
                color=r_color,
                label="Reasoning",
            )
        ax2.set_xlabel("Number of Agents", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Average Agent Time (seconds)", fontsize=12, fontweight="bold")
        ax2.set_title("Average Agent Execution Time", fontsize=13, fontweight="bold")
        ax2.legend(loc="upper left")
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale("log")

        # 3. Throughput (Agents per Second)
        ax3 = axes[1, 0]
        if nr_throughputs:
            ax3.plot(
                agent_counts,
                nr_throughputs,
                marker="o",
                linewidth=2,
                markersize=8,
                color=nr_color,
                label="Non-Reasoning",
            )
        if r_throughputs:
            ax3.plot(
                agent_counts,
                r_throughputs,
                marker="s",
                linewidth=2,
                markersize=8,
                color=r_color,
                label="Reasoning",
            )
        ax3.set_xlabel("Number of Agents", fontsize=12, fontweight="bold")
        ax3.set_ylabel("Throughput (agents/second)", fontsize=12, fontweight="bold")
        ax3.set_title("Agent Throughput Comparison", fontsize=13, fontweight="bold")
        ax3.legend(loc="upper left")
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale("log")

        # 4. Speedup Analysis for both modes
        ax4 = axes[1, 1]
        if nr_results:
            nr_single_time = nr_results[0]["avg_agent_time"]
            nr_theoretical = [nr_single_time * n for n in agent_counts]
            nr_speedups = [t / a for t, a in zip(nr_theoretical, nr_total_times)]
            ax4.plot(
                agent_counts,
                nr_speedups,
                marker="o",
                linewidth=2,
                markersize=8,
                color=nr_color,
                label="Non-Reasoning Speedup",
            )
        if r_results:
            r_single_time = r_results[0]["avg_agent_time"]
            r_theoretical = [r_single_time * n for n in agent_counts]
            r_speedups = [t / a for t, a in zip(r_theoretical, r_total_times)]
            ax4.plot(
                agent_counts,
                r_speedups,
                marker="s",
                linewidth=2,
                markersize=8,
                color=r_color,
                label="Reasoning Speedup",
            )

        # Ideal linear speedup
        ax4.plot(
            agent_counts,
            agent_counts,
            linestyle="--",
            linewidth=2,
            label="Ideal Linear Speedup",
            color="#06A77D",
            alpha=0.7,
        )
        ax4.set_xlabel("Number of Agents", fontsize=12, fontweight="bold")
        ax4.set_ylabel("Speedup Factor", fontsize=12, fontweight="bold")
        ax4.set_title("Speedup vs Ideal Linear Speedup", fontsize=13, fontweight="bold")
        ax4.legend(loc="upper left")
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale("log")
        ax4.set_yscale("log")

        plt.tight_layout()

        # Save the main figure
        graph_path = f"{output_dir}/benchmark_graphs_{self.timestamp}.png"
        plt.savefig(graph_path, dpi=300, bbox_inches="tight")
        print(f"Main graphs saved to: {graph_path}")

        # Create comparison bar chart
        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
        fig2.suptitle(
            f"Mode Comparison\n{self.timestamp}", fontsize=14, fontweight="bold"
        )

        x = np.arange(len(agent_counts))
        width = 0.35

        # Total time comparison
        ax_time = axes2[0]
        if nr_total_times:
            bars1 = ax_time.bar(
                x - width / 2,
                nr_total_times,
                width,
                label="Non-Reasoning",
                color=nr_color,
                alpha=0.8,
            )
        if r_total_times:
            bars2 = ax_time.bar(
                x + width / 2,
                r_total_times,
                width,
                label="Reasoning",
                color=r_color,
                alpha=0.8,
            )
        ax_time.set_xlabel("Number of Agents", fontsize=12, fontweight="bold")
        ax_time.set_ylabel("Total Time (seconds)", fontsize=12, fontweight="bold")
        ax_time.set_title(
            "Total Execution Time Comparison", fontsize=13, fontweight="bold"
        )
        ax_time.set_xticks(x)
        ax_time.set_xticklabels([str(n) for n in agent_counts])
        ax_time.legend()
        ax_time.grid(True, alpha=0.3, axis="y")

        # Throughput comparison
        ax_tp = axes2[1]
        if nr_throughputs:
            ax_tp.bar(
                x - width / 2,
                nr_throughputs,
                width,
                label="Non-Reasoning",
                color=nr_color,
                alpha=0.8,
            )
        if r_throughputs:
            ax_tp.bar(
                x + width / 2,
                r_throughputs,
                width,
                label="Reasoning",
                color=r_color,
                alpha=0.8,
            )
        ax_tp.set_xlabel("Number of Agents", fontsize=12, fontweight="bold")
        ax_tp.set_ylabel("Throughput (agents/second)", fontsize=12, fontweight="bold")
        ax_tp.set_title("Throughput Comparison", fontsize=13, fontweight="bold")
        ax_tp.set_xticks(x)
        ax_tp.set_xticklabels([str(n) for n in agent_counts])
        ax_tp.legend()
        ax_tp.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        comparison_path = f"{output_dir}/mode_comparison_{self.timestamp}.png"
        plt.savefig(comparison_path, dpi=300, bbox_inches="tight")
        print(f"Mode comparison graph saved to: {comparison_path}")

        # Create success rate comparison
        fig3, ax = plt.subplots(figsize=(12, 6))

        nr_success_rates = (
            [r["successful_agents"] / r["num_agents"] * 100 for r in nr_results]
            if nr_results
            else []
        )
        r_success_rates = (
            [r["successful_agents"] / r["num_agents"] * 100 for r in r_results]
            if r_results
            else []
        )

        if nr_success_rates:
            ax.bar(
                x - width / 2,
                nr_success_rates,
                width,
                label="Non-Reasoning",
                color=nr_color,
                alpha=0.8,
            )
        if r_success_rates:
            ax.bar(
                x + width / 2,
                r_success_rates,
                width,
                label="Reasoning",
                color=r_color,
                alpha=0.8,
            )

        ax.set_xlabel("Number of Agents", fontsize=12, fontweight="bold")
        ax.set_ylabel("Success Rate (%)", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Agent Success Rate by Mode\n{self.timestamp}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in agent_counts])
        ax.set_ylim([0, 105])
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        success_graph_path = f"{output_dir}/success_rate_{self.timestamp}.png"
        plt.savefig(success_graph_path, dpi=300, bbox_inches="tight")
        print(f"Success rate graph saved to: {success_graph_path}")

        # Create LLM calls comparison
        fig4, ax4 = plt.subplots(figsize=(12, 6))

        nr_llm_calls = [r["total_llm_calls"] for r in nr_results] if nr_results else []
        r_llm_calls = [r["total_llm_calls"] for r in r_results] if r_results else []

        if nr_llm_calls:
            ax4.bar(
                x - width / 2,
                nr_llm_calls,
                width,
                label="Non-Reasoning",
                color=nr_color,
                alpha=0.8,
            )
        if r_llm_calls:
            ax4.bar(
                x + width / 2,
                r_llm_calls,
                width,
                label="Reasoning",
                color=r_color,
                alpha=0.8,
            )

        ax4.set_xlabel("Number of Agents", fontsize=12, fontweight="bold")
        ax4.set_ylabel("Total LLM Calls", fontsize=12, fontweight="bold")
        ax4.set_title(
            f"Total LLM Calls by Mode\n{self.timestamp}", fontsize=14, fontweight="bold"
        )
        ax4.set_xticks(x)
        ax4.set_xticklabels([str(n) for n in agent_counts])
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        llm_graph_path = f"{output_dir}/llm_calls_{self.timestamp}.png"
        plt.savefig(llm_graph_path, dpi=300, bbox_inches="tight")
        print(f"LLM calls graph saved to: {llm_graph_path}")

        return graph_path, comparison_path, success_graph_path, llm_graph_path


async def main():
    """Main benchmark execution function."""
    benchmark = ParallelBenchmark()

    # Test configurations: 1, 10, 100, 500 agents
    agent_counts = [1, 10, 100, 500]

    # Run the benchmark suite
    await benchmark.run_benchmark_suite(agent_counts)

    # Generate consolidated report
    benchmark.generate_consolidated_report()

    # Save results to JSON
    benchmark.save_results_json()

    # Generate and save graphs
    benchmark.generate_graphs()

    print(f"\n{'#'*80}")
    print("# BENCHMARK COMPLETE!")
    print(f"{'#'*80}\n")


if __name__ == "__main__":
    asyncio.run(main())
