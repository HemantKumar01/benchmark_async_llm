# DroidRun Performance Benchmark

This directory contains tools for benchmarking the performance of DroidRun agents with multiple parallel devices using mock LLMs.

## Files

- **`mock_llm.py`**: Contains `MockLLM` and `MockAgent` classes that simulate LLM response latency
  - **MockLLM**: waits for a constant duration (wait_time) then returns a hard-coded response based on agent type
  - **MockAgent**: simulates an agent workflow (non-reasoning or reasoning) using the MockLLM 
- **`benchmark_llm.py`**: Main benchmark script that runs comprehensive performance tests
- **`main.py`**: Running benchmark 

Saves the benchmarks in `benchmark_results/` directory.

## Requirements

```bash
pip install -r requirements.txt
```

## Quick Start

Run the complete benchmark suite:

```bash
cd benchmark
python benchmark_llm.py
```

This will:
1. Run benchmarks with 1, 10, and 100 concurrent agents
2. Test both non-reasoning and reasoning modes
3. Generate comparison tables and statistics
4. Save results to JSON
5. Create visualization charts (if matplotlib is installed)

## Configuration

You can customize the benchmark parameters in `benchmark_llm.py`:

```python
WAIT_TIME = 3.0        # seconds per LLM call (simulated latency)
NUM_ITERATIONS = 5     # iterations per agent
```

## Mock Components

### MockLLM Class

Simulates an LLM with configurable latency:

```python
from mock_llm import MockLLM

# Create a mock LLM for codeact agent
llm = MockLLM(agent_type="codeact", wait_time=3.0)

# Async call
response = await llm.generate("prompt")

# Sync call
response = llm("prompt")
```

Supported agent types:
- `"codeact"`: Non-reasoning agent
- `"manager"`: Planning agent (reasoning mode)
- `"executor"`: Execution agent (reasoning mode)

### MockAgent Class

Simulates a complete agent workflow:

```python
from mock_llm import MockAgent

# Non-reasoning mode (uses codeact agent)
agent = MockAgent(wait_time=3.0, reasoning=False, num_iter=5)
result = await agent.run(device_id="device_0")

# Reasoning mode (alternates between manager and executor)
agent = MockAgent(wait_time=3.0, reasoning=True, num_iter=3)
result = await agent.run(device_id="device_0")
```

## Benchmark Results

The benchmark generates:

### 1. Console Output
- Real-time progress and statistics
- Comparison tables
- Summary of key findings

### 2. JSON Results
Saved to `benchmark_results/benchmark_results_TIMESTAMP.json`:

```json
{
  "timestamp": "2025-11-11T...",
  "num_agents": 10,
  "reasoning_mode": false,
  "total_execution_time": 15.23,
  "speedup": 9.85,
  "efficiency": 98.5,
  ...
}
```

### 3. Visualizations
Saved to `benchmark_results/` (requires matplotlib):

- **`benchmark_visualization_TIMESTAMP.png`**: Main dashboard with 6 charts
  - Execution time vs number of agents
  - Speedup comparison
  - Parallel efficiency
  - Bar chart comparison
  - Throughput analysis
  - Summary statistics table

- **`benchmark_detailed_TIMESTAMP.png`**: Detailed analysis with 4 charts
  - Side-by-side execution time comparison
  - Scalability analysis
  - Average agent execution time
  - Key findings summary

## Understanding the Metrics

### Speedup
How much faster the concurrent execution is compared to sequential execution:
- **Speedup = Expected Sequential Time / Actual Time**
- Ideal: Linear (e.g., 10x speedup with 10 agents)

### Efficiency
How well we're utilizing the concurrent execution:
- **Efficiency = (Speedup / Number of Agents) × 100%**
- 100% = perfect utilization
- Lower values indicate overhead or contention

### Throughput
Number of LLM calls processed per second:
- **Throughput = Total LLM Calls / Execution Time**
- Higher is better
- Should scale linearly with number of agents
