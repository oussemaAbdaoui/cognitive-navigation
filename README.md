# Cognitive Navigation System with LLM-Based Planning

[![tests](https://github.com/oussemaAbdaoui/cognitive-navigation/actions/workflows/tests.yml/badge.svg)](https://github.com/oussemaAbdaoui/cognitive-navigation/actions/workflows/tests.yml)

## Overview

This project is a grid-world robot navigation system that combines classical spatial bookkeeping with Large Language Model (LLM) decision making. At each step the robot reads three range sensors (front, left, right), updates an internal occupancy-grid map, and sends the map, its state and the sensor readings to an LLM, which returns a single action (`TURN LEFT/RIGHT` or `MOVE FORWARD n`). The returned action is validated against a schema before it is executed.

The system is evaluated in a simulated grid environment. The LLM is called through the Groq chat-completions API (model `llama-3.3-70b-versatile`).

## Key Features

- **Hybrid architecture**: deterministic grid mapping, motion model and progress tracking in Python; high-level action selection by an LLM.
- **Multiple reasoning kernels**: a kernel is a system-prompt prefix placed before the shared navigation prompt. `saamv1` and `saamv2` are SAAM-style structured prompts; `native` adds no prefix (the plain LLM planner). All three kernels call the LLM; there is no non-LLM baseline planner.
- **Internal map**: an occupancy grid with per-cell type (`CLEAR`, `OBSTACLE`, `ROBOT`, `GOAL`, `UNKNOWN`) and a confidence value (fixed per source: 0.8 for cells sensed clear, 0.9 for sensed obstacles, 1.0 for known cells). Confidence is stored but not yet used in decisions.
- **Progress monitoring**: Manhattan distance to the goal is tracked after every move and consecutive increases are counted. The count is reported to the LLM as a warning in the prompt; reaching 3 is logged as a critical event. Recovery behaviour (backtracking, rescanning) is requested from the LLM in the prompt, not implemented in code.
- **Safety checks**: LLM output is parsed and validated (`Action` schema, moves of 1–5 cells). In the benchmark simulator, every cell a move passes through is checked, and a run ends on the first collision. Obstacle avoidance itself is left to the LLM.
- **Movement constraints**: the prompt asks the LLM to shorten moves near obstacles and in unknown territory; the code enforces only the 1–5 cell limit.
- **Benchmarking**: 10 scenarios, each run with all three kernels, scored with deterministic DeepEval metrics.

## System Architecture

```
├── src/
│   ├── core/
│   │   ├── navigation.py      # LLMNavigationSystem: map updates, prompts, LLM call, action parsing/execution
│   │   ├── models.py          # Data models (SensorReading, Action, GridCell, RobotState, InternalMap)
│   │   └── utils.py           # Geometry, path and I/O helpers
│   ├── kernels/
│   │   ├── saamv1.py          # SAAM v1 system-prompt prefix (spatial constraint satisfaction)
│   │   ├── saamv2.py          # SAAM v2 system-prompt prefix (cognitive architecture)
│   │   └── native.py          # Empty prefix: plain LLM planner
│   └── benchmarks/
│       ├── environment.py     # Grid simulator: ray-cast range sensors, bounds/obstacle checks
│       ├── scenarios.py       # The 10 benchmark scenarios
│       ├── metrics.py         # DeepEval metrics (deterministic, no LLM judge)
│       └── runner.py          # Runs every scenario with every kernel and writes results
├── tests/                     # Offline pytest suite (LLM calls are mocked)
├── requirements.txt           # Runtime dependencies
├── requirements-dev.txt       # Runtime dependencies + pytest
└── .github/workflows/tests.yml
```

Coordinates are Cartesian `(x, y)`: the robot starts at `(0, 0)` facing `UP` (+y); `RIGHT` is +x. Each grid cell is 30 cm and the sensors reach 300 cm (10 cells).

## Installation

Requires Python 3.11 or newer (CI runs on 3.11; also tested locally on 3.12).

```bash
git clone https://github.com/oussemaAbdaoui/cognitive-navigation.git
cd cognitive-navigation
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # or requirements-dev.txt to also run the tests
```

Running the navigation system against a real LLM needs a Groq API key:

```bash
export GROQ_API_KEY="your_api_key_here"   # Windows PowerShell: $env:GROQ_API_KEY="..."
```

The tests do not need a key.

## Quickstart

Run from the repository root (the code is imported as the `src` package). This example drives the robot in the simulator, one LLM call per step:

```python
import asyncio

from src.benchmarks.environment import NavigationEnvironment
from src.core.navigation import LLMNavigationSystem


async def main():
    env = NavigationEnvironment(obstacles=[(0, 2)], bounds=(-10, 10, -10, 10))
    nav = LLMNavigationSystem(target_position=(0, 4), kernel="saamv2")  # reads GROQ_API_KEY

    for _ in range(20):
        if nav.is_goal_reached():
            break
        sensors = env.get_sensor_readings(nav.robot_state.position, nav.robot_state.facing)
        action = await nav.navigate_step(sensors)  # one LLM call
        if action is None:
            break
        nav.execute_action(action)
        if not env.is_valid_position(nav.robot_state.position):
            print("Collision!")
            break

    print(nav.get_current_map())
    print(nav.get_robot_status())


asyncio.run(main())
```

On Windows, set `PYTHONUTF8=1` if the console cannot print the map symbols.

### Running the benchmarks

```bash
python -m src.benchmarks.runner
```

This needs `GROQ_API_KEY` and makes one API call per step: 10 scenarios × 3 kernels, up to each scenario's `max_steps` (15–50) steps per run. It writes:

- `benchmark_results/<timestamp>/<Scenario_Name>/<kernel>/result.json` and `path_visualization.png` for every run;
- `benchmark_results/run_<timestamp>/benchmark_report.json` (success rates by difficulty and kernel, failure analysis, per-run details) and `raw_results.json`.

`evaluate_with_deepeval()` is also called, but its scores are currently not written to disk.

In the benchmark, the runner pre-loads the scenario's true obstacle positions into the robot's internal map before the first step, so the LLM plans with full knowledge of the static obstacles; the sensors add the clear cells it observes.

## Run tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

The suite runs fully offline and needs no API key: network sockets are blocked and LLM calls go through `httpx.MockTransport` with scripted replies. It covers the data models, utilities, sensor-to-map updates, the motion model, progress/regression tracking, the ASCII map, LLM response parsing, the simulator, scenario definitions (including a BFS reachability check), the metrics, and the runner's collision, timeout and reporting logic. It does not measure planning quality, which depends on the LLM.

## Benchmark Scenarios

10 scenarios defined in `src/benchmarks/scenarios.py`. The last column is the 4-connected shortest path in cells (BFS over the obstacle layout, within the runner's bounds of ±10), computed from the scenario definitions.

| Scenario | Type | Difficulty | Start → Goal | Shortest path (cells) |
|----------|------|------------|--------------|----------------------|
| Simple Path | simple_path | easy | (0,0) → (0,5) | 5 |
| Single Obstacle | obstacle_maze | easy | (0,0) → (0,4) | 6 |
| Narrow Corridor | narrow_corridor | medium | (0,0) → (6,0) | 6 |
| U-Turn Challenge | u_turn | medium | (0,0) → (0,-3) | 3 |
| Dead End Escape | dead_end | hard | (0,0) → (3,0) | 3 |
| Multi-Path Choice | multi_path | medium | (0,0) → (4,0) | 8 |
| Complex Maze | complex_maze | expert | (0,0) → (5,5) | 12 |
| Open Space Efficiency | large_open_space | easy | (0,0) → (8,8) | 16 |
| Spiral Navigation | spiral_path | hard | (0,0) → (2,2) | unreachable |
| Adaptive Challenge | dynamic_obstacles | expert | (0,0) → (6,0) | 8 |

Known issues with the scenario definitions:

- **Spiral Navigation**: the goal cell is fully enclosed by obstacles, so it cannot be reached (marked as an expected failure in the tests).
- **Adaptive Challenge**: obstacles are static; nothing modifies them during a run yet.
- **U-Turn Challenge** and **Dead End Escape**: with 4-connected moves the goal can be reached in a straight line, so these layouts do not yet force a U-turn or backtracking.

## Evaluation Metrics

Computed per run by the runner and by the DeepEval metrics in `src/benchmarks/metrics.py` (these parse the run's JSON; no LLM judge is involved):

1. **Navigation success**: goal reached with no collision and within the step budget.
2. **Path efficiency**: `expected_path_length_range[0] / steps_taken`, clipped to at most 1. Steps count actions (turns included), and the lower bound is hand-set per scenario rather than computed. The runner also clips the value to at least 0.1.
3. **Collision avoidance**: 1 if no move entered an obstacle or left the bounds, else 0.
4. **Robustness**: 1 if the run ended with no timeout and no error (for example, no unparseable LLM response), else 0.

Wall-clock execution time is recorded per run and averaged per kernel in the report.

## Results

No benchmark results are included yet. Every kernel, including `native`, needs live LLM calls, so the benchmark cannot be run offline to produce a baseline; no numbers are reported here until a real run has been made.

## Configuration

```python
LLMNavigationSystem(
    target_position: Tuple[int, int],  # Goal coordinates (x, y)
    kernel: str = "native",            # "saamv1", "saamv2" or "native"
    cell_size_cm: int = 30,            # Grid cell size in cm
    max_sensor_range_cm: int = 300,    # Readings at or beyond this are treated as "nothing in range"
    api_key: Optional[str] = None,     # Defaults to the GROQ_API_KEY environment variable
)
```

## Dependencies

Python 3.11+, with `pydantic`, `httpx`, `loguru`, `numpy`, `matplotlib` (path plots) and `deepeval` (benchmark metrics). Minimum versions are in `requirements.txt`; they are the versions the test suite was run with.

## Future Work

- [ ] Add multi-sensor integration (LIDAR, cameras)
- [ ] Implement multi-agent coordination
- [ ] Implement dynamic obstacles in the simulator
- [ ] Implement the recovery behaviours in code rather than only in the prompt
- [ ] Add a non-LLM baseline planner (e.g. A* or wall-following) for comparison
- [ ] Develop ROS integration package
- [ ] Add reinforcement learning components
