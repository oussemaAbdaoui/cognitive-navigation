# Cognitive Navigation System with LLM-Based Planning


## Overview

This project implements an advanced robotic navigation system that combines classical spatial reasoning with Large Language Model (LLM)-based decision making. The system processes real-time sensor data to build an internal map while leveraging LLMs for high-level path planning with sophisticated cognitive capabilities.

## Key Features

- **Hybrid AI Architecture**: Combines grid-based mapping with LLM-based planning
- **Multiple Reasoning Kernels**: Supports SAAMv1, SAAMv2, and native navigation strategies
- **Advanced Spatial Reasoning**: Maintains detailed internal map with confidence scoring
- **Progress Monitoring**: Tracks distance to goal and detects regressions
- **Safety Mechanisms**: Collision avoidance and error recovery protocols
- **Adaptive Movement**: Adjusts constraints based on environment certainty
- **Comprehensive Benchmarking**: 10+ test scenarios with quantitative evaluation

## System Architecture

```
├── core/
│   ├── navigation.py      # Main navigation controller
│   ├── models.py          # Data models and schemas
│   ├── environment.py     # Simulation environment
│   └── utils.py           # Utility functions
├── kernels/
│   ├── saamv1.py          # Spatial constraint satisfaction kernel
│   ├── saamv2.py          # Cognitive architecture kernel  
│   └── native.py          # Baseline implementation
├── benchmarks/
│   ├── runner.py          # Benchmark execution
│   ├── scenarios.py       # Test scenarios
│   └── metrics.py         # Evaluation metrics
└── tests/                 # Unit tests
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cognitive-navigation.git
   cd cognitive-navigation
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment:
   ```bash
   export GROQ_API_KEY="your_api_key_here"
   export ACTION_DELAY=0.5  # Optional: Set action delay in seconds
   ```

## Usage

### Basic Navigation

```python
from core.navigation import LLMNavigationSystem
from core.models import SensorReading

# Initialize system with target position
nav = LLMNavigationSystem(
    target_position=(5, 5),
    kernel="saamv2",  # Try "saamv1" or "native"
    cell_size_cm=30
)

# Navigation loop
while not nav.is_goal_reached():
    # Get sensor readings (simulated or real)
    sensors = SensorReading(front=120.0, left=80.0, right=200.0)
    
    # Get next action
    action = await nav.navigate_step(sensors)
    
    # Execute action
    nav.execute_action(action)
    
    # View current status
    print(nav.get_current_map())
```

### Running Benchmarks

```bash
python -m benchmarks.runner
```

This executes all benchmark scenarios and generates:
- Performance reports in `benchmark_results/`
- Path visualizations
- Comparative analysis between kernels

## Benchmark Scenarios

10 diverse test environments:

| Scenario | Difficulty | Description |
|----------|------------|-------------|
| Simple Path | Easy | Straight-line navigation |
| Obstacle Maze | Medium | Complex obstacle arrangement |
| Narrow Corridor | Medium | Tight passage navigation |  
| U-Turn | Medium | Requires 180-degree turn |
| Dead End | Hard | Needs backtracking |
| Multi-Path | Medium | Multiple route options |
| Dynamic Obstacles | Expert | Changing environment |
| Large Open Space | Easy | Efficiency test |
| Spiral Path | Hard | Continuous turning |
| Complex Maze | Expert | Advanced challenge |

## Evaluation Metrics

1. **Navigation Success**: Goal reached without errors
2. **Path Efficiency**: Optimal vs actual path length ratio  
3. **Collision Avoidance**: Obstacle detection reliability
4. **Robustness**: Handling of errors and timeouts
5. **Time Efficiency**: Performance considering action delays

## Results Output

Example benchmark structure:
```
benchmark_results/
├── 20240515_143000/
│   ├── Simple_Path/
│   │   ├── saamv1/
│   │   │   ├── result.json
│   │   │   └── path_visualization.png
│   ├── benchmark_report.json
│   └── raw_results.json
```

Reports include:
- Success rates by scenario and kernel
- Path efficiency comparisons
- Execution time analysis
- Performance by difficulty level

## Configuration

Key parameters for `LLMNavigationSystem`:

```python
LLMNavigationSystem(
    target_position: Tuple[int, int],  # Goal coordinates
    kernel: str = "native",           # "saamv1", "saamv2", or "native"
    cell_size_cm: int = 30,           # Grid cell size in cm
    max_sensor_range_cm: int = 300    # Maximum sensor range
)
```

## Dependencies

- Python 3.8+
- Required packages:
  - pydantic
  - httpx
  - loguru
  - numpy
  - matplotlib (for visualization)
  - deepeval (for benchmarking)

## Future Work

- [ ] Add multi-sensor integration (LIDAR, cameras)
- [ ] Implement multi-agent coordination
- [ ] Enhance dynamic obstacle handling
- [ ] Develop ROS integration package
- [ ] Add reinforcement learning components
