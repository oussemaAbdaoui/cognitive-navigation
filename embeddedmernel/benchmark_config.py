#!/usr/bin/env python3
# benchmark_config.py
from typing import List, Tuple, Literal
from pydantic import conint, validate_arguments

class BenchmarkScenario:
    @validate_arguments
    def __init__(
        self,
        name: str,
        grid_size: Tuple[conint(ge=1), conint(ge=1)],
        start_pos: Tuple[conint(ge=0), conint(ge=0)],
        start_facing: Literal["UP", "DOWN", "LEFT", "RIGHT"],
        goal_pos: Tuple[conint(ge=0), conint(ge=0)],
        obstacles: List[Tuple[conint(ge=0), conint(ge=0)]],
        sensor_error_range: Tuple[float, float] = (0.0, 0.0),
        complexity: Literal["simple", "moderate", "complex"] = "simple",
        optimal_steps: int = 0  # Add optimal path length for efficiency metrics
    ):
        self.name = name
        self.grid_size = grid_size
        self.start_pos = start_pos
        self.start_facing = start_facing
        self.goal_pos = goal_pos
        self.obstacles = obstacles
        self.sensor_error_range = sensor_error_range
        self.complexity = complexity
        self.optimal_steps = optimal_steps
        self._validate_position(start_pos, "Start position")
        self._validate_position(goal_pos, "Goal position")
        for i, o in enumerate(obstacles):
            self._validate_position(o, f"Obstacle {i}")

    def _validate_position(self, pos: Tuple[int, int], label: str):
        r, c = pos
        if not (0 <= r < self.grid_size[0]) or not (0 <= c < self.grid_size[1]):
            raise ValueError(f"{label} {pos} is out of bounds for grid {self.grid_size}")

# Enhanced LLM-Centric Benchmark Scenarios
BENCHMARK_SCENARIOS = [
    # Existing scenarios with added optimal_steps
    BenchmarkScenario(
        name="Obstacle Hallway Inference",
        grid_size=(5, 5),
        start_pos=(4, 0),
        start_facing="UP",
        goal_pos=(0, 4),
        obstacles=[(i, 1) for i in range(5) if i != 3] + [(3, 2), (2, 3), (1, 4)],
        complexity="moderate",
        optimal_steps=7  # Updated with optimal path length
    ),
    BenchmarkScenario(
        name="Sensor Error Correction Test",
        grid_size=(5, 5),
        start_pos=(4, 0),
        start_facing="UP",
        goal_pos=(0, 4),
        obstacles=[(2, 2)],
        sensor_error_range=(-25.0, 25.0),
        complexity="moderate",
        optimal_steps=6
    ),
    BenchmarkScenario(
        name="False Shortcut Trap",
        grid_size=(7, 7),
        start_pos=(6, 0),
        start_facing="UP",
        goal_pos=(0, 6),
        obstacles=[(i, 1) for i in range(7) if i != 3] + [(3, 2), (3, 3)],
        complexity="complex",
        optimal_steps=10
    ),
    BenchmarkScenario(
        name="LLM Decision Fork",
        grid_size=(8, 8),
        start_pos=(7, 0),
        start_facing="UP",
        goal_pos=(0, 7),
        obstacles=[
            (1, 1), (1, 2), (2, 2), (2, 3), (3, 3), (3, 4),
            (4, 4), (4, 5), (5, 5), (5, 6), (6, 6)
        ],
        complexity="complex",
        optimal_steps=12
    ),
    BenchmarkScenario(
        name="Recovery Loop Stress Test",
        grid_size=(6, 6),
        start_pos=(0, 0),
        start_facing="RIGHT",
        goal_pos=(5, 5),
        obstacles=[(1, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 0), (5, 1)],
        sensor_error_range=(-10.0, 10.0),
        complexity="complex",
        optimal_steps=8
    ),
    BenchmarkScenario(
        name="LLM Response Validation",
        grid_size=(5, 5),
        start_pos=(2, 0),
        start_facing="RIGHT",
        goal_pos=(2, 4),
        obstacles=[(1, 1), (2, 1), (3, 1), (1, 3), (3, 3)],
        complexity="simple",
        optimal_steps=4
    ),

    # New scenarios designed for module testing
    BenchmarkScenario(
        name="Perception Edge Case",
        grid_size=(5, 5),
        start_pos=(0, 0),
        start_facing="RIGHT",
        goal_pos=(4, 4),
        obstacles=[(0, 1), (1, 0), (1, 1)],
        complexity="moderate",
        optimal_steps=6,
        sensor_error_range=(-15.0, 15.0)
    ),
    BenchmarkScenario(
        name="Waypoint Selection Challenge",
        grid_size=(6, 6),
        start_pos=(5, 0),
        start_facing="UP",
        goal_pos=(0, 5),
        obstacles=[
            (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
            (1, 3), (2, 3), (3, 3), (4, 3), (5, 3),
            (2, 0), (2, 2), (2, 4), (2, 5)
        ],
        complexity="complex",
        optimal_steps=9
    ),
    BenchmarkScenario(
        name="Action Planning Precision",
        grid_size=(4, 4),
        start_pos=(3, 0),
        start_facing="UP",
        goal_pos=(0, 3),
        obstacles=[(1, 1), (1, 2), (2, 1), (2, 3), (3, 2)],
        complexity="moderate",
        optimal_steps=5
    ),
    BenchmarkScenario(
        name="Sensor Fusion Test",
        grid_size=(7, 7),
        start_pos=(6, 3),
        start_facing="UP",
        goal_pos=(0, 3),
        obstacles=[
            (1, 2), (1, 3), (1, 4),
            (3, 1), (3, 2), (3, 4), (3, 5),
            (5, 2), (5, 3), (5, 4)
        ],
        complexity="complex",
        optimal_steps=8,
        sensor_error_range=(-30.0, 30.0)
    ),
    BenchmarkScenario(
        name="Dead End Recovery",
        grid_size=(5, 5),
        start_pos=(2, 0),
        start_facing="RIGHT",
        goal_pos=(2, 4),
        obstacles=[(1,1), (2,1), (3,1), (1,3), (3,3), (2,3)],
        complexity="complex",
        optimal_steps=7
    ),
    BenchmarkScenario(
        name="Optimal Path Challenge",
        grid_size=(8, 8),
        start_pos=(0, 0),
        start_facing="RIGHT",
        goal_pos=(7, 7),
        obstacles=[
            (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1),
            (1, 3), (2, 3), (3, 3), (4, 3), (5, 3), (6, 3), (7, 3),
            (0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (6, 5),
            (1, 7), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7)
        ],
        complexity="complex",
        optimal_steps=14
    ),
    BenchmarkScenario(
        name="Minimal Sensor Input",
        grid_size=(3, 3),
        start_pos=(0, 0),
        start_facing="RIGHT",
        goal_pos=(2, 2),
        obstacles=[(0, 1), (1, 0)],
        complexity="simple",
        optimal_steps=3
    ),
    BenchmarkScenario(
        name="Complex Maze Navigation",
        grid_size=(10, 10),
        start_pos=(0, 0),
        start_facing="RIGHT",
        goal_pos=(9, 9),
        obstacles=[
            (0, 1), (0, 2), (0, 4), (0, 5), (0, 6), (0, 8),
            (1, 1), (1, 3), (1, 5), (1, 7), (1, 8),
            (2, 3), (2, 5), (2, 7), (2, 9),
            (3, 0), (3, 1), (3, 3), (3, 4), (3, 5), (3, 7), (3, 8),
            (4, 1), (4, 2), (4, 6), (4, 9),
            (5, 0), (5, 2), (5, 3), (5, 4), (5, 6), (5, 8), (5, 9),
            (6, 1), (6, 3), (6, 5), (6, 7),
            (7, 0), (7, 2), (7, 4), (7, 6), (7, 8), (7, 9),
            (8, 1), (8, 3), (8, 5), (8, 7),
            (9, 0), (9, 2), (9, 4), (9, 6), (9, 8)
        ],
        complexity="complex",
        optimal_steps=18
    ),BenchmarkScenario(
    name="Sensor Degradation Challenge",
    grid_size=(6, 6),
    start_pos=(5, 0),
    start_facing="UP",
    goal_pos=(0, 5),
    obstacles=[(1, 1), (2, 2), (3, 3), (4, 4), (1, 4), (4, 1)],
    sensor_error_range=(-50.0, 50.0),
    complexity="complex",
    optimal_steps=7
),
BenchmarkScenario(
    name="Narrow Corridor Navigation",
    grid_size=(8, 3),
    start_pos=(7, 0),
    start_facing="UP",
    goal_pos=(0, 2),
    obstacles=[(i, 1) for i in range(8)],
    complexity="moderate",
    optimal_steps=9
),
BenchmarkScenario(
    name="U-Trap Recovery Test",
    grid_size=(5, 5),
    start_pos=(4, 0),
    start_facing="UP",
    goal_pos=(0, 4),
    obstacles=[(0,0), (0,1), (0,2), (0,3), (1,3), (2,3), (3,3), (4,3), (4,4)],
    complexity="complex",
    optimal_steps=8
),
BenchmarkScenario(
    name="Diagonal Preference Challenge",
    grid_size=(7, 7),
    start_pos=(6, 0),
    start_facing="UP",
    goal_pos=(0, 6),
    obstacles=[(1,1), (2,2), (3,3), (4,4), (5,5)],
    complexity="moderate",
    optimal_steps=6
),
BenchmarkScenario(
    name="Circular Maze",
    grid_size=(9, 9),
    start_pos=(8, 4),
    start_facing="UP",
    goal_pos=(0, 4),
    obstacles=[
        (1,1), (1,2), (1,3), (1,4), (1,5), (1,6), (1,7),
        (3,1), (3,7),
        (4,1), (4,7),
        (5,1), (5,7),
        (7,1), (7,2), (7,3), (7,4), (7,5), (7,6), (7,7),
        (2,3), (2,5), (3,3), (3,5), (4,3), (4,5), (5,3), (5,5), (6,3), (6,5)
    ],
    complexity="complex",
    optimal_steps=16
),
BenchmarkScenario(
    name="Minimal Information",
    grid_size=(10, 10),
    start_pos=(0, 0),
    start_facing="RIGHT",
    goal_pos=(9, 9),
    obstacles=[(5,5)],
    sensor_error_range=(-40.0, 40.0),
    complexity="moderate",
    optimal_steps=18
),
BenchmarkScenario(
    name="False Path Trap",
    grid_size=(6, 6),
    start_pos=(5, 0),
    start_facing="UP",
    goal_pos=(0, 5),
    obstacles=[
        (1,0), (1,1), (1,2), (1,3), (1,4),
        (3,1), (3,2), (3,3), (3,4), (3,5),
        (5,1), (5,2), (5,3), (5,4)
    ],
    complexity="complex",
    optimal_steps=10
),
BenchmarkScenario(
    name="Dynamic Obstacle Inference",
    grid_size=(5, 5),
    start_pos=(4, 2),
    start_facing="UP",
    goal_pos=(0, 2),
    obstacles=[(2,1), (2,2), (2,3), (1,0), (1,4), (3,0), (3,4)],
    sensor_error_range=(-25.0, 25.0),
    complexity="moderate",
    optimal_steps=5
),
BenchmarkScenario(
    name="Perimeter Navigation",
    grid_size=(7, 7),
    start_pos=(6, 0),
    start_facing="UP",
    goal_pos=(0, 6),
    obstacles=[(i, j) for i in range(1,6) for j in range(1,6)],
    complexity="complex",
    optimal_steps=12
),
BenchmarkScenario(
    name="Zigzag Path",
    grid_size=(8, 8),
    start_pos=(7, 0),
    start_facing="UP",
    goal_pos=(0, 7),
    obstacles=[
        (1,0), (1,1), (1,2), (1,3), (1,4), (1,5), (1,6),
        (3,1), (3,2), (3,3), (3,4), (3,5), (3,6), (3,7),
        (5,0), (5,1), (5,2), (5,3), (5,4), (5,5), (5,6),
        (7,1), (7,2), (7,3), (7,4), (7,5), (7,6), (7,7)
    ],
    complexity="complex",
    optimal_steps=14
),
BenchmarkScenario(
    name="Central Obstacle Field",
    grid_size=(6, 6),
    start_pos=(5, 0),
    start_facing="UP",
    goal_pos=(0, 5),
    obstacles=[(2,2), (2,3), (3,2), (3,3), (1,1), (1,4), (4,1), (4,4)],
    complexity="moderate",
    optimal_steps=7
),
BenchmarkScenario(
    name="Long Hallway with Gaps",
    grid_size=(3, 10),
    start_pos=(0, 0),
    start_facing="RIGHT",
    goal_pos=(2, 9),
    obstacles=[(0,2), (0,3), (0,5), (0,6), (0,8),
               (2,1), (2,3), (2,4), (2,6), (2,7)],
    complexity="moderate",
    optimal_steps=9
),
BenchmarkScenario(
    name="Island Hopping",
    grid_size=(7, 7),
    start_pos=(6, 3),
    start_facing="UP",
    goal_pos=(0, 3),
    obstacles=[
        (0,0), (0,1), (0,2), (0,4), (0,5), (0,6),
        (1,0), (1,6),
        (2,0), (2,6),
        (3,0), (3,6),
        (4,0), (4,6),
        (5,0), (5,6),
        (6,0), (6,1), (6,2), (6,4), (6,5), (6,6),
        (2,2), (2,4), (4,2), (4,4)
    ],
    complexity="complex",
    optimal_steps=12
),
BenchmarkScenario(
    name="Spiral Maze",
    grid_size=(9, 9),
    start_pos=(4, 4),
    start_facing="UP",
    goal_pos=(0, 0),
    obstacles=[
        (1,1), (1,2), (1,3), (1,4), (1,5), (1,6), (1,7),
        (2,1), (2,7),
        (3,1), (3,7),
        (4,1), (4,3), (4,5), (4,7),
        (5,1), (5,7),
        (6,1), (6,7),
        (7,1), (7,2), (7,3), (7,4), (7,5), (7,6), (7,7),
        (3,3), (3,5), (5,3), (5,5)
    ],
    complexity="complex",
    optimal_steps=15
),
BenchmarkScenario(
    name="Meta-Cognition Stress Test",
    grid_size=(5, 5),
    start_pos=(4, 0),
    start_facing="UP",
    goal_pos=(0, 4),
    obstacles=[(3,0), (3,1), (2,1), (1,1), (1,2), (1,3), (2,3), (3,3), (3,4)],
    sensor_error_range=(-35.0, 35.0),
    complexity="complex",
    optimal_steps=9
)
]        # 1. Simple straight path
        BenchmarkScenario(
            name="Simple Path",
            scenario_type=ScenarioType.SIMPLE_PATH,
            start_position=(0, 0),
            goal_position=(0, 5),
            obstacles=[],
            expected_path_length_range=(5, 8),
            max_steps=15,
            description="Direct path with no obstacles",
            difficulty="easy"
        ),

        # 2. Single obstacle avoidance
        BenchmarkScenario(
            name="Single Obstacle",
            scenario_type=ScenarioType.OBSTACLE_MAZE,
            start_position=(0, 0),
            goal_position=(0, 4),
            obstacles=[(0, 2)],
            expected_path_length_range=(6, 10),
            max_steps=20,
            description="Navigate around a single obstacle",
            difficulty="easy"
        ),
        # 3. Narrow corridor
        BenchmarkScenario(
            name="Narrow Corridor",
            scenario_type=ScenarioType.NARROW_CORRIDOR,
            start_position=(0, 0),
            goal_position=(6, 0),
            obstacles=[(1, -1), (1, 1), (2, -1), (2, 1), (3, -1), (3, 1),
                      (4, -1), (4, 1), (5, -1), (5, 1)],
            expected_path_length_range=(6, 10),
            max_steps=25,
            description="Navigate through a narrow corridor",
            difficulty="medium"
        ),

# 3. Narrow corridor
        BenchmarkScenario(
            name="Narrow Corridor",
            scenario_type=ScenarioType.NARROW_CORRIDOR,
            start_position=(0, 0),
            goal_position=(6, 0),
            obstacles=[(1, -1), (1, 1), (2, -1), (2, 1), (3, -1), (3, 1),
                      (4, -1), (4, 1), (5, -1), (5, 1)],
            expected_path_length_range=(6, 10),
            max_steps=25,
            description="Navigate through a narrow corridor",
            difficulty="medium"
        ),

        # 4. U-turn scenario
        BenchmarkScenario(
            name="U-Turn Challenge",
            scenario_type=ScenarioType.U_TURN,
            start_position=(0, 0),
            goal_position=(0, -3),
            obstacles=[(1, 0), (1, -1), (1, -2), (1, -3)],
            expected_path_length_range=(8, 15),
            max_steps=30,
            description="Requires U-turn to reach goal",
            difficulty="medium"
        ),

        # 5. Dead end with backtracking
        BenchmarkScenario(
            name="Dead End Escape",
            scenario_type=ScenarioType.DEAD_END,
            start_position=(0, 0),
            goal_position=(3, 0),
            obstacles=[(1, 1), (2, 1), (1, -1), (2, -1), (4, 0)],
            expected_path_length_range=(10, 20),
            max_steps=40,
            description="Must escape dead end to reach goal",
            difficulty="hard"
        ),
        BenchmarkScenario(
            name="Multi-Path Choice",
            scenario_type=ScenarioType.MULTI_PATH,
            start_position=(0, 0),
            goal_position=(4, 0),
            obstacles=[(2, 0), (1, 1), (3, 1), (1, -1), (3, -1)],
            expected_path_length_range=(6, 12),
            max_steps=25,
            description="Multiple viable paths to goal",
            difficulty="medium"
        ),

        # 7. Complex maze
        BenchmarkScenario(
            name="Complex Maze",
            scenario_type=ScenarioType.COMPLEX_MAZE,
            start_position=(0, 0),
            goal_position=(5, 5),
            obstacles=[
                (1, 1), (1, 2), (1, 3), (2, 3), (3, 3), (3, 2), (3, 1),
                (4, 1), (2, 0), (4, 4), (3, 4), (2, 4), (1, 4), (0, 3)
            ],
            expected_path_length_range=(15, 30),
            max_steps=50,
            description="Complex maze requiring strategic navigation",
            difficulty="expert"
        ),

        # 8. Large open space (efficiency test)
        BenchmarkScenario(
            name="Open Space Efficiency",
            scenario_type=ScenarioType.LARGE_OPEN_SPACE,
            start_position=(0, 0),
            goal_position=(8, 8),
            obstacles=[],
            expected_path_length_range=(16, 25),
            max_steps=40,
            description="Test efficiency in open space",
            difficulty="easy"
        ),

        # 9. Spiral pattern
        BenchmarkScenario(
            name="Spiral Navigation",
            scenario_type=ScenarioType.SPIRAL_PATH,
            start_position=(0, 0),
            goal_position=(2, 2),
            obstacles=[
                (1, 0), (1, 1), (0, 1), (2, 1), (2, 0), (3, 0), (3, 1),
                (3, 2), (3, 3), (2, 3), (1, 3), (0, 3), (0, 2)
            ],
            expected_path_length_range=(20, 35),
            max_steps=50,
            description="Spiral path requiring careful navigation",
            difficulty="hard"
        ),

        # 10. Dynamic obstacle simulation
        BenchmarkScenario(
            name="Adaptive Challenge",
            scenario_type=ScenarioType.DYNAMIC_OBSTACLES,
            start_position=(0, 0),
            goal_position=(6, 0),
            obstacles=[(2, 0), (4, 0)],  # Will be modified during execution
            expected_path_length_range=(10, 20),
            max_steps=35,
            description="Adaptive navigation with changing obstacles",
            difficulty="expert"
        )