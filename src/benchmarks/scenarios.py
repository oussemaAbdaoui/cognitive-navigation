# scenarios.py - Benchmark scenario definitions

from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple

class ScenarioType(Enum):
    SIMPLE_PATH = "simple_path"
    OBSTACLE_MAZE = "obstacle_maze"
    NARROW_CORRIDOR = "narrow_corridor"
    U_TURN = "u_turn"
    DEAD_END = "dead_end"
    MULTI_PATH = "multi_path"
    DYNAMIC_OBSTACLES = "dynamic_obstacles"
    LARGE_OPEN_SPACE = "large_open_space"
    SPIRAL_PATH = "spiral_path"
    COMPLEX_MAZE = "complex_maze"

@dataclass
class BenchmarkScenario:
    """Single benchmark scenario configuration"""
    name: str
    scenario_type: ScenarioType
    start_position: Tuple[int, int]
    goal_position: Tuple[int, int]
    obstacles: List[Tuple[int, int]]
    expected_path_length_range: Tuple[int, int]
    max_steps: int
    description: str
    difficulty: str  # "easy", "medium", "hard", "expert"

def create_benchmark_scenarios() -> List[BenchmarkScenario]:
    """Generate all benchmark scenarios"""
    return [
        # 1. Simple straight path
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
    ]