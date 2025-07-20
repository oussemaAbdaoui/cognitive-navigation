#!/usr/bin/env python3
# step_by_step_benchmark.py
from benchmark_config import BENCHMARK_SCENARIOS
from langgraph_nodes import StateManager, WaypointPlanner, ActionPlanner
from test_metrics import PerceptionAccuracyMetric, WaypointOptimalityMetric, ActionCorrectnessMetric
from typing import List, Dict, Tuple, Any, Optional
import time
import json
import random
import numpy as np
from pydantic import BaseModel, Field
from deepeval.test_case import LLMTestCase

# --------------- Data Models ---------------
class TestStep(BaseModel):
    step: int
    robot_pos: Tuple[int, int]
    robot_facing: str
    sensor_data: Dict[str, float]
    expected_grid: str
    actual_grid: str = ""
    actual_waypoint: Tuple[int, int] = (-1, -1)
    actual_actions: List[Dict] = []
    perception_score: float = 0.0
    waypoint_score: float = 0.0
    action_score: float = 0.0

class TestCase(BaseModel):
    scenario_name: str
    complexity: str
    steps: List[TestStep] = []
    avg_perception: float = 0.0
    avg_waypoint: float = 0.0
    avg_action: float = 0.0
    success: bool = False
    path_length: int = 0
    optimal_path: int = 0
    path_efficiency: float = 0.0

# --------------- Sensor Simulation ---------------
def calculate_sensor_readings(robot_pos: Tuple[int, int],
                             robot_facing: str,
                             obstacles: List[Tuple[int, int]],
                             grid_size: Tuple[int, int],
                             error_range: Tuple[float, float] = (0.0, 0.0)) -> Dict[str, float]:
    """Generate realistic sensor readings with optional error"""
    readings = {'front': 300.0, 'left': 300.0, 'right': 300.0}
    r, c = robot_pos
    rows, cols = grid_size

    # Directions relative to robot facing
    directions = {
        "UP": {"front": (-1, 0), "left": (0, -1), "right": (0, 1)},
        "DOWN": {"front": (1, 0), "left": (0, 1), "right": (0, -1)},
        "LEFT": {"front": (0, -1), "left": (1, 0), "right": (-1, 0)},
        "RIGHT": {"front": (0, 1), "left": (-1, 0), "right": (1, 0)}
    }

    for sensor, (dr, dc) in directions[robot_facing].items():
        distance = 1
        while True:
            nr, nc = r + dr * distance, c + dc * distance
            # Check if out of bounds
            if not (0 <= nr < rows and 0 <= nc < cols):
                break
            # Check for obstacle
            if (nr, nc) in obstacles:
                # Add sensor error if specified
                error = random.uniform(*error_range) if error_range != (0.0, 0.0) else 0.0
                readings[sensor] = max(10, distance * 30 + error)
                break
            distance += 1

    return readings

# --------------- Step-by-Step Benchmark ---------------
class StepByStepBenchmark:
    def __init__(self):
        self.scenario_count = 0
        self.total_perception = 0.0
        self.total_waypoint = 0.0
        self.total_action = 0.0
        self.total_steps = 0
        self.test_cases: List[TestCase] = []

    def print_step(self, message: str):
        print(f"\n[STEP {time.strftime('%H:%M:%S')}] {message}")
        print("-" * 60)

    def print_evaluation(self, metric_name: str, score: float, threshold: float):
        result = "PASS" if score >= threshold else "FAIL"
        color = "\033[92m" if result == "PASS" else "\033[91m"
        print(f"{color}{metric_name}: {score:.2f} (Threshold: {threshold}) {result}\033[0m")

    def get_expected_grid(self, state_manager: StateManager,
                         sensor_data: Dict[str, float]) -> str:
        """Generate expected grid based on sensor readings"""
        grid = [row[:] for row in state_manager.obstacle_grid]
        r, c = state_manager.robot_position
        rows, cols = state_manager.grid_rows, state_manager.grid_cols

        # Clear non-permanent markers (keep obstacles and goal)
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] not in ['■', 'G']:
                    grid[i][j] = '·'

        # Place robot
        grid[r][c] = state_manager._get_direction_arrow()

        # Place goal if not at robot position
        gr, gc = state_manager.goal_position
        if (gr, gc) != (r, c):
            grid[gr][gc] = 'G'

        # Add obstacles based on sensor readings
        for sensor, distance in sensor_data.items():
            if distance < 300:  # Obstacle detection threshold
                if sensor == "front":
                    if state_manager.robot_facing == "UP" and r > 0:
                        grid[r-1][c] = '■'
                    elif state_manager.robot_facing == "DOWN" and r < rows-1:
                        grid[r+1][c] = '■'
                    elif state_manager.robot_facing == "LEFT" and c > 0:
                        grid[r][c-1] = '■'
                    elif state_manager.robot_facing == "RIGHT" and c < cols-1:
                        grid[r][c+1] = '■'
                elif sensor == "left":
                    if state_manager.robot_facing == "UP" and c > 0:
                        grid[r][c-1] = '■'
                    elif state_manager.robot_facing == "DOWN" and c < cols-1:
                        grid[r][c+1] = '■'
                    elif state_manager.robot_facing == "LEFT" and r < rows-1:
                        grid[r+1][c] = '■'
                    elif state_manager.robot_facing == "RIGHT" and r > 0:
                        grid[r-1][c] = '■'
                elif sensor == "right":
                    if state_manager.robot_facing == "UP" and c < cols-1:
                        grid[r][c+1] = '■'
                    elif state_manager.robot_facing == "DOWN" and c > 0:
                        grid[r][c-1] = '■'
                    elif state_manager.robot_facing == "LEFT" and r > 0:
                        grid[r-1][c] = '■'
                    elif state_manager.robot_facing == "RIGHT" and r < rows-1:
                        grid[r+1][c] = '■'

        return "\n".join(" ".join(row) for row in grid)

    def simulate_actions(self, actions: List[Dict],
                        start_pos: Tuple[int, int],
                        start_facing: str,
                        grid_size: Tuple[int, int],
                        obstacles: List[Tuple[int, int]]) -> Tuple[Tuple[int, int], str]:
        """Simulate action execution and return new position and facing"""
        pos = list(start_pos)
        facing = start_facing
        rows, cols = grid_size
        obstacle_set = set(obstacles)

        for action in actions:
            if action['type'] == "TURN":
                if action['direction'] == "RIGHT":
                    # Turn right logic
                    if facing == "UP": facing = "RIGHT"
                    elif facing == "RIGHT": facing = "DOWN"
                    elif facing == "DOWN": facing = "LEFT"
                    elif facing == "LEFT": facing = "UP"
                elif action['direction'] == "LEFT":
                    # Turn left logic
                    if facing == "UP": facing = "LEFT"
                    elif facing == "LEFT": facing = "DOWN"
                    elif facing == "DOWN": facing = "RIGHT"
                    elif facing == "RIGHT": facing = "UP"
                elif action['direction'] in ["UP", "DOWN"]:
                    facing = action['direction']
            elif action['type'] == "MOVE" and action['direction'] == "FORWARD":
                cells = action['cells']
                dr, dc = 0, 0
                if facing == "UP": dr = -1
                elif facing == "DOWN": dr = 1
                elif facing == "LEFT": dc = -1
                elif facing == "RIGHT": dc = 1

                for _ in range(cells):
                    new_pos = [pos[0] + dr, pos[1] + dc]
                    # Check boundaries
                    if not (0 <= new_pos[0] < rows and 0 <= new_pos[1] < cols):
                        raise ValueError(f"Move out of bounds: {new_pos}")
                    # Check obstacle
                    if tuple(new_pos) in obstacle_set:
                        raise ValueError(f"Hit obstacle at {new_pos}")
                    pos = new_pos

        return tuple(pos), facing

    def run_test_case(self, scenario) -> TestCase:
        """Execute a full test case with step-by-step evaluation"""
        test_case = TestCase(
            scenario_name=scenario.name,
            complexity=scenario.complexity
        )

        # Initialize components
        state_manager = StateManager(
            target=scenario.goal_pos,
            grid_size=scenario.grid_size,
            start_position=scenario.start_pos,
            start_facing=scenario.start_facing,
            initial_obstacles=scenario.obstacles
        )
        waypoint_planner = WaypointPlanner()
        action_planner = ActionPlanner(
            cm_per_cell=30,
            grid_size=scenario.grid_size,
            obstacle_grid=state_manager.obstacle_grid
        )

        # Set initial state
        robot_pos = list(scenario.start_pos)
        robot_facing = scenario.start_facing
        step_count = 0
        max_steps = 50

        # Get optimal path length
        if hasattr(scenario, 'optimal_steps') and scenario.optimal_steps > 0:
            test_case.optimal_path = scenario.optimal_steps
        else:
            # Calculate if not provided
            test_case.optimal_path = self.calculate_optimal_path(
                tuple(scenario.start_pos),
                tuple(scenario.goal_pos),
                scenario.obstacles,
                scenario.grid_size
            )

        while step_count < max_steps:
            step_count += 1
            self.total_steps += 1
            print(f"\n\033[1m>>> Step {step_count} | Robot at {robot_pos} facing {robot_facing}\033[0m")

            # Generate sensor data
            sensor_data = calculate_sensor_readings(
                robot_pos=tuple(robot_pos),
                robot_facing=robot_facing,
                obstacles=scenario.obstacles,
                grid_size=scenario.grid_size,
                error_range=scenario.sensor_error_range
            )
            print(f"Sensor Readings: Front={sensor_data['front']:.1f}cm, "
                  f"Left={sensor_data['left']:.1f}cm, Right={sensor_data['right']:.1f}cm")

            # Create test step
            test_step = TestStep(
                step=step_count,
                robot_pos=tuple(robot_pos),
                robot_facing=robot_facing,
                sensor_data=sensor_data,
                expected_grid=self.get_expected_grid(state_manager, sensor_data)
            )

            # Test Perception
            self.print_step("Testing Perception System")
            print("Expected Grid:")
            print(test_step.expected_grid)

            # Update state with sensor data
            perception_actual = state_manager.process_sensor_data(sensor_data)
            test_step.actual_grid = perception_actual
            print("Actual Grid:")
            print(perception_actual)

            # Evaluate perception
            perception_metric = PerceptionAccuracyMetric(expected_grid=test_step.expected_grid)
            perception_test = LLMTestCase(
                input=json.dumps(sensor_data),
                actual_output=perception_actual,
                expected_output=test_step.expected_grid
            )
            test_step.perception_score = perception_metric.measure(perception_test)
            self.print_evaluation("Perception Accuracy", test_step.perception_score, 0.8)

            # Test Waypoint Planning
            self.print_step("Testing Waypoint Planning")
            visual_grid = state_manager.get_visual_grid()
            print("Current Grid:")
            print(visual_grid)

            waypoint = waypoint_planner.plan_waypoint(
                visual_grid,
                tuple(robot_pos),
                tuple(scenario.goal_pos)
            )
            test_step.actual_waypoint = waypoint or (-1, -1)
            print(f"Selected Waypoint: {test_step.actual_waypoint}")

            # Evaluate waypoint
            waypoint_test = LLMTestCase(
                input=visual_grid,
                actual_output=str(waypoint) if waypoint else "None"
            )
            waypoint_metric = WaypointOptimalityMetric(
                robot_pos=tuple(robot_pos),
                goal_pos=tuple(scenario.goal_pos),
                grid_size=scenario.grid_size,
                obstacle_grid=state_manager.obstacle_grid
            )
            test_step.waypoint_score = waypoint_metric.measure(waypoint_test)
            self.print_evaluation("Waypoint Optimality", test_step.waypoint_score, 0.7)

            # Test Action Planning
            self.print_step("Testing Action Planning")
            waypoint_target = waypoint or tuple(scenario.goal_pos)
            print(f"Navigating to: {waypoint_target}")

            actions = action_planner.plan_actions(
                waypoint_target,
                tuple(robot_pos),
                robot_facing
            ) or []
            test_step.actual_actions = [a.dict() for a in actions] if actions else []
            print(f"Generated Actions: {test_step.actual_actions}")

            # Evaluate actions
            action_test = LLMTestCase(
                input=f"Waypoint: {waypoint_target}",
                actual_output=str(test_step.actual_actions)
            )
            action_metric = ActionCorrectnessMetric(
                start_pos=tuple(robot_pos),
                start_facing=robot_facing,
                waypoint=waypoint_target,
                grid_size=scenario.grid_size,
                obstacle_grid=state_manager.obstacle_grid
            )
            test_step.action_score = action_metric.measure(action_test)
            self.print_evaluation("Action Correctness", test_step.action_score, 0.9)

            # Add step to test case
            test_case.steps.append(test_step)

            # Check for termination
            if not actions or tuple(robot_pos) == tuple(scenario.goal_pos):
                break

            # Simulate action execution
            try:
                new_pos, new_facing = self.simulate_actions(
                    test_step.actual_actions,
                    tuple(robot_pos),
                    robot_facing,
                    scenario.grid_size,
                    scenario.obstacles
                )
                robot_pos = list(new_pos)
                robot_facing = new_facing

                # Update state manager position
                state_manager.robot_position = robot_pos
                state_manager.robot_facing = robot_facing

                # Update grid visualization
                state_manager.obstacle_grid[robot_pos[0]][robot_pos[1]] = (
                    state_manager._get_direction_arrow()
                )

                # Check if goal reached
                if tuple(robot_pos) == tuple(scenario.goal_pos):
                    test_case.success = True
                    print("\033[92mGoal reached!\033[0m")
                    break
            except Exception as e:
                print(f"\033[91mAction simulation failed: {str(e)}\033[0m")
                break

        # Calculate test case metrics
        test_case.path_length = step_count
        if test_case.optimal_path > 0:
            test_case.path_efficiency = min(1.0, test_case.optimal_path / test_case.path_length)

        if test_case.steps:
            test_case.avg_perception = sum(s.perception_score for s in test_case.steps) / len(test_case.steps)
            test_case.avg_waypoint = sum(s.waypoint_score for s in test_case.steps) / len(test_case.steps)
            test_case.avg_action = sum(s.action_score for s in test_case.steps) / len(test_case.steps)

        return test_case

    def calculate_optimal_path(self, start: Tuple[int, int],
                              goal: Tuple[int, int],
                              obstacles: List[Tuple[int, int]],
                              grid_size: Tuple[int, int]) -> int:
        """Calculate optimal path length using Manhattan distance"""
        # For complex paths, we'll just use Manhattan distance as baseline
        return abs(start[0] - goal[0]) + abs(start[1] - goal[1])

    def run_benchmarks(self):
        print("\n\033[1m" + "="*60)
        print("STARTING STEP-BY-STEP LLM NAVIGATION BENCHMARK")
        print("="*60 + "\033[0m")

        self.scenario_count = len(BENCHMARK_SCENARIOS)

        for scenario in BENCHMARK_SCENARIOS:
            print(f"\n\033[1mSCENARIO: {scenario.name} ({scenario.complexity.upper()})\033[0m")
            print(f"Grid: {scenario.grid_size[0]}x{scenario.grid_size[1]}")
            print(f"Start: {scenario.start_pos} facing {scenario.start_facing}")
            print(f"Goal: {scenario.goal_pos}")
            print(f"Obstacles: {len(scenario.obstacles)}")

            test_case = self.run_test_case(scenario)
            self.test_cases.append(test_case)

            # Update global metrics
            self.total_perception += test_case.avg_perception
            self.total_waypoint += test_case.avg_waypoint
            self.total_action += test_case.avg_action

            # Print scenario summary
            status = "\033[92mSUCCESS\033[0m" if test_case.success else "\033[91mFAIL\033[0m"
            print(f"\nScenario Result: {status}")
            print(f"Perception Accuracy: {test_case.avg_perception:.2f}")
            print(f"Waypoint Optimality: {test_case.avg_waypoint:.2f}")
            print(f"Action Correctness: {test_case.avg_action:.2f}")
            if test_case.optimal_path > 0:
                print(f"Path Efficiency: {test_case.path_efficiency:.2f} "
                      f"({test_case.path_length} steps vs optimal {test_case.optimal_path})")

        self.generate_report()

    def generate_report(self):
        print("\n\033[1m" + "="*60)
        print("BENCHMARK SUMMARY REPORT")
        print("="*60 + "\033[0m")

        # Individual scenario results
        for tc in self.test_cases:
            status = "\033[92mPASS\033[0m" if tc.success else "\033[91mFAIL\033[0m"
            print(f"\nScenario: {tc.scenario_name} ({tc.complexity}) - {status}")
            print(f"- Perception: {tc.avg_perception:.2f}")
            print(f"- Waypoint: {tc.avg_waypoint:.2f}")
            print(f"- Action: {tc.avg_action:.2f}")
            if tc.optimal_path > 0:
                print(f"- Path Efficiency: {tc.path_efficiency:.2f} ({tc.path_length}/{tc.optimal_path} steps)")

        # Averages
        if self.scenario_count > 0:
            avg_perception = self.total_perception / self.scenario_count
            avg_waypoint = self.total_waypoint / self.scenario_count
            avg_action = self.total_action / self.scenario_count

            print("\n\033[1mOVERALL AVERAGES\033[0m")
            print(f"Perception Accuracy: {avg_perception:.2f}")
            print(f"Waypoint Optimality: {avg_waypoint:.2f}")
            print(f"Action Correctness: {avg_action:.2f}")

            # Success rate
            success_count = sum(1 for tc in self.test_cases if tc.success)
            success_rate = success_count / self.scenario_count
            print(f"Success Rate: {success_rate:.2f} ({success_count}/{self.scenario_count})")

            # Final verdict
            overall_pass = (avg_perception >= 0.8 and
                           avg_waypoint >= 0.7 and
                           avg_action >= 0.9 and
                           success_rate >= 0.8)
            color = "\033[92m" if overall_pass else "\033[91m"
            status = "PASS" if overall_pass else "FAIL"
            print(f"\n{color}FINAL VERDICT: {status}\033[0m")

        # Save detailed results
        with open("benchmark_results.json", "w") as f:
            json.dump([tc.dict() for tc in self.test_cases], f, indent=2)
        print("\nDetailed results saved to benchmark_results.json")

if __name__ == "__main__":
    benchmark = StepByStepBenchmark()
    benchmark.run_benchmarks()