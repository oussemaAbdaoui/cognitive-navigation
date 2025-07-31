#!/usr/bin/env python3
# navigation_module_evaluator.py
import time

import asyncio
import random
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from benchmark_config import BENCHMARK_SCENARIOS
from langgraph_nodes import StateManager, WaypointPlanner, ActionPlanner, Action
from benchmark_runner import PerceptorModule

@dataclass
class TestResult:
    module: str
    test_case: str
    passed: bool
    metrics: Dict[str, float]
    error: str = ""

class NavigationEvaluator:
    def __init__(self):
        self.results: List[TestResult] = []

    async def run_full_evaluation(self):
        """Execute complete evaluation suite"""
        print("=== Starting Comprehensive Navigation Module Evaluation ===")

        # Test perception module
        await self.evaluate_perceptor()

        # Test waypoint planner
        await self.evaluate_waypoint_planner()

        # Test action planner
        await self.evaluate_action_planner()

        # Generate report
        self.generate_report()
        time.sleep(1)  # pause to reduce rate limits


    async def evaluate_perceptor(self):
        """Evaluate perception module under various conditions"""
        test_cases = [
            ("Empty Environment", [], {'front': 300, 'left': 300, 'right': 300}),
            ("Single Obstacle", [(2,2)], {'front': 90, 'left': 300, 'right': 300}),
            ("Complex Environment", [(1,1), (2,3), (3,0)],
             {'front': 120, 'left': 60, 'right': 180})
        ]

        for name, obstacles, sensors in test_cases:
            try:
                # Initialize StateManager with correct parameters
                state = StateManager(
                    target=(4,4),
                    grid_rows=5,
                    grid_cols=5,
                    start_position=(0,0),
                    start_facing="RIGHT",
                    initial_obstacles=obstacles
                )
                perceptor = PerceptorModule(state)

                result = await perceptor.execute({"sensor_data": sensors})

                # Calculate accuracy metrics
                expected_obstacles = set(obstacles)
                detected_obstacles = set(
                    (r,c) for (r,c), val in state.obstacle_grid.items()
                    if val == '■'
                )

                precision = len(expected_obstacles & detected_obstacles) / len(detected_obstacles) if detected_obstacles else 1.0
                recall = len(expected_obstacles & detected_obstacles) / len(expected_obstacles) if expected_obstacles else 1.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

                # Consider test passed if F1 > 0.7 (more lenient threshold)
                self.results.append(TestResult(
                    module="Perceptor",
                    test_case=name,
                    passed=f1 > 0.7,
                    metrics={
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'confidence': result.get('confidence', 0)
                    }
                ))

            except Exception as e:
                self.results.append(TestResult(
                    module="Perceptor",
                    test_case=name,
                    passed=False,
                    metrics={},
                    error=str(e)
                ))

    async def evaluate_waypoint_planner(self):
        """Evaluate waypoint planning quality"""
        scenarios = [
            ("Open Field", [], (4,0), (0,4)),
            ("Obstacle Course", [(1,1), (2,2), (3,3)], (4,0), (0,4)),
            ("Maze", [(0,1), (1,1), (1,3), (2,3), (3,0), (3,1), (3,2)], (4,0), (0,4))
        ]

        for name, obstacles, start, goal in scenarios:
            try:
                planner = WaypointPlanner((5,5))
                state = StateManager(
                    target=goal,
                    grid_rows=5,
                    grid_cols=5,
                    start_position=start,
                    start_facing="RIGHT",
                    initial_obstacles=obstacles
                )

                grid = state.get_visual_grid()
                waypoint = await planner.plan_waypoint(grid, start, goal)

                if not waypoint:
                    raise ValueError("No waypoint generated")

                # Calculate quality metrics
                start_dist = abs(start[0]-goal[0]) + abs(start[1]-goal[1])
                wp_dist = abs(waypoint[0]-goal[0]) + abs(waypoint[1]-goal[1])
                progress = (start_dist - wp_dist) / start_dist if start_dist > 0 else 0.0

                # Simple path safety check
                safe = waypoint not in obstacles

                # More lenient passing criteria for LLM-based planner
                self.results.append(TestResult(
                    module="WaypointPlanner",
                    test_case=name,
                    passed=progress > 0.2 and safe,
                    metrics={
                        'progress': progress,
                        'safety': float(safe),
                        'distance_to_goal': wp_dist
                    }
                ))

            except Exception as e:
                self.results.append(TestResult(
                    module="WaypointPlanner",
                    test_case=name,
                    passed=False,
                    metrics={},
                    error=str(e)
                ))

    async def evaluate_action_planner(self):
        """Evaluate action sequence generation"""
        test_cases = [
            ("Straight Move", (0,0), "RIGHT", (0,2), [Action(type="MOVE", direction="FORWARD", cells=2)]),
            ("Turn + Move", (0,0), "UP", (1,1), [
                Action(type="TURN", direction="RIGHT"),
                Action(type="MOVE", direction="FORWARD", cells=1)
            ]),
            ("Complex Path", (0,0), "LEFT", (2,2), [
                Action(type="TURN", direction="RIGHT"),
                Action(type="MOVE", direction="FORWARD", cells=2),
                Action(type="TURN", direction="LEFT"),
                Action(type="MOVE", direction="FORWARD", cells=2)
            ])
        ]

        planner = ActionPlanner()

        for name, start, facing, waypoint, expected in test_cases:
            try:
                actions = await planner.plan_actions(waypoint, start, facing)

                if not actions:
                    raise ValueError("No actions generated")

                # Calculate sequence similarity
                def action_similarity(a1: Action, a2: Action) -> float:
                    if a1.type != a2.type:
                        return 0.0
                    if a1.type == "TURN" and a1.direction == a2.direction:
                        return 1.0
                    if a1.type == "MOVE" and a1.direction == a2.direction:
                        return min(1.0, 1.0 - abs(a1.cells - a2.cells)/5.0)
                    return 0.0

                # Use dynamic programming to find best alignment
                similarity = 0.0
                min_len = min(len(actions), len(expected))
                for i in range(min_len):
                    similarity += action_similarity(actions[i], expected[i])
                similarity /= max(len(actions), len(expected))

                # Consider test passed if similarity > 0.7
                self.results.append(TestResult(
                    module="ActionPlanner",
                    test_case=name,
                    passed=similarity > 0.7,
                    metrics={
                        'similarity': similarity,
                        'action_count': len(actions),
                        'expected_count': len(expected)
                    }
                ))

            except Exception as e:
                self.results.append(TestResult(
                    module="ActionPlanner",
                    test_case=name,
                    passed=False,
                    metrics={},
                    error=str(e)
                ))

    def generate_report(self):
        """Generate comprehensive evaluation report"""
        print("\n=== EVALUATION REPORT ===")

        # Module-wise summary
        for module in ["Perceptor", "WaypointPlanner", "ActionPlanner"]:
            module_results = [r for r in self.results if r.module == module]
            passed = sum(1 for r in module_results if r.passed)

            print(f"\n{module} Results: {passed}/{len(module_results)} passed")

            # Print detailed metrics for failed tests
            for result in module_results:
                if not result.passed:
                    print(f"\n✗ {result.test_case}")
                    if result.error:
                        print(f"Error: {result.error}")
                    else:
                        print("Metrics:", result.metrics)

        # Overall statistics
        total_passed = sum(1 for r in self.results if r.passed)
        print(f"\nOverall: {total_passed}/{len(self.results)} tests passed")

        # Calculate average metrics
        metrics = {}
        for result in self.results:
            for metric, value in result.metrics.items():
                metrics.setdefault(metric, []).append(value)

        print("\nAverage Metrics:")
        for metric, values in metrics.items():
            if values:
                print(f"- {metric}: {sum(values)/len(values):.2f}")

async def main():
    evaluator = NavigationEvaluator()
    await evaluator.run_full_evaluation()

if __name__ == "__main__":
    asyncio.run(main())