#!/usr/bin/env python3
# runner.py - Benchmark execution and reporting
from __future__ import annotations
from datetime import datetime
import shutil
import asyncio
import json
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Any, Literal
from pathlib import Path
from enum import Enum

import numpy as np
from deepeval import evaluate
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset

from ..core.navigation import LLMNavigationSystem
from ..core.models import SensorReading
from .scenarios import BenchmarkScenario, ScenarioType
from .environment import NavigationEnvironment
from .metrics import (
    NavigationSuccessMetric,
    PathEfficiencyMetric,
    CollisionAvoidanceMetric,
    RobustnessMetric
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_CELL_SIZE_CM = 30
MAX_API_RETRIES = 3
API_RETRY_DELAY = 0.5
DEFAULT_BOUNDS = (-10, 10, -10, 10)
MIN_PATH_EFFICIENCY = 0.1
MAX_EXECUTION_TIME = 60.0  # seconds

KernelType = Literal["saamv1", "saamv2", "native"]
DifficultyType = Literal["easy", "medium", "hard", "expert"]

@dataclass
class NavigationResult:
    """Results from a single navigation scenario run"""
    scenario_name: str
    kernel: KernelType
    success: bool
    steps_taken: int
    path_efficiency: float
    goal_reached: bool
    timeout: bool
    collision_occurred: bool
    final_position: Tuple[int, int]
    execution_time: float
    path_trace: List[Tuple[int, int]] = field(default_factory=list)
    action_sequence: List[Dict] = field(default_factory=list)
    error_message: Optional[str] = None

class NavigationBenchmark:
    """Main benchmark runner for navigation scenarios"""

    def __init__(self, scenarios: List[BenchmarkScenario], api_key: str):
        """Initialize benchmark with scenarios and API key"""
        if not scenarios:
            raise ValueError("At least one scenario is required")
        if not api_key or not isinstance(api_key, str):
            raise ValueError("Valid API key is required")

        self.scenarios = scenarios
        self.api_key = api_key
        self.results: List[NavigationResult] = []
        self.kernels: List[KernelType] = ["saamv1", "saamv2", "native"]

    async def _run_single_scenario(
        self,
        scenario: BenchmarkScenario,
        nav_system: LLMNavigationSystem
    ) -> NavigationResult:
        """Run a single scenario with a specific navigation system"""
        logger.info(f"Running scenario: {scenario.name}")
        logger.debug(f"Difficulty: {scenario.difficulty}")
        logger.debug(f"Start: {scenario.start_position} → Goal: {scenario.goal_position}")

        # Validate scenario parameters
        self._validate_scenario(scenario)

        environment = NavigationEnvironment(scenario.obstacles, DEFAULT_BOUNDS)
        nav_system.robot_state.position = scenario.start_position

        # Initialize map with obstacles
        for obstacle_pos in scenario.obstacles:
            nav_system.internal_map.set_cell(obstacle_pos, "OBSTACLE", 1.0)

        # Run navigation
        start_time = time.time()
        steps_taken = 0
        collision_occurred = False
        path_trace = [scenario.start_position]
        action_sequence = []
        error_message = None

        try:
            for step in range(scenario.max_steps):
                if time.time() - start_time > MAX_EXECUTION_TIME:
                    error_message = "Execution timeout"
                    break

                steps_taken = step + 1
                current_pos = nav_system.robot_state.position
                current_facing = nav_system.robot_state.facing

                # Get sensor readings with retries
                sensor_data = await self._get_sensor_data_with_retry(
                    environment, current_pos, current_facing
                )

                # Get next action with retries
                action_dict = await self._get_action_with_retry(
                    nav_system, sensor_data
                )

                if not action_dict:
                    error_message = "No action returned from navigation system"
                    break

                action_sequence.append(action_dict)

                # Execute action
                if not nav_system.execute_action(action_dict):
                    error_message = "Action execution failed"
                    break

                # Check for collision
                new_pos = nav_system.robot_state.position
                if not environment.is_valid_position(new_pos):
                    collision_occurred = True
                    error_message = "Collision detected"
                    break

                path_trace.append(new_pos)

                # Check if goal reached
                if nav_system.is_goal_reached():
                    break

                # Prevent API rate limiting
                await asyncio.sleep(API_RETRY_DELAY)

        except Exception as e:
            error_message = str(e)
            logger.error(f"Error running scenario {scenario.name}: {error_message}")

        execution_time = time.time() - start_time
        goal_reached = nav_system.is_goal_reached()
        timeout = steps_taken >= scenario.max_steps and not goal_reached

        # Calculate path efficiency with bounds checking
        optimal_steps = scenario.expected_path_length_range[0]
        try:
            path_efficiency = min(1.0, max(MIN_PATH_EFFICIENCY,
                optimal_steps / max(1, steps_taken)))
        except (ZeroDivisionError, TypeError):
            path_efficiency = MIN_PATH_EFFICIENCY

        success = goal_reached and not collision_occurred and not timeout

        result = NavigationResult(
            scenario_name=scenario.name,
            kernel=nav_system.kernel,
            success=success,
            steps_taken=steps_taken,
            path_efficiency=path_efficiency,
            goal_reached=goal_reached,
            timeout=timeout,
            collision_occurred=collision_occurred,
            final_position=nav_system.robot_state.position,
            execution_time=execution_time,
            path_trace=path_trace,
            action_sequence=action_sequence,
            error_message=error_message
        )

        status = "SUCCESS" if success else "FAILED"
        logger.info(
            f"{status} | Steps: {steps_taken} | "
            f"Efficiency: {path_efficiency:.2f} | "
            f"Time: {execution_time:.1f}s"
        )

        return result

    def _validate_scenario(self, scenario: BenchmarkScenario) -> None:
        """Validate scenario parameters"""
        if not scenario.name:
            raise ValueError("Scenario must have a name")
        if not isinstance(scenario.start_position, tuple) or len(scenario.start_position) != 2:
            raise ValueError("Invalid start position")
        if not isinstance(scenario.goal_position, tuple) or len(scenario.goal_position) != 2:
            raise ValueError("Invalid goal position")
        if scenario.max_steps <= 0:
            raise ValueError("Max steps must be positive")
        if (scenario.expected_path_length_range[0] <= 0 or
            scenario.expected_path_length_range[1] <= 0):
            raise ValueError("Path length range must be positive")

    async def _get_sensor_data_with_retry(
        self,
        environment: NavigationEnvironment,
        position: Tuple[int, int],
        facing: str,
        max_retries: int = MAX_API_RETRIES
    ) -> SensorReading:
        """Get sensor data with retry logic"""
        last_error = None
        for attempt in range(max_retries):
            try:
                return environment.get_sensor_readings(position, facing)
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    await asyncio.sleep(API_RETRY_DELAY * (attempt + 1))
                    continue
        raise RuntimeError(f"Failed to get sensor data after {max_retries} attempts: {str(last_error)}")

    async def _get_action_with_retry(
        self,
        nav_system: LLMNavigationSystem,
        sensor_data: SensorReading,
        max_retries: int = MAX_API_RETRIES
    ) -> Optional[Dict]:
        """Get navigation action with retry logic"""
        last_error = None
        for attempt in range(max_retries):
            try:
                return await nav_system.navigate_step(sensor_data)
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    await asyncio.sleep(API_RETRY_DELAY * (attempt + 1))
                    continue
        logger.error(f"Failed to get action after {max_retries} attempts: {str(last_error)}")
        return None

    async def run_all_scenarios(self) -> List[NavigationResult]:
        """Run all benchmark scenarios with all kernels"""
        logger.info("Starting Navigation System Benchmark")
        logger.info(f"Running {len(self.scenarios)} scenarios with {len(self.kernels)} kernels each")

        self.results = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = Path("benchmark_results") / timestamp
        base_dir.mkdir(parents=True, exist_ok=True)

        for scenario in self.scenarios:
            scenario_dir = base_dir / scenario.name.replace(" ", "_")
            scenario_dir.mkdir(exist_ok=True)

            for kernel in self.kernels:
                kernel_dir = scenario_dir / kernel
                kernel_dir.mkdir(exist_ok=True)

                logger.info(f"Running {scenario.name} with {kernel} kernel")
                try:
                    nav_system = LLMNavigationSystem(
                        target_position=scenario.goal_position,
                        kernel=kernel,
                        api_key=self.api_key,
                        cell_size_cm=DEFAULT_CELL_SIZE_CM
                    )

                    result = await self._run_single_scenario(scenario, nav_system)
                    self.results.append(result)
                    self._save_kernel_results(result, kernel_dir)
                except Exception as e:
                    logger.error(f"Failed to run scenario {scenario.name} with kernel {kernel}: {str(e)}")
                    continue

        logger.info(f"All tests completed. Results saved to {base_dir}")
        return self.results

    def _save_kernel_results(self, result: NavigationResult, output_dir: Path) -> None:
        """Save results for a single kernel run"""
        try:
            # Save raw data
            with open(output_dir / "result.json", 'w') as f:
                json.dump(asdict(result), f, indent=2)

            # Save visualization
            self._generate_kernel_visualization(result, output_dir)
        except IOError as e:
            logger.error(f"Failed to save results to {output_dir}: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error saving results: {str(e)}")

    def _generate_kernel_visualization(
        self,
        result: NavigationResult,
        output_dir: Path
    ) -> None:
        """Generate kernel-specific visualization"""
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))

            if result.path_trace:
                x, y = zip(*result.path_trace)
                plt.plot(x, y, 'b-o', label='Path', linewidth=2, markersize=5)

            # Plot start and goal
            plt.plot(result.path_trace[0][0], result.path_trace[0][1],
                    'go', markersize=10, label='Start')
            plt.plot(result.path_trace[-1][0], result.path_trace[-1][1],
                    'ro', markersize=10, label='Goal')

            plt.title(f"{result.scenario_name} - {result.kernel} Kernel\n"
                     f"Status: {'Success' if result.success else 'Failed'}")
            plt.xlabel("X Position")
            plt.ylabel("Y Position")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            output_path = output_dir / "path_visualization.png"
            plt.savefig(output_path, dpi=150)
            plt.close()
            logger.debug(f"Saved visualization to {output_path}")
        except ImportError:
            logger.warning("Matplotlib not available - skipping visualization")
        except Exception as e:
            logger.error(f"Failed to generate visualization: {str(e)}")

    def create_deepeval_dataset(self) -> EvaluationDataset:
        """Create DeepEval dataset from results with proper string formatting"""
        test_cases = []

        for scenario, result in zip(self.scenarios, self.results):
            try:
                # Convert dictionaries to JSON strings as expected by DeepEval
                actual_output = json.dumps({
                    'goal_reached': result.goal_reached,
                    'steps_taken': result.steps_taken,
                    'path_efficiency': result.path_efficiency,
                    'collision_occurred': result.collision_occurred,
                    'timeout': result.timeout,
                    'error_message': result.error_message,
                    'execution_time': result.execution_time,
                    'final_position': result.final_position,
                    'success': result.success
                })

                expected_output = json.dumps({
                    'expected_path_length_range': scenario.expected_path_length_range,
                    'max_steps': scenario.max_steps,
                    'goal_position': scenario.goal_position,
                    'difficulty': scenario.difficulty,
                    'scenario_type': scenario.scenario_type.value
                })

                test_case = LLMTestCase(
                    input=f"Navigate from {scenario.start_position} to {scenario.goal_position}",
                    actual_output=actual_output,
                    expected_output=expected_output,
                    context=[
                        f"Scenario: {scenario.name}",
                        f"Description: {scenario.description}",
                        f"Kernel: {result.kernel}"
                    ]
                )
                test_cases.append(test_case)
            except Exception as e:
                logger.error(f"Failed to create test case for {scenario.name}: {str(e)}")
                continue

        dataset = EvaluationDataset()
        dataset.test_cases = test_cases
        return dataset

    def evaluate_with_deepeval(self) -> Dict[str, Any]:
        """Evaluate results using DeepEval metrics with proper error handling"""
        try:
            dataset = self.create_deepeval_dataset()

            metrics = [
                NavigationSuccessMetric(threshold=1.0),
                PathEfficiencyMetric(threshold=0.7),
                CollisionAvoidanceMetric(threshold=1.0),
                RobustnessMetric(threshold=1.0)
            ]

            logger.info("Running DeepEval metrics...")
            evaluation_results = evaluate(
                test_cases=dataset.test_cases,
                metrics=metrics
            )

            processed_results = {
                'total_test_cases': len(dataset.test_cases),
                'metrics_summary': {},
                'individual_results': []
            }

            # Process evaluation results
            for i, test_result in enumerate(evaluation_results):
                if i >= len(self.scenarios):
                    break

                result_data = {
                    'scenario_name': self.scenarios[i].name,
                    'kernel': self.results[i].kernel,
                    'success': getattr(test_result, 'success', False),
                    'metrics': {}
                }

                for metric_data in getattr(test_result, 'metrics_data', []):
                    metric_name = getattr(metric_data, '__name__', str(type(metric_data).__name__))
                    result_data['metrics'][metric_name] = {
                        'score': getattr(metric_data, 'score', 0.0),
                        'success': getattr(metric_data, 'success', False),
                        'reason': getattr(metric_data, 'reason', 'No reason provided')
                    }

                processed_results['individual_results'].append(result_data)

            # Calculate overall metrics
            for metric in metrics:
                metric_name = metric.__name__
                scores = []
                successes = []

                for result in processed_results['individual_results']:
                    if metric_name in result['metrics']:
                        scores.append(result['metrics'][metric_name]['score'])
                        successes.append(result['metrics'][metric_name]['success'])

                if scores:
                    processed_results['metrics_summary'][metric_name] = {
                        'average_score': np.mean(scores),
                        'success_rate': np.mean(successes),
                        'total_tests': len(scores)
                    }

            return processed_results

        except Exception as e:
            logger.error(f"DeepEval evaluation failed: {str(e)}")
            return {
                'error': str(e),
                'total_test_cases': len(self.results),
                'metrics_summary': {},
                'individual_results': []
            }

    def generate_detailed_report(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark report"""
        if not self.results:
            return {'error': 'No results available'}

        # Overall statistics
        total_scenarios = len(self.results)
        successful_scenarios = sum(1 for r in self.results if r.success)
        success_rate = successful_scenarios / total_scenarios if total_scenarios > 0 else 0

        # Performance by difficulty and kernel
        difficulty_stats = {}
        kernel_stats = {}

        for scenario, result in zip(self.scenarios, self.results):
            # Difficulty stats
            diff = scenario.difficulty
            if diff not in difficulty_stats:
                difficulty_stats[diff] = {'total': 0, 'success': 0, 'avg_efficiency': []}

            difficulty_stats[diff]['total'] += 1
            if result.success:
                difficulty_stats[diff]['success'] += 1
            difficulty_stats[diff]['avg_efficiency'].append(result.path_efficiency)

            # Kernel stats
            kernel = result.kernel
            if kernel not in kernel_stats:
                kernel_stats[kernel] = {'total': 0, 'success': 0, 'avg_time': []}

            kernel_stats[kernel]['total'] += 1
            if result.success:
                kernel_stats[kernel]['success'] += 1
            kernel_stats[kernel]['avg_time'].append(result.execution_time)

        # Calculate averages
        for diff, data in difficulty_stats.items():
            data['success_rate'] = data['success'] / data['total']
            data['avg_efficiency'] = float(np.mean(data['avg_efficiency']))

        for kernel, data in kernel_stats.items():
            data['success_rate'] = data['success'] / data['total']
            data['avg_time'] = float(np.mean(data['avg_time']))

        # Failure analysis
        failures = [r for r in self.results if not r.success]
        failure_reasons = {
            'timeout': sum(1 for f in failures if f.timeout),
            'collision': sum(1 for f in failures if f.collision_occurred),
            'error': sum(1 for f in failures if f.error_message),
            'goal_not_reached': sum(1 for f in failures if not f.goal_reached and not f.timeout and not f.collision_occurred)
        }

        # Performance metrics
        avg_execution_time = float(np.mean([r.execution_time for r in self.results]))
        avg_steps = float(np.mean([r.steps_taken for r in self.results]))
        avg_efficiency = float(np.mean([r.path_efficiency for r in self.results]))

        report = {
            'summary': {
                'total_scenarios': total_scenarios,
                'successful_scenarios': successful_scenarios,
                'success_rate': success_rate,
                'avg_execution_time': avg_execution_time,
                'avg_steps': avg_steps,
                'avg_path_efficiency': avg_efficiency
            },
            'difficulty_breakdown': difficulty_stats,
            'kernel_breakdown': kernel_stats,
            'failure_analysis': failure_reasons,
            'scenario_details': [
                {
                    'name': scenario.name,
                    'difficulty': scenario.difficulty,
                    'kernel': result.kernel,
                    'success': result.success,
                    'steps': result.steps_taken,
                    'efficiency': result.path_efficiency,
                    'time': result.execution_time,
                    'failure_reason': self._get_failure_reason(result)
                }
                for scenario, result in zip(self.scenarios, self.results)
            ]
        }

        return report

    def _get_failure_reason(self, result: NavigationResult) -> Optional[str]:
        """Determine primary failure reason"""
        if result.success:
            return None
        if result.timeout:
            return "timeout"
        if result.collision_occurred:
            return "collision"
        if result.error_message:
            return "error"
        if not result.goal_reached:
            return "goal_not_reached"
        return "unknown"

    def save_results(self, base_dir: str = "benchmark_results") -> str:
        """Save results to a new timestamped folder. Returns path to results."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(base_dir) / f"run_{timestamp}"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save report
            report = self.generate_detailed_report()
            with open(output_dir / "benchmark_report.json", 'w') as f:
                json.dump(report, f, indent=2, default=str)

            # Save raw data
            with open(output_dir / "raw_results.json", 'w') as f:
                json.dump([asdict(r) for r in self.results], f, indent=2)

            # Copy the visualization
            if (Path(base_dir) / "latest_benchmark.png").exists():
                shutil.copy(
                    Path(base_dir) / "latest_benchmark.png",
                    output_dir / "benchmark_overview.png"
                )

            logger.info(f"Results saved to {output_dir}")
            return str(output_dir)
        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")
            raise

async def main():
    """Main entry point for benchmark execution"""
    import os
    from .scenarios import create_benchmark_scenarios

    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")

        scenarios = create_benchmark_scenarios()
        benchmark = NavigationBenchmark(scenarios, api_key)
        await benchmark.run_all_scenarios()

        # Generate and save reports
        benchmark.evaluate_with_deepeval()
        benchmark.save_results()

    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())