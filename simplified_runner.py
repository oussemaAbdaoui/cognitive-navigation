#!/usr/bin/env python3
# navigation_benchmark.py - Comprehensive benchmark suite for LLM Navigation System

import asyncio
import json
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

# DeepEval imports
from deepeval import evaluate
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset

# Import your navigation system
from langgraph_new import LLMNavigationSystem, SensorReading
from loguru import logger

# Configure logging
logger.add(
    "benchmark.log",
    rotation="10 MB",
    retention="7 days",
    level="INFO",
    enqueue=True,
    backtrace=True,
    diagnose=True
)
# ------------------- Benchmark Configuration -------------------

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
    """Individual benchmark scenario"""
    name: str
    scenario_type: ScenarioType
    start_position: Tuple[int, int]
    goal_position: Tuple[int, int]
    obstacles: List[Tuple[int, int]]
    expected_path_length_range: Tuple[int, int]  # min, max expected steps
    max_steps: int
    description: str
    difficulty: str  # "easy", "medium", "hard", "expert"

@dataclass
class NavigationResult:
    """Results from a navigation attempt"""
    scenario_name: str
    success: bool
    steps_taken: int
    path_efficiency: float
    goal_reached: bool
    timeout: bool
    collision_occurred: bool
    final_position: Tuple[int, int]
    execution_time: float
    error_message: Optional[str] = None
    path_trace: List[Tuple[int, int]] = field(default_factory=list)
    action_sequence: List[Dict] = field(default_factory=list)

# ------------------- Custom DeepEval Metrics -------------------
class NavigationSuccessMetric(BaseMetric):
    """Metric to evaluate navigation success"""

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self.score = 0.0
        self.reason = ""

    def measure(self, test_case: LLMTestCase) -> float:
        """Measure navigation success (1.0 for success, 0.0 for failure)"""
        try:
            # Parse the actual output JSON string
            result = json.loads(test_case.actual_output)
            self.score = 1.0 if result.get('goal_reached', False) else 0.0
            self.reason = f"Goal reached: {result.get('goal_reached', False)}"
            return self.score
        except (json.JSONDecodeError, AttributeError) as e:
            self.score = 0.0
            self.reason = f"Error parsing output: {str(e)}"
            return self.score

    def is_successful(self) -> bool:
        return self.score >= self.threshold

    @property
    def __name__(self):
        return "Navigation Success"

class PathEfficiencyMetric(BaseMetric):
    """Metric to evaluate path efficiency"""

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self.score = 0.0
        self.reason = ""

    def measure(self, test_case: LLMTestCase) -> float:
        """Measure path efficiency (optimal_length / actual_length)"""
        try:
            result = json.loads(test_case.actual_output)
            expected = json.loads(test_case.expected_output)

            expected_range = expected.get('expected_path_length_range', [1, 100])
            actual_steps = result.get('steps_taken', float('inf'))

            if actual_steps == 0:
                self.score = 0.0
                self.reason = "No steps taken"
                return self.score

            optimal_length = expected_range[0]
            self.score = min(1.0, optimal_length / actual_steps)
            self.reason = f"Efficiency: {optimal_length}/{actual_steps} = {self.score:.3f}"
            return self.score

        except (json.JSONDecodeError, AttributeError, ZeroDivisionError) as e:
            self.score = 0.0
            self.reason = f"Error calculating efficiency: {str(e)}"
            return self.score

    def is_successful(self) -> bool:
        return self.score >= self.threshold

    @property
    def __name__(self):
        return "Path Efficiency"

class CollisionAvoidanceMetric(BaseMetric):
    """Metric to evaluate collision avoidance"""

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self.score = 0.0
        self.reason = ""

    def measure(self, test_case: LLMTestCase) -> float:
        """Measure collision avoidance (1.0 for no collisions, 0.0 for collision)"""
        try:
            result = json.loads(test_case.actual_output)
            collision_occurred = result.get('collision_occurred', False)
            self.score = 0.0 if collision_occurred else 1.0
            self.reason = f"Collision occurred: {collision_occurred}"
            return self.score
        except (json.JSONDecodeError, AttributeError) as e:
            self.score = 0.0
            self.reason = f"Error parsing collision data: {str(e)}"
            return self.score

    def is_successful(self) -> bool:
        return self.score >= self.threshold

    @property
    def __name__(self):
        return "Collision Avoidance"

class RobustnessMetric(BaseMetric):
    """Metric to evaluate robustness (no timeouts or errors)"""

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self.score = 0.0
        self.reason = ""

    def measure(self, test_case: LLMTestCase) -> float:
        """Measure robustness"""
        try:
            result = json.loads(test_case.actual_output)
            timeout = result.get('timeout', False)
            error_message = result.get('error_message', None)

            if timeout or error_message:
                self.score = 0.0
                self.reason = f"Timeout: {timeout}, Error: {bool(error_message)}"
            else:
                self.score = 1.0
                self.reason = "No timeouts or errors"
            return self.score
        except (json.JSONDecodeError, AttributeError) as e:
            self.score = 0.0
            self.reason = f"Error parsing robustness data: {str(e)}"
            return self.score

    def is_successful(self) -> bool:
        return self.score >= self.threshold

    @property
    def __name__(self):
        return "Robustness"
# ------------------- Environment Simulator -------------------

class NavigationEnvironment:
    """Simulated environment for navigation testing"""

    def __init__(self, obstacles: List[Tuple[int, int]], bounds: Tuple[int, int, int, int]):
        self.obstacles = set(obstacles)
        self.bounds = bounds  # min_row, max_row, min_col, max_col
        self.cell_size_cm = 30

    def get_sensor_readings(self, position: Tuple[int, int], facing: str) -> SensorReading:
        """Simulate sensor readings from current position"""

        # Direction vectors for sensor orientations
        direction_vectors = {
            "UP": {"front": (-1, 0), "left": (0, -1), "right": (0, 1)},
            "DOWN": {"front": (1, 0), "left": (0, 1), "right": (0, -1)},
            "LEFT": {"front": (0, -1), "left": (1, 0), "right": (-1, 0)},
            "RIGHT": {"front": (0, 1), "left": (-1, 0), "right": (1, 0)}
        }

        vectors = direction_vectors[facing]
        max_range_cells = 10  # ~300cm range

        readings = {}
        for sensor_dir, (dr, dc) in vectors.items():
            distance_cm = self._cast_ray(position, dr, dc, max_range_cells)
            readings[sensor_dir] = distance_cm

        return SensorReading(
            front=readings["front"],
            left=readings["left"],
            right=readings["right"]
        )

# In NavigationEnvironment._cast_ray()
    def _cast_ray(self, start_pos: Tuple[int, int], dr: int, dc: int, max_cells: int) -> float:
        """Cast a ray and return distance to first obstacle"""
        r, c = start_pos

        # Check current position first (important for obstacle right in front)
        if (r, c) in self.obstacles:
            return 0.0  # Already in obstacle

        for i in range(1, max_cells + 1):
            check_pos = (r + i * dr, c + i * dc)

            if check_pos in self.obstacles:
                return i * self.cell_size_cm

        return max_cells * self.cell_size_cm

    def is_valid_position(self, position: Tuple[int, int]) -> bool:
        """Check if position is valid (within bounds and not obstacle)"""
        r, c = position
        return (self.bounds[0] <= r <= self.bounds[1] and
                self.bounds[2] <= c <= self.bounds[3] and
                position not in self.obstacles)

# ------------------- Benchmark Scenarios -------------------

def create_benchmark_scenarios() -> List[BenchmarkScenario]:
    """Create 10 diverse benchmark scenarios"""

    scenarios = [

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
    ]

    return scenarios

# ------------------- Benchmark Runner -------------------

class NavigationBenchmark:
    """Main benchmark runner"""

    def __init__(self, scenarios: List[BenchmarkScenario], api_key: str):
        self.scenarios = scenarios
        self.api_key = api_key
        self.results: List[NavigationResult] = []

    async def run_single_scenario(self, scenario: BenchmarkScenario) -> NavigationResult:
        """Run a single scenario and return results"""
        print(f"\n🚀 Running scenario: {scenario.name}")
        print(f"   Difficulty: {scenario.difficulty}")
        print(f"   Start: {scenario.start_position} → Goal: {scenario.goal_position}")

        # Create environment
        bounds = (-10, 10, -10, 10)  # Reasonable bounds for testing
        environment = NavigationEnvironment(scenario.obstacles, bounds)

        # Initialize navigation system
        try:
            nav_system = LLMNavigationSystem(
                target_position=scenario.goal_position,
                cell_size_cm=30
            )
            nav_system.robot_state.position = scenario.start_position

            # Place obstacles in the internal map
            for obstacle_pos in scenario.obstacles:
                nav_system.internal_map.set_cell(obstacle_pos, "OBSTACLE", 1.0)

        except Exception as e:
            return NavigationResult(
                scenario_name=scenario.name,
                success=False,
                steps_taken=0,
                path_efficiency=0.0,
                goal_reached=False,
                timeout=False,
                collision_occurred=False,
                final_position=scenario.start_position,
                execution_time=0.0,
                error_message=f"Setup error: {str(e)}"
            )

        # Run navigation
        start_time = time.time()
        steps_taken = 0
        collision_occurred = False
        path_trace = [scenario.start_position]
        action_sequence = []

        try:
            for step in range(scenario.max_steps):
                steps_taken = step + 1

                # Get sensor readings
                current_pos = nav_system.robot_state.position
                current_facing = nav_system.robot_state.facing
                sensor_data = environment.get_sensor_readings(current_pos, current_facing)

                # Get next action
                action_dict = await nav_system.navigate_step(sensor_data)

                if not action_dict:
                    break

                action_sequence.append(action_dict)

                # Execute action
                success = nav_system.execute_action(action_dict)
                if not success:
                    break

                # Check for collision
                new_pos = nav_system.robot_state.position
                if not environment.is_valid_position(new_pos):
                    collision_occurred = True
                    break

                path_trace.append(new_pos)

                # Check if goal reached
                if nav_system.is_goal_reached():
                    break

                # Small delay to prevent API rate limiting
                await asyncio.sleep(0.1)

        except Exception as e:
            execution_time = time.time() - start_time
            return NavigationResult(
                scenario_name=scenario.name,
                success=False,
                steps_taken=steps_taken,
                path_efficiency=0.0,
                goal_reached=False,
                timeout=False,
                collision_occurred=collision_occurred,
                final_position=nav_system.robot_state.position,
                execution_time=execution_time,
                error_message=str(e),
                path_trace=path_trace,
                action_sequence=action_sequence
            )

        execution_time = time.time() - start_time
        goal_reached = nav_system.is_goal_reached()
        timeout = steps_taken >= scenario.max_steps and not goal_reached

        # Calculate path efficiency
        optimal_steps = scenario.expected_path_length_range[0]
        path_efficiency = min(1.0, optimal_steps / steps_taken) if steps_taken > 0 else 0.0

        success = goal_reached and not collision_occurred and not timeout

        result = NavigationResult(
            scenario_name=scenario.name,
            success=success,
            steps_taken=steps_taken,
            path_efficiency=path_efficiency,
            goal_reached=goal_reached,
            timeout=timeout,
            collision_occurred=collision_occurred,
            final_position=nav_system.robot_state.position,
            execution_time=execution_time,
            path_trace=path_trace,
            action_sequence=action_sequence
        )

        # Print immediate results
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"   {status} | Steps: {steps_taken} | Efficiency: {path_efficiency:.2f} | Time: {execution_time:.1f}s")

        return result

    async def run_all_scenarios(self) -> List[NavigationResult]:
        """Run all benchmark scenarios"""
        print("🔍 Starting Navigation System Benchmark")
        print(f"📊 Running {len(self.scenarios)} scenarios...")

        self.results = []
        for scenario in self.scenarios:
            result = await self.run_single_scenario(scenario)
            self.results.append(result)

        return self.results


    def create_deepeval_dataset(self) -> EvaluationDataset:
        """Create DeepEval dataset from results with proper string formatting"""
        test_cases = []

        for scenario, result in zip(self.scenarios, self.results):
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

            # Create test case with string inputs/outputs
            test_case = LLMTestCase(
                input=f"Navigate from {scenario.start_position} to {scenario.goal_position} in scenario: {scenario.name}",
                actual_output=actual_output,
                expected_output=expected_output,
                context=[scenario.description]
            )
            test_cases.append(test_case)

        # Create dataset using the correct method
        dataset = EvaluationDataset()
        dataset.test_cases = test_cases  # Direct assignment as per v3.2.8 API
        return dataset

    def evaluate_with_deepeval(self) -> Dict[str, Any]:
        """Evaluate results using DeepEval metrics with proper error handling"""
        try:
            dataset = self.create_deepeval_dataset()

            # Define metrics with proper initialization
            metrics = [
                NavigationSuccessMetric(threshold=1.0),
                PathEfficiencyMetric(threshold=0.7),
                CollisionAvoidanceMetric(threshold=1.0),
                RobustnessMetric(threshold=1.0)
            ]

            # Run evaluation with error handling
            print("  🔄 Running DeepEval metrics...")

            # Use the correct evaluate function call for your DeepEval version
            evaluation_results = evaluate(
                test_cases=dataset.test_cases,
                metrics=metrics
            )

            # Process and return results
            processed_results = {
                'total_test_cases': len(dataset.test_cases),
                'metrics_summary': {},
                'individual_results': []
            }

            # Handle different DeepEval result formats
            test_results = []
            if hasattr(evaluation_results, 'test_results'):
                test_results = evaluation_results.test_results
            elif isinstance(evaluation_results, list):
                test_results = evaluation_results
            else:
                # Try to iterate directly if it's iterable
                try:
                    test_results = list(evaluation_results)
                except:
                    print("  ⚠️ Unable to parse DeepEval results format")
                    return processed_results

            # Extract metric summaries
            for i, test_result in enumerate(test_results):
                if i >= len(self.scenarios):
                    break

                result_data = {
                    'scenario_name': self.scenarios[i].name,
                    'input': getattr(test_result, 'input', f"Scenario {i}"),
                    'success': getattr(test_result, 'success', False),
                    'metrics': {}
                }

                # Handle metrics data
                metrics_data = getattr(test_result, 'metrics_data', [])
                for metric_data in metrics_data:
                    metric_name = getattr(metric_data, 'metric', str(type(metric_data).__name__))
                    result_data['metrics'][metric_name] = {
                        'score': getattr(metric_data, 'score', 0.0),
                        'success': getattr(metric_data, 'success', False),
                        'reason': getattr(metric_data, 'reason', 'No reason provided')
                    }

                processed_results['individual_results'].append(result_data)

            # Calculate overall metric summaries
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
                        'average_score': sum(scores) / len(scores),
                        'success_rate': sum(successes) / len(successes),
                        'total_tests': len(scores)
                    }

            return processed_results

        except Exception as e:
            logger.error(f"DeepEval evaluation failed: {str(e)}")
            return {
                'error': str(e),
                'total_test_cases': len(self.results) if hasattr(self, 'results') else 0,
                'metrics_summary': {},
                'individual_results': []
            }
    def generate_detailed_report(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark report"""

        # Overall statistics
        total_scenarios = len(self.results)
        successful_scenarios = sum(1 for r in self.results if r.success)
        success_rate = successful_scenarios / total_scenarios if total_scenarios > 0 else 0

        # Performance by difficulty
        difficulty_stats = {}
        for scenario, result in zip(self.scenarios, self.results):
            diff = scenario.difficulty
            if diff not in difficulty_stats:
                difficulty_stats[diff] = {'total': 0, 'success': 0, 'avg_efficiency': []}

            difficulty_stats[diff]['total'] += 1
            if result.success:
                difficulty_stats[diff]['success'] += 1
            difficulty_stats[diff]['avg_efficiency'].append(result.path_efficiency)

        # Calculate averages
        for diff_data in difficulty_stats.values():
            diff_data['success_rate'] = diff_data['success'] / diff_data['total']
            diff_data['avg_efficiency'] = np.mean(diff_data['avg_efficiency'])

        # Failure analysis
        failures = [r for r in self.results if not r.success]
        failure_reasons = {
            'timeout': sum(1 for f in failures if f.timeout),
            'collision': sum(1 for f in failures if f.collision_occurred),
            'error': sum(1 for f in failures if f.error_message),
            'goal_not_reached': sum(1 for f in failures if not f.goal_reached and not f.timeout and not f.collision_occurred)
        }

        # Performance metrics
        avg_execution_time = np.mean([r.execution_time for r in self.results])
        avg_steps = np.mean([r.steps_taken for r in self.results])
        avg_efficiency = np.mean([r.path_efficiency for r in self.results])

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
            'failure_analysis': failure_reasons,
            'scenario_details': [
                {
                    'name': scenario.name,
                    'difficulty': scenario.difficulty,
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

    def save_results(self, output_dir: str = "benchmark_results"):
        """Save benchmark results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Save detailed report
        report = self.generate_detailed_report()
        with open(output_path / "benchmark_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Save raw results
        results_data = []
        for scenario, result in zip(self.scenarios, self.results):
            results_data.append({
                'scenario': scenario.__dict__,
                'result': result.__dict__
            })

        with open(output_path / "raw_results.json", 'w') as f:
            json.dump(results_data, f, indent=2, default=str)

        print(f"📁 Results saved to {output_path}")

# ------------------- Visualization -------------------

def create_benchmark_visualizations(benchmark: NavigationBenchmark, output_dir: str = "benchmark_results"):
    """Create visualizations for benchmark results"""

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Prepare data
    scenario_names = [s.name for s in benchmark.scenarios]
    success_rates = [1 if r.success else 0 for r in benchmark.results]
    efficiencies = [r.path_efficiency for r in benchmark.results]
    execution_times = [r.execution_time for r in benchmark.results]
    difficulties = [s.difficulty for s in benchmark.scenarios]

    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")

    # 1. Success Rate by Scenario
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    colors = ['green' if s else 'red' for s in success_rates]
    ax1.bar(range(len(scenario_names)), success_rates, color=colors)
    ax1.set_title('Success Rate by Scenario')
    ax1.set_ylabel('Success (1) / Failure (0)')
    ax1.set_xticks(range(len(scenario_names)))
    ax1.set_xticklabels(scenario_names, rotation=45, ha='right')

    # 2. Path Efficiency
    ax2.bar(range(len(scenario_names)), efficiencies, color='skyblue')
    ax2.set_title('Path Efficiency by Scenario')
    ax2.set_ylabel('Efficiency Score')
    ax2.set_xticks(range(len(scenario_names)))
    ax2.set_xticklabels(scenario_names, rotation=45, ha='right')
    ax2.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='Threshold (0.7)')
    ax2.legend()

    # 3. Execution Time
    ax3.bar(range(len(scenario_names)), execution_times, color='orange')
    ax3.set_title('Execution Time by Scenario')
    ax3.set_ylabel('Time (seconds)')
    ax3.set_xticks(range(len(scenario_names)))
    ax3.set_xticklabels(scenario_names, rotation=45, ha='right')

    # 4. Performance by Difficulty
    df = pd.DataFrame({
        'Difficulty': difficulties,
        'Success': success_rates,
        'Efficiency': efficiencies
    })

    difficulty_stats = df.groupby('Difficulty').agg({
        'Success': 'mean',
        'Efficiency': 'mean'
    }).reset_index()

    x_pos = range(len(difficulty_stats))
    width = 0.35

    ax4.bar([x - width/2 for x in x_pos], difficulty_stats['Success'],
            width, label='Success Rate', color='green', alpha=0.7)
    ax4.bar([x + width/2 for x in x_pos], difficulty_stats['Efficiency'],
            width, label='Avg Efficiency', color='blue', alpha=0.7)

    ax4.set_title('Performance by Difficulty Level')
    ax4.set_ylabel('Score')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(difficulty_stats['Difficulty'])
    ax4.legend()
    ax4.set_ylim(0, 1)# In NavigationEnvironment._cast_ray()
def _cast_ray(self, start_pos: Tuple[int, int], dr: int, dc: int, max_cells: int) -> float:
    """Cast a ray and return distance to first obstacle"""
    r, c = start_pos

    # Check current position first (important for obstacle right in front)
    if (r, c) in self.obstacles:
        return 0.0  # Already in obstacle

    for i in range(0, max_cells + 1):
        check_pos = (r + i * dr, c + i * dc)

        if check_pos in self.obstacles:
            return i * self.cell_size_cm

    return max_cells * self.cell_size_cm

    plt.tight_layout()
    plt.savefig(output_path / "benchmark_overview.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📊 Visualizations saved to {output_path}")

# ------------------- Main Benchmark Execution -------------------

async def main():
    """Main benchmark execution function"""

    # Check for API key
    import os
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ Error: GROQ_API_KEY environment variable not set")
        return

    # Create scenarios
    scenarios = create_benchmark_scenarios()

    # Initialize benchmark
    benchmark = NavigationBenchmark(scenarios, api_key)

    # Run benchmark
    print("🏁 Starting comprehensive navigation benchmark...")
    results = await benchmark.run_all_scenarios()

    # Generate reports
    print("\n📊 Generating benchmark report...")
    report = benchmark.generate_detailed_report()

    # Print summary
    print("\n" + "="*60)
    print("🎯 BENCHMARK SUMMARY")
    print("="*60)
    print(f"Total Scenarios: {report['summary']['total_scenarios']}")
    print(f"Successful: {report['summary']['successful_scenarios']}")
    print(f"Success Rate: {report['summary']['success_rate']:.2%}")
    print(f"Average Execution Time: {report['summary']['avg_execution_time']:.2f}s")
    print(f"Average Steps: {report['summary']['avg_steps']:.1f}")
    print(f"Average Path Efficiency: {report['summary']['avg_path_efficiency']:.2f}")

    # Performance by difficulty
    print("\n📈 PERFORMANCE BY DIFFICULTY:")
    for difficulty, stats in report['difficulty_breakdown'].items():
        print(f"  {difficulty.upper()}: {stats['success_rate']:.2%} success, "
              f"{stats['avg_efficiency']:.2f} avg efficiency")

    # Failure analysis
    print("\n⚠️  FAILURE ANALYSIS:")
    total_failures = sum(report['failure_analysis'].values())
    if total_failures > 0:
        for reason, count in report['failure_analysis'].items():
            if count > 0:
                percentage = (count / total_failures) * 100
                print(f"  {reason.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
    else:
        print("  No failures detected! 🎉")

    # Detailed scenario results
    print("\n📋 DETAILED SCENARIO RESULTS:")
    for detail in report['scenario_details']:
        status = "✅" if detail['success'] else "❌"
        failure_info = f" ({detail['failure_reason']})" if detail['failure_reason'] else ""
        print(f"  {status} {detail['name']:<20} | "
              f"Steps: {detail['steps']:>3} | "
              f"Efficiency: {detail['efficiency']:>5.2f} | "
              f"Time: {detail['time']:>5.1f}s{failure_info}")

    # Save results
    print("\n💾 Saving results...")
    benchmark.save_results()

    # Run DeepEval evaluation
    print("\n🔍 Running DeepEval evaluation...")
    try:
        deepeval_results = benchmark.evaluate_with_deepeval()
        print("✅ DeepEval evaluation completed")

        # Print DeepEval metrics summary
        print("\n📏 DEEPEVAL METRICS:")
        if hasattr(deepeval_results, 'test_results'):
            for test_result in deepeval_results.test_results[:5]:  # Show first 5
                print(f"  Test: {test_result.input[:30]}...")
                for metric_result in test_result.metrics_data:
                    status = "✅" if metric_result.success else "❌"
                    print(f"    {status} {metric_result.metric}: {metric_result.score:.3f}")
                print()
    except Exception as e:
        print(f"⚠️  DeepEval evaluation failed: {str(e)}")

    # Create visualizations
    print("\n📊 Creating visualizations...")
    try:
        create_benchmark_visualizations(benchmark)
        print("✅ Visualizations created successfully")
    except Exception as e:
        print(f"⚠️  Visualization creation failed: {str(e)}")

    # Final recommendations
    print("\n💡 RECOMMENDATIONS:")
    success_rate = report['summary']['success_rate']
    avg_efficiency = report['summary']['avg_path_efficiency']

    if success_rate < 0.7:
        print("  🔴 LOW SUCCESS RATE: Consider improving obstacle detection and path planning")
    elif success_rate < 0.9:
        print("  🟡 MODERATE SUCCESS RATE: Fine-tune navigation algorithms for edge cases")
    else:
        print("  🟢 HIGH SUCCESS RATE: System performing well!")

    if avg_efficiency < 0.6:
        print("  🔴 LOW EFFICIENCY: Optimize pathfinding algorithms for shorter routes")
    elif avg_efficiency < 0.8:
        print("  🟡 MODERATE EFFICIENCY: Consider implementing more advanced planning strategies")
    else:
        print("  🟢 HIGH EFFICIENCY: Excellent path optimization!")

    # Identify problematic scenarios
    failed_scenarios = [d for d in report['scenario_details'] if not d['success']]
    if failed_scenarios:
        print(f"\n🎯 FOCUS AREAS ({len(failed_scenarios)} failed scenarios):")
        for scenario in failed_scenarios[:3]:  # Show top 3 failures
            print(f"  • {scenario['name']} ({scenario['difficulty']}) - {scenario['failure_reason']}")

    print("\n🏆 Benchmark completed successfully!")
    print(f"📁 Results available in: benchmark_results/")

if __name__ == "__main__":
    asyncio.run(main())