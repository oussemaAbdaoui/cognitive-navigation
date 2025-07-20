#!/usr/bin/env python3
import time
import random
import numpy as np
import psutil
from typing import Dict, List, Tuple, Any
from loguru import logger
from benchmark_config import BENCHMARK_SCENARIOS
from langgraph_nodes import StateManager, WaypointPlanner, ActionPlanner, Action

class EmergencyStop(Exception):
    pass

class Direction:
    @staticmethod
    def turn_right(facing: str) -> str:
        turns = {"UP": "RIGHT", "RIGHT": "DOWN", "DOWN": "LEFT", "LEFT": "UP"}
        return turns.get(facing, facing)

    @staticmethod
    def turn_left(facing: str) -> str:
        turns = {"UP": "LEFT", "LEFT": "DOWN", "DOWN": "RIGHT", "RIGHT": "UP"}
        return turns.get(facing, facing)

# ------------------- SAAM Modules -------------------
class PerceptorModule:
    def __init__(self, state_manager: StateManager):
        self.state_manager = state_manager
        self.name = "PERCEPTOR"

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process sensor data and update grid state"""
        try:
            sensor_data = input_data["sensor_data"]
            grid_vis = self.state_manager.process_sensor_data(sensor_data)
            return {
                "grid_vis": grid_vis,
                "robot_pos": tuple(self.state_manager.robot_position),
                "robot_facing": self.state_manager.robot_facing,
                "confidence": 0.9  # High confidence for perception
            }
        except Exception as e:
            logger.error(f"Perception failed: {str(e)}")
            return {"error": f"Perception failed: {str(e)}", "confidence": 0.1}

class PlannerModule:
    def __init__(self, waypoint_planner: WaypointPlanner, action_planner: ActionPlanner):
        self.waypoint_planner = waypoint_planner
        self.action_planner = action_planner
        self.name = "PLANNER"

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Plan waypoint and generate actions"""
        try:
            # Access data from Perceptor
            grid_vis = input_data["grid_vis"]
            robot_pos = input_data["robot_pos"]
            goal_pos = input_data["goal_pos"]
            robot_facing = input_data["robot_facing"]

            waypoint = self.waypoint_planner.plan_waypoint(
                grid_vis,
                robot_pos,
                goal_pos
            )
            if not waypoint:
                return {"error": "Waypoint planning failed", "confidence": 0.2}

            actions = self.action_planner.plan_actions(
                waypoint,
                robot_pos,
                robot_facing
            )
            if not actions:
                return {"error": "Action planning failed", "confidence": 0.3}

            return {
                "waypoint": waypoint,
                "actions": actions,
                "confidence": 0.85  # Moderate confidence for planning
            }
        except KeyError as e:
            logger.error(f"Missing input: {str(e)}")
            return {"error": f"Missing input: {str(e)}", "confidence": 0.1}
        except Exception as e:
            logger.error(f"Planning failed: {str(e)}")
            return {"error": f"Planning failed: {str(e)}", "confidence": 0.1}

class ExecutorModule:
    def __init__(self, grid_size: Tuple[int, int], true_grid: List[List[str]]):
        self.grid_size = grid_size
        self.true_grid = true_grid
        self.name = "EXECUTOR"

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute planned actions in true environment"""
        try:
            # Access data from Planner
            actions = input_data["actions"]
            current_pos = list(input_data["true_robot_pos"])
            current_facing = input_data["true_robot_facing"]

            for action in actions:
                self._execute_action(
                    action,
                    current_pos,
                    current_facing,
                    self.grid_size,
                    self.true_grid
                )

            return {
                "new_robot_pos": tuple(current_pos),
                "new_robot_facing": current_facing,
                "confidence": 0.95  # High confidence for execution
            }
        except KeyError as e:
            logger.error(f"Missing input: {str(e)}")
            return {"error": f"Missing input: {str(e)}", "confidence": 0.1}
        except Exception as e:
            logger.error(f"Execution failed: {str(e)}")
            return {"error": f"Execution failed: {str(e)}", "confidence": 0.1}

    def _execute_action(self, action: Action, current_pos: List[int],
                        current_facing: str, grid_size: Tuple[int, int],
                        true_grid: List[List[str]]):
        """Execute a single action"""
        new_pos = list(current_pos)
        new_facing = current_facing

        if action.type == "TURN":
            if action.direction == "RIGHT":
                new_facing = Direction.turn_right(current_facing)
            elif action.direction == "LEFT":
                new_facing = Direction.turn_left(current_facing)
            else:
                raise EmergencyStop(f"Invalid turn direction: {action.direction}")

        elif action.type == "MOVE":
            cells = action.cells
            dr, dc = 0, 0

            if current_facing == "UP":
                dr = -cells
            elif current_facing == "DOWN":
                dr = cells
            elif current_facing == "LEFT":
                dc = -cells
            elif current_facing == "RIGHT":
                dc = cells

            new_r = current_pos[0] + dr
            new_c = current_pos[1] + dc

            # Validate path
            if not (0 <= new_r < grid_size[0] and 0 <= new_c < grid_size[1]):
                raise EmergencyStop(f"Move would go out of bounds: ({new_r}, {new_c})")

            if true_grid[new_r][new_c] == '■':
                raise EmergencyStop(f"Move would hit obstacle at ({new_r}, {new_c})")

            # Update position if valid
            new_pos[0] = new_r
            new_pos[1] = new_c

        # Commit changes
        current_pos[0], current_pos[1] = new_pos
        current_facing = new_facing

class EvaluatorModule:
    def __init__(self):
        self.name = "EVALUATOR"
        self.error_history = []

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate plan safety and progress"""
        try:
            # Access data from Executor
            current_pos = input_data["new_robot_pos"]
            goal_pos = input_data["goal_pos"]
            grid_size = input_data["grid_size"]

            # Calculate progress toward goal
            distance = abs(current_pos[0]-goal_pos[0]) + abs(current_pos[1]-goal_pos[1])
            max_distance = grid_size[0] + grid_size[1]
            progress = 1.0 - (distance / max_distance)

            # Check for recent errors
            error_score = 0.0
            if "error" in input_data:
                error_score = 0.5
                self.error_history.append(input_data["error"])
            elif len(self.error_history) > 0:
                error_score = 0.2

            return {
                "progress": progress,
                "safety_score": 1.0 - error_score,
                "confidence": 0.8  # High confidence for evaluation
            }
        except KeyError as e:
            logger.error(f"Missing input: {str(e)}")
            return {"error": f"Missing input: {str(e)}", "confidence": 0.1}
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            return {"error": f"Evaluation failed: {str(e)}", "confidence": 0.1}

class MetaCognitionModule:
    def __init__(self):
        self.name = "META"
        self.weights = None
        self.recovery_count = 0

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor system state and adapt behavior"""
        try:
            # Access data from Evaluator
            if input_data["robot_pos"] == input_data["goal_pos"]:
                return {
                "waypoint": input_data["goal_pos"],
                "actions": [],
                "confidence": 1.0
            }
            safety_score = input_data["safety_score"]

            # Recovery logic
            recovery_needed = False
            if "error" in input_data:
                self.recovery_count += 1
                recovery_needed = True
            elif safety_score < 0.5:
                self.recovery_count += 1
                recovery_needed = True

            # Adapt weights based on situation
            new_weights = self.adapt_weights(input_data, recovery_needed)

            return {
                "recovery_needed": recovery_needed,
                "recovery_count": self.recovery_count,
                "weights": new_weights,
                "confidence": 0.7  # Moderate confidence for meta-cognition
            }
        except KeyError as e:
            logger.error(f"Missing input: {str(e)}")
            return {"error": f"Missing input: {str(e)}", "confidence": 0.1}
        except Exception as e:
            logger.error(f"Meta-cognition failed: {str(e)}")
            return {"error": f"Meta-cognition failed: {str(e)}", "confidence": 0.1}

    def adapt_weights(self, input_data: Dict[str, Any], recovery_needed: bool) -> np.ndarray:
        """Dynamically adjust weight matrix based on context"""
        if self.weights is None:
            # Initial weight matrix
            weights = np.array([
                # Perc, Plan, Exec, Eval, Meta
                [1.0,  0.7, -0.3,  0.0,  0.1],  # Perceptor
                [0.7,  1.0,  0.5,  0.1,  0.0],  # Planner
                [-0.3, 0.5,  1.0,  0.8,  0.2],  # Executor
                [0.0,  0.1,  0.8,  1.0,  0.3],  # Evaluator
                [0.1,  0.0,  0.2,  0.3,  1.0]   # Meta
            ])
        else:
            weights = self.weights.copy()

        # Adjust weights based on context
        if recovery_needed:
            # Boost evaluator and meta during recovery
            weights[3, :] *= 1.2  # Evaluator
            weights[4, :] *= 1.3  # Meta
            weights[:, 3] *= 1.2  # Influence to evaluator
            weights[:, 4] *= 1.3  # Influence to meta

        if "progress" in input_data and input_data["progress"] < 0.3:
            # Boost planner when progress is slow
            weights[1, :] *= 1.2  # Planner
            weights[:, 1] *= 1.1  # Influence to planner

        return weights

# ------------------- SAAM Kernel -------------------
class SAAMKernel:
    def __init__(self, modules: list, initial_weights: np.ndarray):
        self.modules = modules
        self.weights = initial_weights
        self.module_names = [m.name for m in modules]
        logger.info(f"SAAM Kernel initialized with modules: {self.module_names}")

    def run_cycle(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a full SAAM cognitive cycle"""
        cycle_data = input_data.copy()
        cycle_output = {}
        module_results = {}
        module_confidences = {}

        # Execute modules in sequence
        for i, module in enumerate(self.modules):
            module_name = module.name
            logger.debug(f"Executing {module_name} module")

            # Prepare module input by combining global data and previous outputs
            module_input = {**cycle_data, **cycle_output}

            # Execute module
            result = module.execute(module_input)
            module_results[module_name] = result
            module_confidences[module_name] = result.get("confidence", 0.5)

            # Merge results into cycle output
            cycle_output.update(result)

            # Handle errors
            if "error" in result:
                logger.warning(f"{module_name} error: {result['error']}")
                if module_name == "EXECUTOR":
                    raise EmergencyStop(result["error"])

        # Update global confidence
        overall_confidence = np.mean(list(module_confidences.values()))
        cycle_output["overall_confidence"] = overall_confidence

        # Add module metrics
        cycle_output["module_usage"] = {name: 1 for name in self.module_names}
        cycle_output["module_confidences"] = module_confidences

        return cycle_output

# ------------------- Benchmark Runner -------------------
class BenchmarkRunner:
    def __init__(self):
        self.results = []
        self.metrics = {
            "success_rate": 0,
            "avg_confidence": 0,
            "module_usage": {},
            "recovery_attempts": 0,
            "total_steps": 0,
            "total_time": 0.0,
            "memory_usage": 0
        }
        logger.info("SAAM Benchmark Runner initialized")

    def run_benchmarks(self):
        for scenario in BENCHMARK_SCENARIOS:
            logger.info(f"\n=== Running Scenario: {scenario.name} ===")
            result = self._run_saam_navigation(scenario)
            self.results.append(result)
            self._update_metrics(result)
            status = "SUCCESS" if result["success"] else "FAIL"
            logger.success(
                f"Scenario {status} in {result['time_elapsed']:.2f}s "
                f"(Steps: {result['steps']}, Confidence: {result.get('avg_confidence', 0):.2f})"
            )

        self._generate_report()

    def _run_saam_navigation(self, scenario) -> Dict:
        start_time = time.time()
        start_mem = psutil.Process().memory_info().rss
        steps = 0
        success = False
        max_steps = 50
        recoveries = 0

        # Create true grid state (for physical simulation)
        true_grid = [['·'] * scenario.grid_size[1] for _ in range(scenario.grid_size[0])]
        for r, c in scenario.obstacles:
            true_grid[r][c] = '■'
        true_grid[scenario.goal_pos[0]][scenario.goal_pos[1]] = 'G'

        # Track true obstacles
        true_obstacles = set(scenario.obstacles)

        # Initialize LangGraph components
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
            obstacle_grid=true_grid
        )

        # Initialize SAAM modules
        perceptor = PerceptorModule(state_manager)
        planner = PlannerModule(waypoint_planner, action_planner)
        executor = ExecutorModule(scenario.grid_size, true_grid)
        evaluator = EvaluatorModule()
        meta = MetaCognitionModule()

        # Create SAAM kernel with weight matrix
        initial_weights = np.array([
            # Perc, Plan, Exec, Eval, Meta
            [1.0,  0.7, -0.3,  0.0,  0.1],  # Perceptor
            [0.7,  1.0,  0.5,  0.1,  0.0],  # Planner
            [-0.3, 0.5,  1.0,  0.8,  0.2],  # Executor
            [0.0,  0.1,  0.8,  1.0,  0.3],  # Evaluator
            [0.1,  0.0,  0.2,  0.3,  1.0]   # Meta
        ])
        kernel = SAAMKernel(
            [perceptor, planner, executor, evaluator, meta],
            initial_weights
        )

        # Set true robot state
        true_robot_pos = list(scenario.start_pos)
        true_robot_facing = scenario.start_facing

        # Initialize input data for first cycle
        input_data = {
            "goal_pos": tuple(scenario.goal_pos),
            "grid_size": scenario.grid_size,
            "true_robot_pos": true_robot_pos,
            "true_robot_facing": true_robot_facing,
            "scenario": scenario.name
        }

        try:
            while steps < max_steps and recoveries < 5:

                steps += 1

                # Simulate sensors with optional error
                sensor_data = {
                    'front': max(10, random.gauss(100, 30)),
                    'left': max(10, random.gauss(100, 30)),
                    'right': max(10, random.gauss(100, 30))
                }
                if scenario.sensor_error_range:
                    for key in sensor_data:
                        sensor_data[key] = max(10, sensor_data[key] +
                                             random.uniform(*scenario.sensor_error_range))

                # Add sensor data to input
                input_data["sensor_data"] = sensor_data

                try:
                    # Run SAAM cognitive cycle
                    cycle_output = kernel.run_cycle(input_data)
                    overall_confidence = cycle_output.get("overall_confidence", 0.5)

                    # Check if goal reached
                    if true_robot_pos == list(scenario.goal_pos):
                        success = True
                        break

                    # Handle executor results
                    if "new_robot_pos" in cycle_output:
                        true_robot_pos = list(cycle_output["new_robot_pos"])
                        true_robot_facing = cycle_output["new_robot_facing"]

                    # Update state manager with new true position
                    old_pos = state_manager.robot_position
                    new_pos = true_robot_pos

                    # Clear old position if not obstacle
                    if state_manager.obstacle_grid[old_pos[0]][old_pos[1]] not in ['■', 'G']:
                        state_manager.obstacle_grid[old_pos[0]][old_pos[1]] = '·'

                    # Update state manager position
                    state_manager.robot_position = new_pos
                    state_manager.robot_facing = true_robot_facing

                    # Set robot marker at new position
                    if state_manager.obstacle_grid[new_pos[0]][new_pos[1]] != '■':
                        state_manager.obstacle_grid[new_pos[0]][new_pos[1]] = (
                            state_manager._get_direction_arrow()
                        )


                    # Ensure goal marker is set
                    gr, gc = scenario.goal_pos
                    state_manager.obstacle_grid[gr][gc] = 'G'

                    # Update input data for next cycle
                    input_data.update({
                        "true_robot_pos": true_robot_pos,
                        "true_robot_facing": true_robot_facing,
                        "overall_confidence": overall_confidence
                    })

                    # Handle meta-cognition recovery
                    if "recovery_needed" in cycle_output and cycle_output["recovery_needed"]:
                        recoveries += 1
                        self.metrics["recovery_attempts"] += 1
                        logger.warning(f"Recovery #{recoveries} at step {steps}")

                        # Reset to last known good state
                        true_robot_pos = list(state_manager.robot_position)
                        input_data["true_robot_pos"] = true_robot_pos

                        # Update kernel weights
                        if "weights" in cycle_output:
                            kernel.weights = cycle_output["weights"]
                            logger.info(f"Updated weights to:\n{kernel.weights}")

                except EmergencyStop as e:
                    recoveries += 1
                    self.metrics["recovery_attempts"] += 1
                    logger.warning(f"Recovery #{recoveries} at step {steps}: {str(e)}")
                    # Reset to last known good state
                    true_robot_pos = list(state_manager.robot_position)
                    input_data["true_robot_pos"] = true_robot_pos
                    continue
                except Exception as e:
                    logger.warning(f"Step {steps} error: {str(e)}")
                    continue

            return {
                "scenario": scenario.name,
                "success": success,
                "steps": steps,
                "time_elapsed": time.time() - start_time,
                "avg_confidence": input_data.get("overall_confidence", 0),
                "final_position": true_robot_pos,
                "goal_position": scenario.goal_pos,
                "recoveries": recoveries,
                "memory_used": psutil.Process().memory_info().rss - start_mem,
                "module_usage": cycle_output.get("module_usage", {}) if 'cycle_output' in locals() else {}
            }

        except Exception as e:
            logger.error(f"Scenario failed: {str(e)}")
            return {
                "scenario": scenario.name,
                "success": False,
                "error": str(e),
                "steps": steps,
                "time_elapsed": time.time() - start_time,
                "recoveries": recoveries,
                "memory_used": psutil.Process().memory_info().rss - start_mem
            }

    def _update_metrics(self, result: Dict):
        self.metrics["success_rate"] += int(result["success"])
        self.metrics["avg_confidence"] += result.get("avg_confidence", 0)
        self.metrics["total_steps"] += result.get("steps", 0)
        self.metrics["total_time"] += result.get("time_elapsed", 0)
        self.metrics["memory_usage"] += result.get("memory_used", 0)
        self.metrics["recovery_attempts"] += result.get("recoveries", 0)

        # Update module usage
        module_usage = result.get("module_usage", {})
        for module, count in module_usage.items():
            self.metrics["module_usage"][module] = self.metrics["module_usage"].get(module, 0) + count

    def _generate_report(self):
        print("\n=== SAAM NAVIGATION BENCHMARK RESULTS ===")
        total = len(self.results)
        successes = self.metrics["success_rate"]

        print(f"\nSummary ({total} scenarios):")
        print(f"- Success Rate: {successes}/{total} ({successes/total:.1%})")
        print(f"- Avg Confidence: {self.metrics['avg_confidence']/total:.2f}")
        print(f"- Recovery Attempts: {self.metrics['recovery_attempts']}")
        print(f"- Avg Steps: {self.metrics['total_steps']/total:.1f}")
        print(f"- Avg Time/Scenario: {self.metrics['total_time']/total:.2f}s")
        print(f"- Avg Memory/Scenario: {self.metrics['memory_usage']/total/1024:.1f} KB")

        print("\nModule Usage:")
        for module, count in sorted(self.metrics["module_usage"].items(), key=lambda x: x[1], reverse=True):
            print(f"- {module}: {count}")

        print("\nScenario Details:")
        for result in self.results:
            status = "PASS" if result["success"] else "FAIL"
            complexity = next(
                (s.complexity for s in BENCHMARK_SCENARIOS if s.name == result['scenario']),
                "N/A"
            )
            print(f"\n{result['scenario']} [{status}] ({complexity})")
            print(f"Steps: {result['steps']}")
            print(f"Time: {result['time_elapsed']:.2f}s")
            print(f"Memory: {result.get('memory_used', 0)/1024:.1f} KB")
            print(f"Confidence: {result.get('avg_confidence', 0):.2f}")
            print(f"Recoveries: {result.get('recoveries', 0)}")
            if not result["success"]:
                print(f"Error: {result.get('error', 'Unknown')}")
            print(f"Final: {result['final_position']} → Goal: {result['goal_position']}")

if __name__ == "__main__":
    logger.add("saam_benchmark.log", rotation="10 MB", level="INFO")
    runner = BenchmarkRunner()
    runner.run_benchmarks()