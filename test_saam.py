#!/usr/bin/env python3
import json
import time
import math
import sys
from kernel_saam import RobotKernel

# Constants
MAX_SENSOR_RANGE = 350  # cm
CM_PER_CELL = 30

def calculate_sensor_readings(robot_row, robot_col, robot_facing, obstacles):
    # Convert facing to degrees for sensor calculation
    facing_to_theta = {
        "RIGHT": 0,
        "UP": 90,
        "LEFT": 180,
        "DOWN": 270
    }
    theta = facing_to_theta[robot_facing]

    sensor_dists = {
        'front': MAX_SENSOR_RANGE,
        'left': MAX_SENSOR_RANGE,
        'right': MAX_SENSOR_RANGE
    }
    for obs_row, obs_col in obstacles:
        dr = obs_row - robot_row
        dc = obs_col - robot_col
        real_distance = math.sqrt(dr**2 + dc**2) * CM_PER_CELL

        if real_distance > MAX_SENSOR_RANGE or real_distance < 1:
            continue

        # Corrected directional logic:
        if theta == 0:  # RIGHT
            if dr == 0 and dc > 0:  # Front
                sensor_dists['front'] = min(sensor_dists['front'], real_distance)
            elif dr < 0 and dc == 0:  # Left (north)
                sensor_dists['left'] = min(sensor_dists['left'], real_distance)
            elif dr > 0 and dc == 0:  # Right (south)
                sensor_dists['right'] = min(sensor_dists['right'], real_distance)

        elif theta == 90:  # UP
            if dr < 0 and dc == 0:  # Front
                sensor_dists['front'] = min(sensor_dists['front'], real_distance)
            elif dr == 0 and dc < 0:  # Left
                sensor_dists['left'] = min(sensor_dists['left'], real_distance)
            elif dr == 0 and dc > 0:  # Right
                sensor_dists['right'] = min(sensor_dists['right'], real_distance)

        elif theta == 180:  # LEFT
            if dr == 0 and dc < 0:  # Front
                sensor_dists['front'] = min(sensor_dists['front'], real_distance)
            elif dr > 0 and dc == 0:  # Left
                sensor_dists['left'] = min(sensor_dists['left'], real_distance)
            elif dr < 0 and dc == 0:  # Right
                sensor_dists['right'] = min(sensor_dists['right'], real_distance)

        elif theta == 270:  # DOWN
            if dr > 0 and dc == 0:  # Front
                sensor_dists['front'] = min(sensor_dists['front'], real_distance)
            elif dr == 0 and dc > 0:  # Left
                sensor_dists['left'] = min(sensor_dists['left'], real_distance)
            elif dr == 0 and dc < 0:  # Right
                sensor_dists['right'] = min(sensor_dists['right'], real_distance)
    return sensor_dists

class DualOutput:
    def __init__(self, *outputs):
        self.outputs = outputs

    def write(self, message):
        for output in self.outputs:
            output.write(message)

    def flush(self):
        for output in self.outputs:
            if hasattr(output, 'flush'):
                output.flush()

def run_simulation(json_file, output_file):
    with open(output_file, 'w') as log_file:
        sys.stdout = DualOutput(sys.stdout, log_file)

        try:
            with open(json_file) as f:
                test_data = json.load(f)

            # Get grid dimensions from JSON
            grid_width = test_data['grid_width']
            grid_height = test_data['grid_height']
            grid_size = (grid_height, grid_width)  # (rows, cols)

            # Extract positions (using grid coordinates directly)
            start_info = test_data['start']  # [x, y, theta]
            target_info = test_data['target']  # [x, y]
            obstacles_info = test_data['obstacles']  # list of [x, y]

            # Convert start info
            start_col, start_row, start_theta = start_info
            start_position = (start_row, start_col)

            # Convert target info
            target_col, target_row = target_info
            target_position = (target_row, target_col)

            # Convert obstacles
            grid_obstacles = []
            for obs in obstacles_info:
                obs_col, obs_row = obs
                grid_obstacles.append((obs_row, obs_col))

            # Convert start theta to facing direction
            theta_to_facing = {
                0: "RIGHT",
                90: "UP",
                180: "LEFT",
                270: "DOWN"
            }
            start_facing = theta_to_facing[start_theta]

            # Initialize kernel
            kernel = RobotKernel(
                target=target_position,
                grid_size=grid_size,
                start_position=start_position,
                start_facing=start_facing,
                cm_per_cell=CM_PER_CELL,
                max_range=MAX_SENSOR_RANGE,
                model_name="mistralai/mistral-7b-instruct:free",
                        initial_obstacles=grid_obstacles  # Add this line

            )

            print(f"Starting SAAM simulation: Navigation Test")
            print(f"Grid size: {grid_size[0]} rows x {grid_size[1]} columns")
            print(f"Start: row={start_row}, col={start_col} facing={start_facing}")
            print(f"Target: row={target_row}, col={target_col}")
            print(f"Obstacles: {grid_obstacles}")

            position_history = []
            step_count, max_steps = 0, 20

            while step_count < max_steps:
                step_count += 1
                sensors = calculate_sensor_readings(
                    kernel.robot_position[0],  # current row
                    kernel.robot_position[1],  # current col
                    kernel.robot_facing,
                    grid_obstacles
                )

                print(f"\n--- Step {step_count} ---")
                print(f"Position: {kernel.robot_position} Facing: {kernel.robot_facing}")
                print(f"Sensors: F={sensors['front']:.1f}cm, L={sensors['left']:.1f}cm, R={sensors['right']:.1f}cm")
                print(f"Waypoint: {kernel.current_waypoint}")
                print(f"Path: {list(kernel.waypoint_path)}")

                start_time = time.time()
                action = kernel.run_step(sensors)
                response_time = time.time() - start_time

                if action is None:
                    print("No action this step. Robot stopped or waiting.")
                else:
                    if action["type"] == "TURN":
                        print(f"Turn {action['direction']}: L={action['left_speed']}, R={action['right_speed']}")
        # Assume fixed duration for turn, e.g., 1s
                        time.sleep(1)
                    elif action["type"] == "MOVE":
                        print(f"Move forward at speed {action['speed']} cm/s for {action['time']} seconds")
                        print(f"Motors: L={action['left_speed']}, R={action['right_speed']}")
                        time.sleep(action['time'])  # Simulate movement duration
                    else:
                        print("Unknown action")

                print(f"LLM Response Time: {response_time:.2f}s")

                # Record position
                position_history.append(tuple(kernel.robot_position))

                # Check target reached
                if kernel.robot_position == [target_row, target_col]:
                    print("\nTarget reached!")
                    break

                # Visualize every 3 steps
                if step_count % 3 == 0:
                    visualize_grid(kernel, grid_obstacles, target_position, step_count)

            print("\nSimulation complete!")
            print(f"Total steps: {step_count}")
            print(f"Final position: {kernel.robot_position}")
            print(f"Position history: {position_history}")
            print("Obstacle Map:")
            for r in range(kernel.grid_rows):
                print(' '.join(str(cell) for cell in kernel.obstacle_map[r]))

        # Enhanced emergency handling
            if any(dist < kernel.emergency_threshold for dist in sensors.values()):
                print("! EMERGENCY STOP !")
        finally:
            sys.stdout = sys.__stdout__

def visualize_grid(kernel, obstacles, target, step_count):
    """Text-based visualization of the entire grid"""
    grid_rows, grid_cols = kernel.grid_rows, kernel.grid_cols
    grid = [['·' for _ in range(grid_cols)] for _ in range(grid_rows)]
    robot_row, robot_col = kernel.robot_position
    target_row, target_col = target

    # Directional representation
    facing_char = {
        "UP": "↑",
        "RIGHT": "→",
        "DOWN": "↓",
        "LEFT": "←"
    }

    # Place obstacles
    for (obs_row, obs_col) in obstacles:
        if 0 <= obs_row < grid_rows and 0 <= obs_col < grid_cols:
            grid[obs_row][obs_col] = '■'

    # Place blocked waypoints
    for (block_row, block_col) in kernel.blocked_waypoints:
        if 0 <= block_row < grid_rows and 0 <= block_col < grid_cols:
            grid[block_row][block_col] = 'X'

    # Place target
    if 0 <= target_row < grid_rows and 0 <= target_col < grid_cols:
        grid[target_row][target_col] = '★'

    # Place current waypoint
    if kernel.current_waypoint:
        wp_row, wp_col = kernel.current_waypoint
        if 0 <= wp_row < grid_rows and 0 <= wp_col < grid_cols:
            grid[wp_row][wp_col] = 'W'

    # Place robot
    if 0 <= robot_row < grid_rows and 0 <= robot_col < grid_cols:
        grid[robot_row][robot_col] = facing_char.get(kernel.robot_facing, 'R')

    # Print grid
    print("\n" + "="*(grid_cols*2+10))
    print(f"Navigation Grid (Step {step_count})")
    print("="*(grid_cols*2+10))
    for r in range(grid_rows):
        # Print row index
        print(f"{r:2d} ", end="")
        for c in range(grid_cols):
            print(grid[r][c], end=" ")
        print()
    # Print column indices
    print("    ", end="")
    for c in range(grid_cols):
        print(f"{c:2d}", end=" ")
    print()
    print("="*(grid_cols*2+10))
    print(f"Robot: ({robot_row}, {robot_col}) {kernel.robot_facing}")
    print(f"Target: ({target_row}, {target_col})")
    print(f"Waypoint: {kernel.current_waypoint}")
    print("="*(grid_cols*2+10))

if __name__ == "__main__":
    for i in range(1, 7):
        json_file = f"test_map{i}.json"
        log_file = f"saam_simu_log_{i}.txt"
        print(f"\n{'#' * 40}")
        print(f"Starting simulation for {json_file}")
        print(f"Logging to {log_file}")
        print(f"{'#' * 40}\n")

        run_simulation(json_file, log_file)

        print(f"\n{'#' * 40}")
        print(f"Completed simulation for {json_file}")
        print(f"{'#' * 40}\n")