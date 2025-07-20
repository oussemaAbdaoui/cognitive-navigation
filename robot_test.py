#!/usr/bin/env python3
import json
import time
import math
import sys
from robot_kernel import RobotKernel

# Constants
MAX_SENSOR_RANGE = 300  # cm
CM_PER_CELL = 4

def calculate_sensor_readings(robot_x, robot_y, robot_theta, obstacles, max_range=MAX_SENSOR_RANGE):
    """Calculate sensor readings based on robot position and map obstacles"""
    front_dist = max_range
    left_dist = max_range
    right_dist = max_range

    # Convert robot position to grid coordinates (FIXED: use rounding instead of int)
    grid_x = round(robot_x / CM_PER_CELL)
    grid_y = round(robot_y / CM_PER_CELL)

    # Calculate sensor angles (0° is front, 90° is left, 270° is right)
    angles = {
        'front': robot_theta % 360,
        'left': (robot_theta + 90) % 360,
        'right': (robot_theta + 270) % 360
    }

    # Check for obstacles in each direction
    for obs_x, obs_y in obstacles:
        # Calculate distance to obstacle
        dx = obs_x - grid_x
        dy = obs_y - grid_y
        distance = math.sqrt(dx**2 + dy**2) * CM_PER_CELL

        if distance > max_range:
            continue

        # Calculate angle to obstacle relative to robot orientation
        angle_to_obs = math.degrees(math.atan2(dy, dx)) % 360
        relative_angle = (angle_to_obs - robot_theta) % 360

        # Determine which sensor would detect this obstacle
        if distance < front_dist and (relative_angle < 30 or relative_angle > 330):
            front_dist = distance
        if distance < left_dist and 60 < relative_angle < 120:
            left_dist = distance
        if distance < right_dist and 240 < relative_angle < 300:
            right_dist = distance

    return {
        'front': min(front_dist, max_range),
        'left': min(left_dist, max_range),
        'right': min(right_dist, max_range)
    }

def run_dynamic_simulation(json_file, output_file):
    # Setup dual output
    class MultiOutput:
        def __init__(self, *outputs):
            self.outputs = outputs

        def write(self, message):
            for output in self.outputs:
                output.write(message)

        def flush(self):
            for output in self.outputs:
                if hasattr(output, 'flush'):
                    output.flush()

    log_file = open(output_file, 'w')
    original_stdout = sys.stdout
    sys.stdout = MultiOutput(original_stdout, log_file)

    try:
        with open(json_file) as f:
            test_data = json.load(f)

        # Create kernel
        kernel = RobotKernel(
            target=tuple(test_data['target']),
            cm_per_cell=CM_PER_CELL,
            max_range=MAX_SENSOR_RANGE,
            emergency_threshold=1,
            model_name="mistralai/mistral-7b-instruct:free",
            temperature=0.0
        )

        # Set initial position
        start_x, start_y, start_theta = test_data['start']
        kernel.x = start_x * CM_PER_CELL
        kernel.y = start_y * CM_PER_CELL
        kernel.theta = start_theta
        kernel.grid_x = start_x
        kernel.grid_y = start_y

        print(f"Starting dynamic simulation: {test_data.get('name', 'Complex Navigation Test')}")
        print(f"Target: {test_data['target']}")
        print(f"Obstacles: {len(test_data['obstacles'])}")
        print(f"Grid size: {test_data['grid_width']}x{test_data['grid_height']} cells")

        # Robot physics constants (FIXED: match kernel parameters)
        MAX_SPEED = 20.0  # cm/s
        WHEEL_BASE = 20.0  # cm (distance between wheels) - MUST match kernel's L=20
        DT = 1.0  # time step in seconds

        position_history = []
        step_count = 0
        max_steps = 200

        # Main simulation loop
        while step_count < max_steps:
            step_count += 1

            # Calculate sensor readings based on current position
            sensors = calculate_sensor_readings(
                kernel.x, kernel.y, kernel.theta,
                test_data['obstacles'],
                MAX_SENSOR_RANGE
            )

            print(f"\n--- Step {step_count} ---")
            print(f"Position: ({kernel.grid_x}, {kernel.grid_y}) Orientation: {kernel.theta:.2f}°")
            print(f"Sensors: F={sensors['front']:.1f}cm, L={sensors['left']:.1f}cm, R={sensors['right']:.1f}cm")

            # Run kernel processing
            start_time = time.time()
            left_speed, right_speed = kernel.run_step(sensors)
            response_time = time.time() - start_time

            print(f"Motors: L={left_speed}, R={right_speed}")
            print(f"Sub-goal: {kernel.sub_goal}")
            print(f"Loop detected: {kernel.loop_detected}")
            print(f"Obstacles mapped: {len(kernel.obstacle_map)}")
            print(f"LLM Response Time: {response_time:.2f}s")

            # Save position for visualization
            position_history.append((kernel.grid_x, kernel.grid_y))

            # Update robot position based on motor speeds
            left_velocity = (left_speed / 100.0) * MAX_SPEED
            right_velocity = (right_speed / 100.0) * MAX_SPEED

            # Calculate linear and angular velocity
            v = (left_velocity + right_velocity) / 2.0
            w = (right_velocity - left_velocity) / WHEEL_BASE  # rad/s

            # Update orientation
            kernel.theta = (kernel.theta + math.degrees(w * DT)) % 360

            # Update position
            theta_rad = math.radians(kernel.theta)
            kernel.x += v * math.cos(theta_rad) * DT
            kernel.y += v * math.sin(theta_rad) * DT

            # Update grid position (FIXED: use rounding instead of int)
            kernel.grid_x = round(kernel.x / CM_PER_CELL)
            kernel.grid_y = round(kernel.y / CM_PER_CELL)

            # Check if target reached
            target_dist = math.dist([kernel.grid_x, kernel.grid_y], test_data['target'])
            if target_dist < 1.0:
                print("\nTarget reached!")
                break

            # Add visualization
            visualize_current_state(kernel, test_data['obstacles'], test_data['target'], position_history)

        kernel.http_client.close()
        print("\nSimulation complete!")
        print(f"Total steps: {step_count}")
        print(f"Final position: ({kernel.grid_x}, {kernel.grid_y})")
        print(f"Distance to target: {target_dist:.2f} cells")

    finally:
        sys.stdout = original_stdout
        log_file.close()

def visualize_current_state(kernel, obstacles, target, history, grid_size=25):
    """Enhanced text-based visualization with better symbols and larger view"""
    grid = [['·' for _ in range(grid_size)] for _ in range(grid_size)]
    origin_x = kernel.grid_x - grid_size // 2
    origin_y = kernel.grid_y - grid_size // 2

    # Directional arrows for robot (more precise)
    arrows = {
        0: '→', 15: '↗', 30: '➡', 45: '↗', 60: '↑↗',
        75: '↑', 90: '↑', 105: '↑', 120: '↑↖', 135: '↖',
        150: '⬅', 165: '↖', 180: '←', 195: '↙', 210: '⬅',
        225: '↙', 240: '↓↙', 255: '↓', 270: '↓', 285: '↓',
        300: '↓↘', 315: '↘', 330: '➡', 345: '↗'
    }

    # Mark obstacles
    for (obs_x, obs_y) in obstacles:
        dx = obs_x - origin_x
        dy = obs_y - origin_y
        if 0 <= dx < grid_size and 0 <= dy < grid_size:
            grid[dy][dx] = '.'  # Solid block for obstacles

    # Mark detected obstacles
    for obs in kernel.obstacle_map:
        dx, dy = obs[0] - origin_x, obs[1] - origin_y
        if 0 <= dx < grid_size and 0 <= dy < grid_size:
            if grid[dy][dx] == '·':  # Only mark free spaces
                grid[dy][dx] = 'D'  # Lighter block for detected

    # Mark target
    tgt_dx = target[0] - origin_x
    tgt_dy = target[1] - origin_y
    if 0 <= tgt_dx < grid_size and 0 <= tgt_dy < grid_size:
        grid[tgt_dy][tgt_dx] = '★'  # Star for target

    # Mark position history
    path_chars = ['⓪','①','②','③','④','⑤','⑥','⑦','⑧','⑨','⑩']
    for i, (hx, hy) in enumerate(history[-15:]):
        dx, dy = hx - origin_x, hy - origin_y
        if 0 <= dx < grid_size and 0 <= dy < grid_size:
            if grid[dy][dx] == '·':  # Only mark free spaces
                idx = min(i, len(path_chars)-1)
                grid[dy][dx] = path_chars[idx]

    # Mark robot position
    dx = kernel.grid_x - origin_x
    dy = kernel.grid_y - origin_y
    if 0 <= dx < grid_size and 0 <= dy < grid_size:
        angle = kernel.theta % 360
        closest_angle = min(arrows.keys(), key=lambda x: min(abs(x - angle), 360 - abs(x - angle)))
        grid[dy][dx] = arrows[closest_angle]

    # Print the grid
    print("\n" + "="*50)
    print(f"=== LOCAL VIEW (Robot at Center) ===")
    print(f"Target: ({target[0]}, {target[1]}) | Robot: ({kernel.grid_x}, {kernel.grid_y})")
    print(f"Orientation: {kernel.theta:.1f}° | Steps: {len(history)}")
    print("="*50)

    for row in reversed(grid):
        print(" ".join(row))

    # Compass and legend
    print("\nLEGEND:")
    print("→↑←↓ : Robot direction  ▓ : Obstacle")
    print("░ : Detected obstacle  ★ : Target")
    print("⓪-⑩ : Recent path (⓪ = newest)")
    print(f"Detected obstacles: {len(kernel.obstacle_map)}")
    print(f"Loop detected: {'Yes' if kernel.loop_detected else 'No'}")
    print(f"Sub-goal: {kernel.sub_goal}")
    print("="*50)

if __name__ == "__main__":
    run_dynamic_simulation("test_map.json", "simu.txt")