#!/usr/bin/env python3
import json
import time
import math
import os
import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Import both kernel implementations
from kernel_saam import RobotKernel as KernelV1
from robot_kernel import RobotKernel as KernelV2

# Constants
MAX_SENSOR_RANGE = 300  # cm
CM_PER_CELL = 10
MAX_SPEED = 20.0  # cm/s
WHEEL_BASE = 15.0  # cm
DT = 1.0  # time step in seconds

def calculate_sensor_readings(robot_x, robot_y, robot_theta, obstacles, max_range=MAX_SENSOR_RANGE):
    """Calculate sensor readings based on robot position and map obstacles"""
    front_dist = max_range
    left_dist = max_range
    right_dist = max_range

    grid_x = int(robot_x / CM_PER_CELL)
    grid_y = int(robot_y / CM_PER_CELL)

    angles = {
        'front': robot_theta % 360,
        'left': (robot_theta + 90) % 360,
        'right': (robot_theta + 270) % 360
    }

    for obs_x, obs_y in obstacles:
        dx = (obs_x - grid_x) * CM_PER_CELL
        dy = (obs_y - grid_y) * CM_PER_CELL
        distance = math.sqrt(dx**2 + dy**2)

        if distance > max_range:
            continue

        angle_to_obs = math.degrees(math.atan2(dy, dx)) % 360
        relative_angle = (angle_to_obs - robot_theta) % 360

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

def run_kernel_simulation(kernel_class, map_data, max_steps=200):
    """Run simulation for a kernel and return metrics"""
    metrics = {
        'steps': 0,
        'reached_target': False,
        'final_distance': 0,
        'llm_calls': 0,
        'llm_time': 0.0,
        'prompt_tokens': 0,
        'response_tokens': 0,
        'loop_detections': 0,
        'stuck_events': 0,
        'path': [],
        'obstacles_mapped': 0,
        'response_times': []
    }
    
    kernel = kernel_class(
        target=tuple(map_data['target']),
        grid_width=map_data['grid_width'],
        grid_height=map_data['grid_height'],
        cm_per_cell=CM_PER_CELL,
        max_range=MAX_SENSOR_RANGE,
        model_name="mistralai/mistral-7b-instruct:free"
    )

    # Set initial position
    start_x, start_y, start_theta = map_data['start']
    kernel.x = start_x * CM_PER_CELL
    kernel.y = start_y * CM_PER_CELL
    kernel.theta = start_theta
    kernel.grid_x = start_x
    kernel.grid_y = start_y
    if hasattr(kernel, 'last_position'):
        kernel.last_position = (start_x, start_y)

    # Main simulation loop
    for step in range(max_steps):
        metrics['steps'] = step + 1
        current_pos = (kernel.grid_x, kernel.grid_y)
        metrics['path'].append(current_pos)
        
        # Calculate sensor readings
        sensors = calculate_sensor_readings(
            kernel.x, kernel.y, kernel.theta,
            map_data['obstacles'],
            MAX_SENSOR_RANGE
        )

        # Run kernel processing
        start_time = time.time()
        left_speed, right_speed = kernel.run_step(sensors)
        process_time = time.time() - start_time
        
        # Update metrics
        metrics['llm_calls'] = kernel.llm_call_count
        metrics['llm_time'] += getattr(kernel, 'llm_total_time', 0)
        metrics['prompt_tokens'] = getattr(kernel, 'llm_prompt_tokens', 0)
        metrics['response_tokens'] = getattr(kernel, 'llm_response_tokens', 0)
        metrics['loop_detections'] = getattr(kernel, 'loop_detection_count', 0)
        metrics['stuck_events'] = getattr(kernel, 'stuck_events', 0)
        metrics['obstacles_mapped'] = len(kernel.obstacle_map)
        metrics['response_times'].append(process_time)

        # Update robot position
        left_velocity = (left_speed / 100.0) * MAX_SPEED
        right_velocity = (right_speed / 100.0) * MAX_SPEED

        v = (left_velocity + right_velocity) / 2.0
        w = (right_velocity - left_velocity) / WHEEL_BASE
        kernel.theta = (kernel.theta + math.degrees(w * DT)) % 360

        theta_rad = math.radians(kernel.theta)
        kernel.x += v * math.cos(theta_rad) * DT
        kernel.y += v * math.sin(theta_rad) * DT
        kernel.grid_x = int(kernel.x / CM_PER_CELL)
        kernel.grid_y = int(kernel.y / CM_PER_CELL)

        # Check target reached
        target_dist = math.dist([kernel.grid_x, kernel.grid_y], map_data['target'])
        if target_dist < 1.0:
            metrics['reached_target'] = True
            metrics['final_distance'] = target_dist
            break

    if not metrics['reached_target']:
        metrics['final_distance'] = math.dist(
            [kernel.grid_x, kernel.grid_y],
            map_data['target']
        )
        
    return metrics

def plot_comparison(results, map_data):
    """Visualize comparison results"""
    # Performance metrics
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Success and steps
    axs[0, 0].bar(['SAAM', 'Enhanced'], 
                 [results['v1']['reached_target'], results['v2']['reached_target']],
                 color=['skyblue', 'lightgreen'])
    axs[0, 0].set_title('Target Reached')
    axs[0, 0].set_ylim(0, 1)
    
    steps = [results['v1']['steps'], results['v2']['steps']]
    axs[0, 1].bar(['SAAM', 'Enhanced'], steps, color=['skyblue', 'lightgreen'])
    axs[0, 1].set_title('Steps Taken')
    axs[0, 1].set_ylabel('Steps')
    
    # Token efficiency
    v1_tokens = results['v1']['prompt_tokens'] + results['v1']['response_tokens']
    v2_tokens = results['v2']['prompt_tokens'] + results['v2']['response_tokens']
    axs[1, 0].bar(['SAAM', 'Enhanced'], [v1_tokens, v2_tokens], color=['skyblue', 'lightgreen'])
    axs[1, 0].set_title('Total Tokens Used')
    axs[1, 0].set_ylabel('Tokens')
    
    # Path visualization
    axs[1, 1].axis('off')
    plt.tight_layout()
    
    # Path comparison
    plt.figure(figsize=(10, 8))
    plt.title('Navigation Path Comparison')
    
    # Plot obstacles
    obstacles = np.array(map_data['obstacles'])
    plt.scatter(obstacles[:, 0], obstacles[:, 1], c='red', marker='s', label='Obstacles')
    
    # Plot paths
    path_v1 = np.array(results['v1']['path'])
    path_v2 = np.array(results['v2']['path'])
    
    plt.plot(path_v1[:, 0], path_v1[:, 1], 'b-', label='SAAM Path')
    plt.plot(path_v2[:, 0], path_v2[:, 1], 'g-', label='Enhanced Path')
    
    # Mark start and target
    start = map_data['start'][:2]
    target = map_data['target']
    plt.plot(start[0], start[1], 'go', markersize=10, label='Start')
    plt.plot(target[0], target[1], 'r*', markersize=15, label='Target')
    
    plt.xlabel('Grid X')
    plt.ylabel('Grid Y')
    plt.grid(True)
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig('kernel_comparison.png')
    plt.show()

def compare_kernels(json_file):
    """Compare performance of both kernels"""
    with open(json_file) as f:
        map_data = json.load(f)
    
    results = {
        'v1': run_kernel_simulation(KernelV1, map_data),
        'v2': run_kernel_simulation(KernelV2, map_data)
    }
    
    # Print results
    print("\n=== SAAM Kernel Results ===")
    print(f"Target reached: {results['v1']['reached_target']}")
    print(f"Steps: {results['v1']['steps']}")
    print(f"Final distance: {results['v1']['final_distance']:.2f} cells")
    print(f"LLM calls: {results['v1']['llm_calls']}")
    print(f"Tokens: {results['v1']['prompt_tokens']} prompt + {results['v1']['response_tokens']} response")
    print(f"Loop detections: {results['v1']['loop_detections']}")
    print(f"Obstacles mapped: {results['v1']['obstacles_mapped']}")
    
    print("\n=== Enhanced Kernel Results ===")
    print(f"Target reached: {results['v2']['reached_target']}")
    print(f"Steps: {results['v2']['steps']}")
    print(f"Final distance: {results['v2']['final_distance']:.2f} cells")
    print(f"LLM calls: {results['v2']['llm_calls']}")
    print(f"Tokens: {results['v2']['prompt_tokens']} prompt + {results['v2']['response_tokens']} response")
    print(f"Loop detections: {results['v2']['loop_detections']}")
    print(f"Stuck events: {results['v2']['stuck_events']}")
    print(f"Obstacles mapped: {results['v2']['obstacles_mapped']}")
    
    # Calculate efficiency metrics
    v1_eff = results['v1']['steps'] / (results['v1']['prompt_tokens'] + results['v1']['response_tokens'] + 1)
    v2_eff = results['v2']['steps'] / (results['v2']['prompt_tokens'] + results['v2']['response_tokens'] + 1)
    
    print("\n=== Efficiency Comparison ===")
    print(f"SAAM Steps/Token: {v1_eff:.4f}")
    print(f"Enhanced Steps/Token: {v2_eff:.4f}")
    print(f"LLM Time Reduction: {100*(results['v1']['llm_time'] - results['v2']['llm_time'])/results['v1']['llm_time']:.1f}%")
    
    # Generate visualizations
    plot_comparison(results, map_data)
    
    return results

if __name__ == "__main__":
    # Create test map if needed
    if not os.path.exists("test_map.json"):
        maze_map = {
            "grid_width": 20,
            "grid_height": 20,
            "start": [0, 0, 0],
            "target": [19, 19],
            "obstacles": [
                [1,0],[2,0],[3,0],[4,0],[5,0],[6,0],[7,0],[8,0],[9,0],[10,0],
                [11,0],[12,0],[13,0],[14,0],[15,0],[16,0],[17,0],[18,0],[19,0],
                [0,1],[1,1],[3,1],[4,1],[5,1],[6,1],[7,1],[8,1],[9,1],[10,1],
                [11,1],[12,1],[13,1],[14,1],[15,1],[16,1],[17,1],[18,1],[19,1],
                [0,2],[2,2],[3,2],[5,2],[7,2],[8,2],[9,2],[10,2],[11,2],[12,2],
                [13,2],[14,2],[15,2],[16,2],[17,2],[19,2],
                [0,3],[1,3],[3,3],[5,3],[7,3],[9,3],[11,3],[13,3],[14,3],[15,3],
                [16,3],[17,3],[18,3],[19,3],
                [0,4],[2,4],[3,4],[5,4],[7,4],[9,4],[11,4],[13,4],[17,4],[19,4],
                [0,5],[1,5],[3,5],[5,5],[7,5],[9,5],[11,5],[13,5],[15,5],[16,5],
                [17,5],[19,5],
                [0,6],[3,6],[5,6],[7,6],[9,6],[11,6],[13,6],[15,6],[19,6],
                [0,7],[1,7],[2,7],[3,7],[5,7],[7,7],[9,7],[11,7],[13,7],[15,7],
                [16,7],[17,7],[18,7],[19,7],
                [0,8],[3,8],[5,8],[7,8],[9,8],[11,8],[13,8],[15,8],[19,8],
                [0,9],[1,9],[2,9],[3,9],[5,9],[7,9],[9,9],[11,9],[13,9],[15,9],
                [17,9],[19,9],
                [0,10],[3,10],[5,10],[7,10],[9,10],[11,10],[13,10],[15,10],[17,10],
                [19,10],
                [0,11],[1,11],[2,11],[3,11],[5,11],[7,11],[9,11],[11,11],[13,11],
                [15,11],[17,11],[19,11],
                [0,12],[3,12],[5,12],[7,12],[9,12],[11,12],[13,12],[15,12],[17,12],
                [19,12],
                [0,13],[1,13],[2,13],[3,13],[5,13],[7,13],[9,13],[11,13],[13,13],
                [15,13],[16,13],[17,13],[19,13],
                [0,14],[3,14],[5,14],[7,14],[9,14],[11,14],[13,14],[15,14],[19,14],
                [0,15],[1,15],[3,15],[5,15],[7,15],[9,15],[11,15],[13,15],[15,15],
                [16,15],[17,15],[18,15],[19,15],
                [0,16],[3,16],[5,16],[7,16],[9,16],[11,16],[13,16],[15,16],[19,16],
                [0,17],[1,17],[2,17],[3,17],[5,17],[7,17],[9,17],[11,17],[13,17],
                [15,17],[16,17],[17,17],[19,17],
                [0,18],[5,18],[7,18],[9,18],[11,18],[13,18],[15,18],[16,18],[17,18],
                [18,18],[19,18],
                [0,19],[1,19],[2,19],[3,19],[4,19],[5,19],[6,19],[7,19],[8,19],
                [9,19],[10,19],[11,19],[12,19],[13,19],[14,19],[15,19],[16,19],
                [17,19],[18,19]
            ]
        }
        with open("test_map.json", "w") as f:
            json.dump(maze_map, f)

    # Run comparison
    results = compare_kernels("test_map.json")
