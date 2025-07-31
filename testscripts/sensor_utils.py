#!/usr/bin/env python3
import math

def calculate_sensor_readings(robot_row, robot_col, robot_facing, obstacles):
    """Generate realistic sensor readings based on robot position and obstacles"""
    sensor_readings = {'front': 1000.0, 'left': 1000.0, 'right': 1000.0}

    # Directions for sensor checks
    directions = {
        "UP": [(-1, 0), (0, -1), (0, 1)],    # Front, Left, Right
        "DOWN": [(1, 0), (0, 1), (0, -1)],
        "LEFT": [(0, -1), (1, 0), (-1, 0)],
        "RIGHT": [(0, 1), (-1, 0), (1, 0)]
    }

    # Get offsets for current facing
    sensor_offsets = directions.get(robot_facing, directions["UP"])

    # Calculate distance to nearest obstacle in each direction
    for sensor_idx, offset in enumerate(sensor_offsets):
        dr, dc = offset
        distance = 0
        r, c = robot_row, robot_col

        while True:
            distance += 1
            r += dr
            c += dc

            # Check if out of bounds
            if r < 0 or r >= 10 or c < 0 or c >= 10:  # Assuming max grid size 10x10
                break

            # Check if obstacle
            if (r, c) in obstacles:
                # Calculate distance in cm (30cm per cell)
                sensor_distance = distance * 30

                # Add sensor noise
                if sensor_idx == 0:  # Front sensor
                    sensor_readings['front'] = sensor_distance
                elif sensor_idx == 1:  # Left sensor
                    sensor_readings['left'] = sensor_distance
                elif sensor_idx == 2:  # Right sensor
                    sensor_readings['right'] = sensor_distance
                break

    return sensor_readings