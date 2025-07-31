#!/usr/bin/env python3
import os
from splited_sys import StateManager, WaypointPlanner, ActionPlanner

def main():
    # Setup environment
    os.environ["OPENROUTER_API_KEY"] = "your_api_key_here"  # Replace with actual key

    # Initialize components
    state_mgr = StateManager(
        target=(3, 3),
        grid_size=(5, 5),
        start_position=(0, 0),
        start_facing="RIGHT",
        initial_obstacles=[(0, 1), (1, 1), (2, 3)]
    )

    waypoint_planner = WaypointPlanner()
    action_planner = ActionPlanner(cm_per_cell=30)

    # Simulate sensor data (front, left, right in cm)
    sensor_data = {"front": 30, "left": 60, "right": 120}

    # Process sensor data and update state
    print("=== Processing Sensor Data ===")
    updated_grid = state_mgr.process_sensor_data(sensor_data)
    print(f"Updated Grid:\n{updated_grid}")
    print(f"Robot Position: {state_mgr.robot_position}")
    print(f"Robot Facing: {state_mgr.robot_facing}\n")

    # Plan waypoint
    print("=== Planning Waypoint ===")
    waypoint = waypoint_planner.plan_waypoint(
        updated_grid,
        state_mgr.robot_position,
        state_mgr.goal_position
    )
    print(f"Selected Waypoint: {waypoint}\n")

    # Plan actions to waypoint
    print("=== Planning Actions ===")
    actions = action_planner.plan_actions(
        waypoint,
        state_mgr.robot_position,
        state_mgr.robot_facing
    )
    print("Actions:")
    for action in actions:
        print(f"- {action['type']} {action.get('direction', '')} {action.get('cells', '')}")

if __name__ == "__main__":
    main()