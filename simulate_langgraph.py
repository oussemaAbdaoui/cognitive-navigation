#!/usr/bin/env python3
import json
import math
import os
from langgraph.graph import StateGraph
from langgraph_nodes import StateManager, WaypointPlanner, ActionPlanner
from typing import TypedDict, Optional, List, Dict, Any
CM_PER_CELL = 30
MAX_SENSOR_RANGE = 350

class RobotState(TypedDict):
    sensor_data: Dict[str, float]
    robot_position: List[int]
    robot_facing: str
    goal_position: List[int]
    visual_grid: str
    waypoint: Optional[List[int]]
    actions: Optional[List[Dict[str, Any]]]
# Constants (must match test_saam.py)

def calculate_sensor_readings(robot_row, robot_col, robot_facing, obstacles):
    """Identical to test_saam.py's implementation"""
    facing_to_theta = {
        "RIGHT": 0,
        "UP": 90,
        "LEFT": 180,
        "DOWN": 270
    }
    theta = facing_to_theta[robot_facing]

    sensor_dists = {'front': MAX_SENSOR_RANGE, 'left': MAX_SENSOR_RANGE, 'right': MAX_SENSOR_RANGE}

    for obs_row, obs_col in obstacles:
        dr = obs_row - robot_row
        dc = obs_col - robot_col
        real_distance = math.sqrt(dr**2 + dc**2) * CM_PER_CELL

        if real_distance > MAX_SENSOR_RANGE or real_distance < 1:
            continue

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

def load_test_map(json_file):
    """Load test map data and convert coordinates"""
    with open(json_file) as f:
        data = json.load(f)

    # Convert to grid coordinates (row, col)
    start_col, start_row, start_theta = data['start']
    target_col, target_row = data['target']

    obstacles = []
    for obs in data['obstacles']:
        obs_col, obs_row = obs
        obstacles.append((obs_row, obs_col))

    return {
        'grid_size': (data['grid_height'], data['grid_width']),
        'start_position': (start_row, start_col),
        'start_facing': {0: "RIGHT", 90: "UP", 180: "LEFT", 270: "DOWN"}[start_theta],
        'target_position': (target_row, target_col),
        'obstacles': obstacles
    }

def create_agents(map_data):
    """Instantiate agents with map configuration"""
    state_manager = StateManager(
        target=map_data['target_position'],
        grid_size=map_data['grid_size'],
        start_position=map_data['start_position'],
        start_facing=map_data['start_facing'],
        cm_per_cell=CM_PER_CELL,
        max_range=MAX_SENSOR_RANGE,
        initial_obstacles=map_data['obstacles']
    )

    waypoint_planner = WaypointPlanner()
    action_planner = ActionPlanner(cm_per_cell=CM_PER_CELL)

    return state_manager, waypoint_planner, action_planner

def create_nodes(state_manager, waypoint_planner, action_planner):
    """Create LangGraph nodes with agent instances"""
    def state_manager_node(state):
        sensor_data = state.get("sensor_data", {})
        visual_grid = state_manager.process_sensor_data(sensor_data)
        return {
            "robot_position": state_manager.robot_position,
            "robot_facing": state_manager.robot_facing,
            "goal_position": state_manager.goal_position,
            "visual_grid": visual_grid
        }

    def waypoint_planner_node(state):
        waypoint = waypoint_planner.plan_waypoint(
            state["visual_grid"],
            state["robot_position"],
            state["goal_position"]
        )
        return {**state, "waypoint": waypoint}

    def action_planner_node(state):
        actions = action_planner.plan_actions(
            state["waypoint"],
            state["robot_position"],
            state["robot_facing"]
        )
        return {**state, "actions": actions}

    return state_manager_node, waypoint_planner_node, action_planner_node

def build_graph(state_manager_node, waypoint_planner_node, action_planner_node):
    """Compile LangGraph pipeline"""
    graph = StateGraph(RobotState)
    graph.add_node("SensorUpdate", state_manager_node)
    graph.add_node("WaypointPlan", waypoint_planner_node)
    graph.add_node("ActionPlan", action_planner_node)

    graph.set_entry_point("SensorUpdate")
    graph.add_edge("SensorUpdate", "WaypointPlan")
    graph.add_edge("WaypointPlan", "ActionPlan")

    return graph.compile()

def run_simulation(json_file):
    """Run complete simulation for a test map"""
    # 1. Load environment
    map_data = load_test_map(json_file)

    # 2. Create agents
    state_manager, wp_planner, action_planner = create_agents(map_data)

    # 3. Create graph nodes
    sm_node, wp_node, ap_node = create_nodes(state_manager, wp_planner, action_planner)

    # 4. Compile graph
    compiled_graph = build_graph(sm_node, wp_node, ap_node)

    # 5. Simulation state
    state = {
        "robot_position": list(map_data['start_position']),
        "robot_facing": map_data['start_facing']
    }

    print(f"\n{'#' * 40}")
    print(f"Starting Simulation: {json_file}")
    print(f"Start: {state['robot_position']} facing {state['robot_facing']}")
    print(f"Target: {map_data['target_position']}")
    print(f"Obstacles: {map_data['obstacles']}")
    print(f"{'#' * 40}\n")

    # 6. Simulation loop
    for step in range(20):  # Max 20 steps
        # Calculate sensor values from current position
        sensor_data = calculate_sensor_readings(
            state['robot_position'][0],
            state['robot_position'][1],
            state['robot_facing'],
            map_data['obstacles']
        )

        # Execute graph with sensor data
        state = compiled_graph.invoke({"sensor_data": sensor_data, **state})

        # Log results
        print(f"\n--- Step {step+1} ---")
        print(f"Position: {state['robot_position']} | Facing: {state['robot_facing']}")
        print(f"Sensors: F={sensor_data['front']:.1f}cm, L={sensor_data['left']:.1f}cm, R={sensor_data['right']:.1f}cm")
        print(f"Waypoint: {state.get('waypoint', 'None')}")
        print(f"Actions: {state.get('actions', [])}")

        # Check termination
        if state['robot_position'] == list(map_data['target_position']):
            print("\n🎯 TARGET REACHED!")
            break

    print("\nSimulation complete!")
    print(f"Final position: {state['robot_position']}")

if __name__ == "__main__":
    for i in range(1, 7):
        run_simulation(f"test_map{i}.json")