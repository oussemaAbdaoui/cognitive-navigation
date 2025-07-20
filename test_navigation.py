#!/usr/bin/env python3
import os
from splited_sys import StateManager
import time

def test_state_manager_with_api():
    # Verify API key is set
    if "OPENROUTER_API_KEY" not in os.environ:
        print("❌ OPENROUTER_API_KEY environment variable not set")
        return

    # Test parameters
    target = (5, 5)
    grid_size = (6, 6)
    start_pos = (0, 0)
    start_facing = "RIGHT"
    initial_obstacles = [(1, 1), (2, 2)]

    print("\n🚀 Creating StateManager instance...")
    try:
        sm = StateManager(
            target=target,
            grid_size=grid_size,
            start_position=start_pos,
            start_facing=start_facing,
            initial_obstacles=initial_obstacles
        )
        print("✅ StateManager created successfully")
    except Exception as e:
        print(f"❌ Failed to create StateManager: {e}")
        return

    # Print initial state
    print("\n📋 Initial Grid:")
    print(sm.get_visual_grid())
    print(f"Robot Position: {sm.robot_position}")
    print(f"Robot Facing: {sm.robot_facing}")
    print(f"Goal Position: {sm.goal_position}")

    # Test cases with different sensor inputs
    test_cases = [
        {
            "name": "Front obstacle detected",
            "sensor_values": {"front": 30, "left": 90, "right": 90},
            "description": "Should detect obstacle directly in front"
        },
        {
            "name": "Left obstacle detected",
            "sensor_values": {"front": 90, "left": 30, "right": 90},
            "description": "Should detect obstacle to the left"
        },
        {
            "name": "Right obstacle detected",
            "sensor_values": {"front": 90, "left": 90, "right": 30},
            "description": "Should detect obstacle to the right"
        },
        {
            "name": "Multiple obstacles",
            "sensor_values": {"front": 30, "left": 30, "right": 30},
            "description": "Should detect obstacles in all directions"
        },
        {
            "name": "No obstacles",
            "sensor_values": {"front": 90, "left": 90, "right": 90},
            "description": "Should show clear path in all directions"
        }
    ]

    for test_case in test_cases:
        print(f"\n🔍 Running test: {test_case['name']}")
        print(f"Description: {test_case['description']}")
        print(f"Sensor values: {test_case['sensor_values']}")

        try:
            # Add delay to avoid rate limiting
            time.sleep(2)

            # Process sensor data
            print("\n🔄 Processing sensor data...")
            updated_grid = sm.process_sensor_data(test_case["sensor_values"])

            # Print results
            print("\n📊 Updated Grid:")
            print(updated_grid)
            print(f"New Robot Position: {sm.robot_position}")
            print(f"New Robot Facing: {sm.robot_facing}")

        except Exception as e:
            print(f"❌ Error during test execution: {e}")
            continue

        print("✅ Test completed")

if __name__ == "__main__":
    test_state_manager_with_api()