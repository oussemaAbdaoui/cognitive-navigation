#!/usr/bin/env python3
# Fixed LLM Navigation System with proper spatial reasoning

import json
import os
import re
from typing import Dict, List, Tuple, Optional, Literal
from pydantic import BaseModel, ValidationError, conint
from loguru import logger
import httpx
from dataclasses import dataclass, field
import time
import asyncio
import math

# ------------------- Data Models -------------------
class SensorReading(BaseModel):
    """Raw sensor distance readings"""
    front: float  # Distance in cm
    left: float   # Distance in cm
    right: float  # Distance in cm

class Action(BaseModel):
    """Robot action commands"""
    type: Literal["TURN", "MOVE"]
    direction: Optional[Literal["LEFT", "RIGHT", "FORWARD"]] = None
    cells: Optional[conint(gt=0, le=5)] = None

@dataclass
class GridCell:
    """Individual grid cell state"""
    type: str = "UNKNOWN"  # UNKNOWN, CLEAR, OBSTACLE, ROBOT, GOAL
    confidence: float = 0.0
    last_updated: int = 0

@dataclass
class RobotState:
    """Complete robot state"""
    position: Tuple[int, int] = (0, 0)
    facing: Literal["UP", "DOWN", "LEFT", "RIGHT"] = "UP"
    step_count: int = 0

@dataclass
class InternalMap:
    """Robot's internal spatial map"""
    grid: Dict[Tuple[int, int], GridCell] = field(default_factory=dict)
    bounds: Dict[str, int] = field(default_factory=lambda: {
        "min_row": 0, "max_row": 0, "min_col": 0, "max_col": 0
    })

    def get_cell(self, pos: Tuple[int, int]) -> GridCell:
        return self.grid.get(pos, GridCell())

    def set_cell(self, pos: Tuple[int, int], cell_type: str, confidence: float = 1.0):
        if pos not in self.grid:
            self.grid[pos] = GridCell()
        self.grid[pos].type = cell_type
        self.grid[pos].confidence = confidence
        self._update_bounds(pos)

    def _update_bounds(self, pos: Tuple[int, int]):
        r, c = pos
        self.bounds["min_row"] = min(self.bounds["min_row"], r)
        self.bounds["max_row"] = max(self.bounds["max_row"], r)
        self.bounds["min_col"] = min(self.bounds["min_col"], c)
        self.bounds["max_col"] = max(self.bounds["max_col"], c)

# ------------------- Fixed LLM Navigation System -------------------
class LLMNavigationSystem:
    """Fixed LLM-based navigation with proper spatial reasoning"""

    def __init__(self,
                 target_position: Tuple[int, int],
                 cell_size_cm: int = 30,
                 max_sensor_range_cm: int = 300):
        self._min_call_interval = 1.0  # 1 second rate limit
        self._daily_requests = 0
        self._tokens_used_today = 0
        self.target_position = target_position
        self.cell_size_cm = cell_size_cm
        self.max_sensor_range_cm = max_sensor_range_cm
        self._last_api_call = 0.0

        # Initialize state
        self.robot_state = RobotState()
        self.internal_map = InternalMap()
        self.current_path = []

        # LLM API setup
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise EnvironmentError("GROQ_API_KEY required")

        # Set initial positions
        self.internal_map.set_cell(self.robot_state.position, "ROBOT", 1.0)
        self.internal_map.set_cell(self.target_position, "GOAL", 1.0)

        logger.info(f"LLM Navigation System initialized | Target: {target_position}")

    def _calculate_goal_direction(self) -> str:
        """Calculate which direction the robot should face to move toward goal"""
        current_pos = self.robot_state.position
        goal_pos = self.target_position

        dr = goal_pos[0] - current_pos[0]  # row difference
        dc = goal_pos[1] - current_pos[1]  # column difference

        # Determine primary direction needed
        if abs(dr) > abs(dc):
            return "RIGHT" if dr > 0 else "LEFT"
        else:
            return "UP" if dc > 0 else "DOWN"

    def _calculate_distance_to_goal(self) -> int:
        """Calculate Manhattan distance to goal"""
        current_pos = self.robot_state.position
        goal_pos = self.target_position
        return abs(goal_pos[0] - current_pos[0]) + abs(goal_pos[1] - current_pos[1])

    def _is_facing_goal(self) -> bool:
        """Check if robot is facing toward the goal"""
        goal_direction = self._calculate_goal_direction()
        return self.robot_state.facing == goal_direction

    def _calculate_turn_needed(self) -> Optional[str]:
        """Calculate which way to turn to face the goal"""
        if self._is_facing_goal():
            return None

        goal_direction = self._calculate_goal_direction()
        current_facing = self.robot_state.facing

        directions = ["UP", "RIGHT", "DOWN", "LEFT"]
        current_idx = directions.index(current_facing)
        goal_idx = directions.index(goal_direction)

        # Calculate shortest turn
        diff = (goal_idx - current_idx) % 4
        if diff == 1 or diff == -3:
            return "RIGHT"
        elif diff == 3 or diff == -1:
            return "LEFT"
        elif diff == 2:
            return "RIGHT"  # Default to right for 180° turns

        return None

    async def navigate_step(self, sensor_data: SensorReading) -> Optional[Dict]:
        """Main navigation step with improved spatial reasoning"""
        try:
            # Update internal map from sensors
            self._update_map_from_sensors(sensor_data)

            # Get current situation analysis
            spatial_analysis = self._analyze_spatial_situation(sensor_data)
            map_visualization = self._generate_map_visualization()

            # Use LLM for navigation decision
            navigation_plan = await self._llm_plan_navigation(
                sensor_data, map_visualization, spatial_analysis
            )

            if not navigation_plan:
                logger.error("LLM failed to generate navigation plan")
                return None

            # Parse and format action
            formatted_action = self._parse_navigation_actions(navigation_plan)

            if formatted_action:
                logger.info(f"Generated action: {formatted_action}")
                return formatted_action
            else:
                logger.warning("No valid actions generated")
                return None

        except Exception as e:
            logger.error(f"Navigation step failed: {str(e)}")
            return None

    def _analyze_spatial_situation(self, sensors: SensorReading) -> Dict:
        """Analyze current spatial situation for better LLM context"""
        goal_direction = self._calculate_goal_direction()
        distance_to_goal = self._calculate_distance_to_goal()
        turn_needed = self._calculate_turn_needed()

        # Check if path ahead is clear
        path_clear = sensors.front > 60  # 60cm safety threshold

        # Analyze sensor readings
        obstacles = {
            "front": sensors.front < 60,
            "left": sensors.left < 60,
            "right": sensors.right < 60
        }

        return {
            "goal_direction": goal_direction,
            "distance_to_goal": distance_to_goal,
            "facing_goal": self._is_facing_goal(),
            "turn_needed": turn_needed,
            "path_clear": path_clear,
            "obstacles": obstacles,
            "recommended_action": self._get_recommended_action(sensors)
        }

    def _get_recommended_action(self, sensors: SensorReading) -> str:
        """Get recommended action based on spatial analysis"""
        turn_needed = self._calculate_turn_needed()

        # If not facing goal, turn first
        if turn_needed:
            return f"TURN {turn_needed} to face goal"

        # If facing goal and path clear, move forward
        if sensors.front > 60:
            distance_to_goal = self._calculate_distance_to_goal()
            max_cells = min(5, max(1, int(sensors.front / self.cell_size_cm) - 1))
            cells_to_move = min(max_cells, distance_to_goal)
            return f"MOVE FORWARD {cells_to_move} cells toward goal"

        # If blocked, need to navigate around obstacle
        if sensors.left > sensors.right:
            return "TURN LEFT to avoid obstacle"
        else:
            return "TURN RIGHT to avoid obstacle"

    def _calculate_next_waypoint(self, action: Action) -> Tuple[int, int]:
        """Calculate the next waypoint position based on current action"""
        current_pos = self.robot_state.position

        if action.type == "TURN":
            return current_pos
        elif action.type == "MOVE":
            facing = self.robot_state.facing
            cells = action.cells or 1

            # Fixed direction deltas
            direction_deltas = {
                "UP": (0, +cells),    # UP increases column (y)
                "DOWN": (0, -cells),  # DOWN decreases column (y)
                "LEFT": (-cells, 0),  # LEFT decreases row (x)
                "RIGHT": (+cells, 0)  # RIGHT increases row (x)
            }

            dr, dc = direction_deltas[facing]
            new_pos = (current_pos[0] + dr, current_pos[1] + dc)
            return new_pos

        return current_pos

    def _update_map_from_sensors(self, sensors: SensorReading):
        """Update internal map based on current sensor readings"""
        self.robot_state.step_count += 1
        current_pos = self.robot_state.position
        facing = self.robot_state.facing

        # Direction vectors based on robot orientation
        direction_vectors = {
            "UP": {"front": (0, +1), "left": (-1, 0), "right": (+1, 0)},
            "DOWN": {"front": (0, -1), "left": (+1, 0), "right": (-1, 0)},
            "LEFT": {"front": (-1, 0), "left": (0, -1), "right": (0, +1)},
            "RIGHT": {"front": (+1, 0), "left": (0, +1), "right": (0, -1)}
        }
        vectors = direction_vectors[facing]
        sensor_readings = {
            "front": sensors.front,
            "left": sensors.left,
            "right": sensors.right
        }

        # Process each sensor
        for sensor_dir, distance in sensor_readings.items():
            if distance >= self.max_sensor_range_cm:
                continue  # No obstacle detected in range

            dr, dc = vectors[sensor_dir]
            cells_to_obstacle = max(1, int(distance / self.cell_size_cm))

            # Mark clear cells up to obstacle
            for i in range(1, cells_to_obstacle):
                cell_pos = (current_pos[0] + i*dr, current_pos[1] + i*dc)
                if self.internal_map.get_cell(cell_pos).type not in ["GOAL", "ROBOT"]:
                    self.internal_map.set_cell(cell_pos, "CLEAR", 0.8)

            # Mark obstacle position
            obstacle_pos = (
                current_pos[0] + cells_to_obstacle*dr,
                current_pos[1] + cells_to_obstacle*dc
            )
            if self.internal_map.get_cell(obstacle_pos).type not in ["GOAL", "ROBOT"]:
                self.internal_map.set_cell(obstacle_pos, "OBSTACLE", 0.9)

    def _generate_map_visualization(self) -> str:
        """Generate ASCII visualization of current map"""
        if not self.internal_map.grid:
            return "Empty map"

        bounds = self.internal_map.bounds
        visualization = []

        # Add header with coordinates
        col_range = range(bounds["min_col"], bounds["max_col"] + 1)
        header = "   " + "".join(f"{c:2}" for c in col_range)
        visualization.append(header)

        # Generate grid rows
        for r in range(bounds["min_row"], bounds["max_row"] + 1):
            row_str = f"{r:2} "
            for c in col_range:
                cell = self.internal_map.get_cell((r, c))
                symbol = self._get_cell_symbol(cell.type, (r, c))
                row_str += f" {symbol}"
            visualization.append(row_str)

        return "\n".join(visualization)

    def _get_cell_symbol(self, cell_type: str, position: Tuple[int, int]) -> str:
        """Get display symbol for cell"""
        if position == self.robot_state.position:
            return {"UP": "↑", "DOWN": "↓", "LEFT": "←", "RIGHT": "→"}[self.robot_state.facing]
        elif position == self.target_position:
            return "G"
        elif cell_type == "OBSTACLE":
            return "■"
        elif cell_type == "CLEAR":
            return "·"
        else:
            return " "

    async def _llm_plan_navigation(self, sensors: SensorReading, map_viz: str,
                                   spatial_analysis: Dict) -> Optional[str]:
        """LLM planning with enhanced spatial context"""
        # Rate limiting logic...
        if self._daily_requests >= 1000:
            logger.error("⚠️ Daily request limit reached!")
            return None

        if self._tokens_used_today >= 500_000:
            logger.error("⚠️ Daily token limit reached!")
            return None

        elapsed = time.time() - self._last_api_call
        if elapsed < self._min_call_interval:
            await asyncio.sleep(self._min_call_interval - elapsed)

        try:
            self._last_api_call = time.time()
            self._daily_requests += 1

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "llama-3.3-70b-versatile",
                        "messages": [
                            {"role": "system", "content": self._get_enhanced_system_prompt()},
                            {"role": "user", "content": self._build_enhanced_navigation_prompt(
                                sensors, map_viz, spatial_analysis)}
                        ],
                        "temperature": 0.1,
                        "max_tokens": 400,
                    },
                    timeout=10.0,
                )

                response_data = response.json()

                if "choices" not in response_data or not response_data["choices"]:
                    logger.error(f"Invalid LLM response: {response_data}")
                    return None

                tokens_used = response_data.get("usage", {}).get("total_tokens", 0)
                self._tokens_used_today += tokens_used

                return response_data["choices"][0]["message"]["content"].strip()

        except Exception as e:
            logger.error(f"LLM planning error: {e}")
            return None

    def _get_enhanced_system_prompt(self) -> str:
        """Enhanced system prompt with better spatial reasoning"""
        return """You are an advanced robot navigation AI with strong spatial reasoning abilities.

COORDINATE SYSTEM:
- Position (row, col) where row=x-axis, col=y-axis
- UP: increases column (+y direction)
- DOWN: decreases column (-y direction)
- LEFT: decreases row (-x direction)
- RIGHT: increases row (+x direction)

NAVIGATION PRINCIPLES:
1. ALWAYS face the goal direction before moving (turn first if needed)
2. Move efficiently toward goal when path is clear
3. Navigate around obstacles intelligently
4. Use sensor data to avoid collisions (60cm safety threshold)
5. Prioritize direct paths but adapt when blocked

DECISION PRIORITY:
1. If not facing goal → TURN toward goal
2. If facing goal and path clear → MOVE FORWARD toward goal
3. If blocked → TURN to find alternate path

OUTPUT FORMAT:
Respond with ONLY valid JSON between ```json ``` markers:
```json
{
  "spatial_reasoning": "Your analysis of position and goal relationship",
  "navigation_decision": "Why this action makes sense",
  "action": {"type": "TURN", "direction": "LEFT"} or {"type": "MOVE", "direction": "FORWARD", "cells": 1-5}
}
```

CONSTRAINTS:
- TURN: "LEFT" or "RIGHT" only
- MOVE: "FORWARD" only, cells: 1-5
- Always provide valid JSON
- Consider spatial analysis provided in prompt"""

    def _build_enhanced_navigation_prompt(self, sensors: SensorReading,
                                          map_viz: str, spatial_analysis: Dict) -> str:
        """Build enhanced navigation prompt with spatial context"""
        return f"""ENHANCED NAVIGATION ANALYSIS

ROBOT STATE:
- Current Position: {self.robot_state.position}
- Currently Facing: {self.robot_state.facing}
- Goal Position: {self.target_position}
- Steps Taken: {self.robot_state.step_count}

SPATIAL ANALYSIS:
- Direction to Goal: {spatial_analysis['goal_direction']}
- Distance to Goal: {spatial_analysis['distance_to_goal']} cells
- Facing Goal: {spatial_analysis['facing_goal']}
- Turn Needed: {spatial_analysis.get('turn_needed', 'None')}
- Path Clear: {spatial_analysis['path_clear']}
- System Recommendation: {spatial_analysis['recommended_action']}

SENSOR READINGS (60cm = obstacle threshold):
- Front: {sensors.front}cm - {'🚫 BLOCKED' if sensors.front < 60 else '✅ CLEAR'}
- Left: {sensors.left}cm - {'🚫 BLOCKED' if sensors.left < 60 else '✅ CLEAR'}
- Right: {sensors.right}cm - {'🚫 BLOCKED' if sensors.right < 60 else '✅ CLEAR'}

CURRENT MAP:
{map_viz}

LEGEND: ↑↓←→=Robot(facing), G=Goal, ■=Obstacle, ·=Clear, space=Unknown

NAVIGATION LOGIC:
1. Am I facing the goal direction? {spatial_analysis['facing_goal']}
2. If NO → Turn {spatial_analysis.get('turn_needed', 'not needed')} to face goal
3. If YES → Can I move forward safely? {spatial_analysis['path_clear']}
4. If blocked → Turn toward clearer path

Choose the ONE best action to efficiently reach the goal while avoiding obstacles.
Consider the system recommendation: {spatial_analysis['recommended_action']}"""

    def _parse_navigation_actions(self, llm_response: str) -> Optional[Dict]:
        """Parse action from LLM response with better error handling"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if not json_match:
                logger.error("No JSON found in LLM response")
                return None

            data = json.loads(json_match.group())

            # Log LLM reasoning
            for key in ["spatial_reasoning", "navigation_decision"]:
                if key in data:
                    logger.info(f"LLM {key}: {data[key]}")

            # Parse action
            if "action" not in data:
                logger.error("No action in LLM response")
                return None

            try:
                action = Action.parse_obj(data["action"])
                next_waypoint = self._calculate_next_waypoint(action)

                # Format output
                formatted_output = {
                    "next_waypoint": next_waypoint,
                    "type": action.type,
                    "direction": action.direction,
                }

                if action.type == "MOVE":
                    formatted_output["cells"] = action.cells or 1

                return formatted_output

            except ValidationError as e:
                logger.warning(f"Invalid action: {data['action']}, error: {e}")
                return None

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"Action parsing error: {e}")
            return None

    def execute_action(self, action_dict: Dict) -> bool:
        """Execute action and update robot state"""
        try:
            current_pos = self.robot_state.position
            action_type = action_dict.get("type")
            direction = action_dict.get("direction")

            if action_type == "TURN":
                directions = ["UP", "RIGHT", "DOWN", "LEFT"]
                current_idx = directions.index(self.robot_state.facing)

                if direction == "RIGHT":
                    new_idx = (current_idx + 1) % 4
                elif direction == "LEFT":
                    new_idx = (current_idx - 1) % 4
                else:
                    logger.error(f"Invalid turn direction: {direction}")
                    return False

                self.robot_state.facing = directions[new_idx]
                logger.info(f"Robot turned {direction}, now facing {self.robot_state.facing}")

            elif action_type == "MOVE":
                facing = self.robot_state.facing
                cells = action_dict.get("cells", 1)

                # Fixed direction deltas
                direction_deltas = {
                    "UP": (0, +cells),
                    "DOWN": (0, -cells),
                    "LEFT": (-cells, 0),
                    "RIGHT": (+cells, 0)
                }

                dr, dc = direction_deltas[facing]
                new_pos = (current_pos[0] + dr, current_pos[1] + dc)

                # Update map
                if self.internal_map.get_cell(current_pos).type == "ROBOT":
                    self.internal_map.set_cell(current_pos, "CLEAR", 1.0)

                self.robot_state.position = new_pos
                self.internal_map.set_cell(new_pos, "ROBOT", 1.0)
                logger.info(f"Robot moved {cells} cells to {new_pos}")

            return True

        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            return False

    def is_goal_reached(self) -> bool:
        """Check if robot has reached the goal"""
        return self.robot_state.position == self.target_position

    def get_current_map(self) -> str:
        """Get current map visualization"""
        return self._generate_map_visualization()

    def get_robot_status(self) -> Dict:
        """Get current robot status"""
        return {
            "position": self.robot_state.position,
            "facing": self.robot_state.facing,
            "step_count": self.robot_state.step_count,
            "goal_position": self.target_position,
            "goal_reached": self.is_goal_reached(),
            "map_size": len(self.internal_map.grid),
            "distance_to_goal": self._calculate_distance_to_goal(),
            "facing_goal": self._is_facing_goal()
        }