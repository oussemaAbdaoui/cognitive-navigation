#!/usr/bin/env python3
# langgraph_nodes.py - Enhanced LLM Path Planning with Spatial Reasoning and Feedback

import json
import os
import re
import math
from typing import Dict, List, Tuple, Optional, Literal
from pydantic import BaseModel, ValidationError, conint
from loguru import logger
import httpx
from dataclasses import dataclass, field
import time
import asyncio

# Configure logging
logger.add(
    "navigation.log",
    rotation="10 MB",
    retention="7 days",
    level="INFO",
    enqueue=True,
    backtrace=True,
    diagnose=True
)

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
    """Enhanced robot state with trajectory memory"""
    position: Tuple[int, int] = (0, 0)
    facing: Literal["UP", "DOWN", "LEFT", "RIGHT"] = "UP"
    step_count: int = 0
    recent_positions: List[Tuple[int, int]] = field(default_factory=list)
    last_goal_distance: float = 0.0
    consecutive_distance_increases: int = 0

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

# ------------------- Enhanced LLM Navigation System -------------------
class LLMNavigationSystem:
    """Enhanced LLM-based navigation with spatial reasoning and feedback"""

    def __init__(self,
                 target_position: Tuple[int, int],
                 cell_size_cm: int = 30,
                 max_sensor_range_cm: int = 300):
        self._min_call_interval = 0.0
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

        # Initialize distance tracking
        self.robot_state.last_goal_distance = self._calculate_distance_to_goal()

        # LLM API setup
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise EnvironmentError("GROQ_API_KEY required")

        # Set initial positions
        self.internal_map.set_cell(self.robot_state.position, "ROBOT", 1.0)
        self.internal_map.set_cell(self.target_position, "GOAL", 1.0)

        logger.info(f"Enhanced LLM Navigation System initialized | Target: {target_position}")

    def _calculate_distance_to_goal(self) -> float:
        """Calculate Manhattan distance to goal"""
        x1, y1 = self.robot_state.position
        x2, y2 = self.target_position
        return abs(x2 - x1) + abs(y2 - y1)

    def _get_relative_goal_description(self) -> str:
        """Get goal position relative to current position and facing"""
        current_x, current_y = self.robot_state.position
        goal_x, goal_y = self.target_position
        dx = goal_x - current_x
        dy = goal_y - current_y

        # Convert to robot-facing coordinates
        if self.robot_state.facing == "UP":
            rel_x = dx
            rel_y = dy
        elif self.robot_state.facing == "DOWN":
            rel_x = -dx
            rel_y = -dy
        elif self.robot_state.facing == "LEFT":
            rel_x = dy
            rel_y = -dx
        else:  # RIGHT
            rel_x = -dy
            rel_y = dx

        # Generate description
        direction_x = "RIGHT" if rel_x > 0 else "LEFT"
        direction_y = "UP" if rel_y > 0 else "DOWN"
        return (
            f"Goal is {abs(rel_x)} cells {direction_x} and "
            f"{abs(rel_y)} cells {direction_y} relative to current facing"
        )

    async def navigate_step(self, sensor_data: SensorReading) -> Optional[Dict]:
        """Enhanced navigation step with spatial reasoning and progress tracking"""
        try:
            # Update internal map from sensors
            self._update_map_from_sensors(sensor_data)

            # Check progress every 5 steps
            if self.robot_state.step_count % 5 == 0:
                self._check_progress()

            # Get current situation
            map_visualization = self._generate_map_visualization()

            # Use LLM for path planning
            navigation_plan = await self._llm_plan_navigation(
                sensor_data, map_visualization
            )

            if not navigation_plan:
                logger.error("LLM failed to generate navigation plan")
                return None

            # Extract actions from plan and format as requested
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

    def _check_progress(self):
        """Check if we're making progress toward the goal"""
        current_distance = self._calculate_distance_to_goal()
        prev_distance = self.robot_state.last_goal_distance

        if current_distance > prev_distance:
            self.robot_state.consecutive_distance_increases += 1
            logger.warning(
                f"⚠️ Distance increased! {prev_distance} → {current_distance} cells "
                f"(Consecutive: {self.robot_state.consecutive_distance_increases})"
            )

            # Trigger error recovery if we've increased distance 3+ times
            if self.robot_state.consecutive_distance_increases >= 3:
                logger.error("🚨 CRITICAL: Triggering error recovery protocol!")
                # In a real system, this would initiate recovery behaviors
        else:
            self.robot_state.consecutive_distance_increases = 0

        self.robot_state.last_goal_distance = current_distance

    def _calculate_next_waypoint(self, action: Action) -> Tuple[int, int]:
        """Calculate the next waypoint position based on current action"""
        current_pos = self.robot_state.position

        if action.type == "TURN":
            return current_pos
        elif action.type == "MOVE":
            facing = self.robot_state.facing
            cells = action.cells or 1

            # Cartesian coordinate system:
            # X-axis: Horizontal (LEFT = -X, RIGHT = +X)
            # Y-axis: Vertical (DOWN = -Y, UP = +Y)
            direction_deltas = {
                "UP": (0, +cells),    # UP increases Y coordinate
                "DOWN": (0, -cells),  # DOWN decreases Y coordinate
                "LEFT": (-cells, 0),  # LEFT decreases X coordinate
                "RIGHT": (+cells, 0)  # RIGHT increases X coordinate
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
        """Generate ASCII visualization of current map with coordinate labels"""
        if not self.internal_map.grid:
            return "Empty map"

        bounds = self.internal_map.bounds
        visualization = []

        # Add header with coordinates
        col_range = range(bounds["min_col"], bounds["max_col"] + 1)
        header = "    " + "".join(f"{c:>3}" for c in col_range)
        visualization.append(header)

        # Generate grid rows
        for r in range(bounds["min_row"], bounds["max_row"] + 1):
            row_str = f"{r:3} "
            for c in col_range:
                cell = self.internal_map.get_cell((r, c))
                symbol = self._get_cell_symbol(cell.type, (r, c))
                row_str += f" {symbol} "
            visualization.append(row_str)

        # Add coordinate system explanation
        visualization.append("\nCoordinate System:")
        visualization.append("  X-axis → Columns (RIGHT = +X, LEFT = -X)")
        visualization.append("  Y-axis → Rows (UP = +Y, DOWN = -Y)")

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

    async def _llm_plan_navigation(self, sensors: SensorReading, map_viz: str) -> Optional[str]:
        # Check daily request limits
        if self._daily_requests >= 1000:
            logger.error("⚠️ Daily request limit (1000) reached!")
            return None
        if self._tokens_used_today >= 500_000:
            logger.error("⚠️ Daily token limit (500K) reached!")
            return None

        # Enforce minimum delay
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
                            {"role": "system", "content": self._get_system_prompt()},
                            {"role": "user", "content": self._build_navigation_prompt(sensors, map_viz)}
                        ],
                        "temperature": 0.1,
                        "max_tokens": 400,
                    },
                    timeout=15.0,
                )

                response_data = response.json()

                # Validate response
                if "choices" not in response_data or not response_data["choices"]:
                    logger.error(f"Invalid LLM response: {response_data}")
                    return None

                # Track token usage
                tokens_used = response_data.get("usage", {}).get("total_tokens", 0)
                self._tokens_used_today += tokens_used

                return response_data["choices"][0]["message"]["content"].strip()

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                retry_after = float(e.response.headers.get("Retry-After", 5.0))
                logger.warning(f"Rate limited. Retrying in {retry_after}s...")
                await asyncio.sleep(retry_after)
                return await self._llm_plan_navigation(sensors, map_viz)
            else:
                logger.error(f"API Error: {e}")
                return None

        except Exception as e:
            logger.error(f"Unexpected error in LLM planning: {e}")
            return None

    def _get_system_prompt(self) -> str:
        """Enhanced system prompt with spatial reasoning and feedback"""
        return """
You are an advanced robot navigation AI that uses real-time sensor data to build maps and plan paths.

CORE CAPABILITIES:
- Process ultrasonic sensor readings (front, left, right)
- Build and update internal spatial maps
- Plan optimal navigation sequences
- Avoid collisions with obstacles
- Reason about spatial relationships and movement

GRID COORDINATE SYSTEM:
- Cartesian coordinates with origin (0,0) at robot start
- X-axis: Horizontal (LEFT = -X, RIGHT = +X)
- Y-axis: Vertical (DOWN = -Y, UP = +Y)
- Example: Goal at (3,2) means 3 cells right, 2 cells up

NAVIGATION PRINCIPLES:
1. Always prioritize safety - avoid obstacles
2. Use sensor data to update your understanding of the environment
3. Plan efficient paths toward the goal
4. Adapt when paths are blocked
5. Minimize unnecessary movements

MOVEMENT CONSTRAINTS:
1. Near obstacles (<2 cells): Max 1 cell moves
2. Sensor uncertainty (>50cm variance): Max 2 cell moves
3. Unknown territory: Max 3 cell moves
4. Clear paths: 5 cell moves allowed
5. Always verify path with sensors before long moves

TURN DECISION PROTOCOL:
1. Always calculate required heading to goal
2. If required_heading ≠ current facing:
   a. MUST turn unless:
      - Blocked path in required direction
      - Shorter path available in current heading
      - Sensor data shows danger in required direction
   b. ALWAYS justify not turning
3. Include 'turn_justification' field in response

PROGRESS MONITORING:
1. Calculate distance-to-goal every step
2. If distance increases for 2+ consecutive steps:
   a. Trigger path reassessment
   b. Consider backtracking
   c. Report 'progress_alert' in analysis

ERROR RECOVERY PROTOCOL:
1. If 3+ consecutive distance increases:
   a. Stop immediately
   b. Revert to last known good position
   c. Perform full map rescan
   d. Generate new path
2. If stuck in loop (repeated positions):
   a. Switch to wall-following mode
   b. Request human intervention if needed

OUTPUT FORMAT:
You MUST respond with ONLY a valid JSON object between ```json ``` markers:
```json
{
  "position_analysis": {
    "delta_x": "ΔX value (goal_x - current_x)",
    "delta_y": "ΔY value (goal_y - current_y)",
    "relative_direction": "e.g., 'Goal is 2 cells forward, 1 cell left'",
    "required_heading": "Optimal direction to face"
  },
  "spatial_analysis": "Your reasoning about the current situation",
  "map_understanding": "What you've learned from the map",
  "navigation_strategy": "Your approach to reach the goal",
  "turn_justification": "Explanation if not turning toward goal",
  "progress_alert": "Warning if distance increasing",
  "action": {
    "type": "TURN",
    "direction": "LEFT"
  } OR {
    "type": "MOVE",
    "direction": "FORWARD",
    "cells": 1-5
  }
}
```

IMPORTANT:
- Always describe goal position RELATIVE to current position/facing
- Express as: 'Goal is [abs(ΔX)] cells [left/right] and [abs(ΔY)] cells [up/down]'
- 'Forward' = current facing, 'Left/Right' relative to heading
- Include ONLY ONE action per response
- Do not include any additional text outside the JSON
- Consider sensor readings when planning"""

    def _build_navigation_prompt(self, sensors: SensorReading, map_viz: str) -> str:
        """Build enhanced navigation prompt with trajectory memory and progress tracking"""
        # Calculate relative goal position
        relative_goal = self._get_relative_goal_description()
        distance_to_goal = self._calculate_distance_to_goal()

        # Progress tracking message
        progress_msg = ""
        if self.robot_state.consecutive_distance_increases > 0:
            progress_msg = (
                f"⚠️ PROGRESS WARNING: Distance increased for "
                f"{self.robot_state.consecutive_distance_increases} consecutive steps!"
            )

        return f"""NAVIGATION SITUATION ANALYSIS

CURRENT ROBOT STATE:
- Position: {self.robot_state.position}
- Facing: {self.robot_state.facing}
- Step: {self.robot_state.step_count}
- Goal: {self.target_position}
- Relative Goal: {relative_goal}
- Distance to Goal: {distance_to_goal:.1f} cells
{progress_msg}

MOVEMENT HISTORY (last 5 positions):
{self.robot_state.recent_positions[-5:]}

SENSOR READINGS (cm):
- Front: {sensors.front} {'(BLOCKED)' if sensors.front < 60 else '(CLEAR)'}
- Left: {sensors.left} {'(BLOCKED)' if sensors.left < 60 else '(CLEAR)'}
- Right: {sensors.right} {'(BLOCKED)' if sensors.right < 60 else '(CLEAR)'}

CURRENT MAP:
{map_viz}

MAP LEGEND:
- ↑↓←→ : Robot (facing direction)
- G : Goal position
- ■ : Detected obstacles
- · : Clear/explored areas
- (space) : Unknown/unexplored

NAVIGATION TASK:
Analyze the sensor data and current map to determine the best SINGLE navigation action.

REQUIRED ANALYSIS:
1. Calculate current-to-goal vector (delta_x, delta_y)
2. Determine optimal facing direction for goal approach
3. If not turning toward goal, provide justification
4. Check if we're making progress toward goal
5. Select safe action considering movement constraints

RESPOND WITH VALID JSON CONTAINING:
- position_analysis with delta values and relative direction
- turn_justification if not turning toward goal
- progress_alert if distance is increasing
- ONE navigation action"""

    def _parse_navigation_actions(self, llm_response: str) -> Optional[Dict]:
        """Parse action from LLM response with enhanced fields"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if not json_match:
                logger.error("No JSON found in LLM response")
                return None

            data = json.loads(json_match.group())

            # Log all LLM analysis fields
            analysis_fields = [
                "position_analysis", "spatial_analysis",
                "navigation_strategy", "turn_justification",
                "progress_alert"
            ]

            for field in analysis_fields:
                if field in data:
                    logger.info(f"LLM {field.replace('_', ' ').title()}: {data[field]}")

            # Parse single action
            if "action" not in data:
                logger.error("No action in LLM response")
                return None

            try:
                action = Action.parse_obj(data["action"])

                # Calculate next waypoint
                next_waypoint = self._calculate_next_waypoint(action)

                # Format output
                formatted_output = {
                    "next_waypoint": next_waypoint,
                    "type": action.type,
                    "direction": action.direction,
                }

                # Add cells only for MOVE actions
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
        """Execute action and update robot state with trajectory memory"""
        try:
            current_pos = self.robot_state.position
            action_type = action_dict.get("type")
            direction = action_dict.get("direction")

            # Update trajectory memory
            self.robot_state.recent_positions.append(current_pos)
            if len(self.robot_state.recent_positions) > 5:
                self.robot_state.recent_positions.pop(0)

            if action_type == "TURN":
                # Update facing direction
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

                # Cartesian coordinate system movements
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

                # Update distance tracking
                new_distance = self._calculate_distance_to_goal()
                if new_distance > self.robot_state.last_goal_distance:
                    self.robot_state.consecutive_distance_increases += 1
                else:
                    self.robot_state.consecutive_distance_increases = 0
                self.robot_state.last_goal_distance = new_distance

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
            "distance_to_goal": self._calculate_distance_to_goal(),
            "goal_reached": self.is_goal_reached(),
            "recent_positions": self.robot_state.recent_positions[-5:],
            "consecutive_distance_increases": self.robot_state.consecutive_distance_increases
        }