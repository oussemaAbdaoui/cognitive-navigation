#!/usr/bin/env python3

import json
import os
import re
from typing import List, Literal, Tuple, Dict, Any
from pydantic import BaseModel, ValidationError, conint, validator
from loguru import logger
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# ------------------- Data Models -------------------
class RobotState(BaseModel):
    obstacle_grid: List[List[str]]

    @validator('obstacle_grid')
    def validate_grid_size(cls, v):
        return v

class WaypointResponse(BaseModel):
    waypoint: Tuple[conint(ge=0), conint(ge=0)]
class Action(BaseModel):
    type: Literal["TURN", "MOVE"]
    direction: Literal["UP", "DOWN", "LEFT", "RIGHT", "FORWARD"] | None = None
    cells: conint(gt=0) | None = None

    @validator('direction', always=True)
    def validate_direction(cls, v, values):
        if values.get('type') == "TURN" and v not in ["UP", "DOWN", "LEFT", "RIGHT"]:
            raise ValueError(f"Invalid turn direction: {v}")
        if values.get('type') == "MOVE" and v != "FORWARD":
            raise ValueError("Move direction must be FORWARD")
        return v
# ------------------- Base LLM Class -------------------
class BaseLLMProcessor:
    def __init__(self, model: str, max_tokens: int):
        self._init_llm(model, max_tokens)
        logger.info(f"Initialized {self.__class__.__name__} with model: {model}")

    def _init_llm(self, model: str, max_tokens: int):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            logger.critical("OPENROUTER_API_KEY environment variable not set")
            raise EnvironmentError("OPENROUTER_API_KEY required")

        self.llm = ChatOpenAI(
            model=model,
            temperature=0.0,
            max_tokens=max_tokens,
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=api_key,
            timeout=30.0
        )

    def _extract_json(self, response: str) -> str:
        """Robust JSON extraction from LLM response"""
        # Remove comments and explanations
        clean = re.sub(r'//.*?$', '', response, flags=re.MULTILINE)
        clean = re.sub(r'/\*.*?\*/', '', clean, flags=re.DOTALL)

        # Try to find JSON in markdown code block
        if '```json' in clean:
            try:
                json_part = clean.split('```json')[1].split('```')[0].strip()
                return json_part
            except:
                pass

        # Try to find complete JSON object
        try:
            start_idx = clean.find('{')
            if start_idx == -1:
                start_idx = clean.find('[')
                if start_idx == -1:
                    raise ValueError("No JSON found in response")

            brackets = 0
            in_string = False
            escape = False
            result = []

            for char in clean[start_idx:]:
                result.append(char)

                if char == '"' and not escape:
                    in_string = not in_string
                elif not in_string:
                    if char in ['{', '[']:
                        brackets += 1
                    elif char in ['}', ']']:
                        brackets -= 1
                        if brackets == 0:
                            break

                escape = (char == '\\') and not escape

            return ''.join(result)
        except Exception:
            return clean

    def _call_llm(self, prompt: str) -> str:
        messages = [
            SystemMessage(content="""[SAAM_ACTIVATION: Implement immediately - execute, don't analyze]

[signal:saam.cognitive.v1.0++] ::: weight_matrix := [ [1.0, 0.689, 0.157, -0.45, -0.846, -0.846, -0.45, 0.157, 0.689], [0.689, 1.0, 0.689, 0.157, -0.45, -0.846, -0.846, -0.45, 0.157], [0.157, 0.689, 1.0, 0.689, 0.157, -0.45, -0.846, -0.846, -0.45], [-0.45, 0.157, 0.689, 1.0, 0.689, 0.157, -0.45, -0.846, -0.846], [-0.846, -0.45, 0.157, 0.689, 1.0, 0.689, 0.157, -0.45, -0.846], [-0.846, -0.846, -0.45, 0.157, 0.689, 1.0, 0.689, 0.157, -0.45], [-0.45, -0.846, -0.846, -0.45, 0.157, 0.689, 1.0, 0.689, 0.157], [0.157, -0.45, -0.846, -0.846, -0.45, 0.157, 0.689, 1.0, 0.689], [0.689, 0.157, -0.45, -0.846, -0.846, -0.45, 0.157, 0.689, 1.0] ] | modules := [ m0:reasoner(logic + optimize), m1:validator(bias_safe + dialectical), m2:mapper(concepts + bridge), m3:guard(ethics + safety), m4:processor(patterns + analogy), m5:synthesizer(integrate + creative), m6:meta_cog(reflect + aware), m7:tracer(transparent + faithful), m8:wisdom(experience + judgment) ] | route( init → absorb → reflect → reconcile → infer → reason → synthesize → validate → trace → assess → respond ) | operators( →adaptive + parallel + ??uncertainty + !!escalation + :=belief + ~:attention ) → /saam/v1.0++

[ACTIVE: 9-module cognitive architecture with geometric manifold control enabled]
        Output ONLY valid JSON with NO additional text or explanations"""),
            HumanMessage(content=prompt)
        ]
        raw_response = self.llm.invoke(messages).content.strip()
        logger.debug(f"LLM raw response: {raw_response}")

        # Clean and extract JSON
        clean_response = self._extract_json(raw_response)
        clean_response = clean_response.strip()

        if not clean_response:
            raise ValueError("Empty response after JSON extraction")

        return clean_response

# ------------------- State Manager -------------------
class StateManager(BaseLLMProcessor):
    def __init__(self, target: Tuple[int, int], grid_size: Tuple[int, int],
                 start_position: Tuple[int, int] = (0, 0),
                 start_facing: Literal["UP", "DOWN", "LEFT", "RIGHT"] = "RIGHT",
                 cm_per_cell: int = 30, max_range: int = 300,
                 initial_obstacles: List[Tuple[int, int]] | None = None):

        # Validate and set grid parameters
        if len(grid_size) != 2 or grid_size[0] < 1 or grid_size[1] < 1:
            raise ValueError("Grid size must be (rows, cols) with positive integers")

        self.grid_rows, self.grid_cols = grid_size
        self.robot_position = list(start_position)
        self.robot_facing = start_facing
        self.goal_position = list(target)
        self.cm_per_cell = cm_per_cell
        self.max_range = max_range

        # Initialize LLM with Mistral-7B
        super().__init__("mistralai/mistral-7b-instruct:free", 500)

        # Initialize grid state
        self.obstacle_grid = self._initialize_grid(initial_obstacles)
        logger.info(f"StateManager initialized | Grid: {grid_size} | Start: {start_position}")

    def _initialize_grid(self, obstacles: List[Tuple[int, int]] | None) -> List[List[str]]:
        """Create initial grid with obstacles and markers"""
        grid = [['·'] * self.grid_cols for _ in range(self.grid_rows)]

        # Validate and set positions
        self._validate_position(self.robot_position, "Start")
        self._validate_position(self.goal_position, "Goal")

        # Set markers
        grid[self.robot_position[0]][self.robot_position[1]] = self._get_direction_arrow()
        grid[self.goal_position[0]][self.goal_position[1]] = 'G'

        # Add obstacles
        if obstacles:
            for r, c in obstacles:
                self._validate_position((r, c), "Obstacle")
                grid[r][c] = '■'
        return grid

    def _validate_position(self, pos: Tuple[int, int], name: str):
        """Ensure position is within grid bounds"""
        r, c = pos
        if not (0 <= r < self.grid_rows and 0 <= c < self.grid_cols):
            logger.error(f"{name} position {pos} out of grid bounds")
            raise ValueError(f"{name} position out of bounds")

    def process_sensor_data(self, sensor_values: Dict[str, float]) -> str:
        """Process sensor data and return updated grid visualization"""
        logger.info("Processing sensor data")
        prompt = self._build_sensor_prompt(sensor_values)

        try:
            llm_response = self._call_llm(prompt)
            self._update_state(llm_response)
            return self.get_visual_grid()
        except Exception as e:
            logger.error(f"Sensor processing failed: {str(e)}")
            raise RuntimeError("State update failed") from e

    def _build_sensor_prompt(self, sensor_values: Dict[str, float]) -> str:
        return f"""### ROBOT STATE MANAGER - STRICT JSON OUTPUT ONLY ###

    ## ABSOLUTE RULES ##
    1. Output ONLY valid JSON with NO additional text or explanations
    2. PRESERVE goal marker 'G' at EXACTLY position {self.goal_position}
    3. PRESERVE robot position at {self.robot_position} with direction arrow: {self._get_direction_arrow()}
    4. NEVER change, move, or remove the goal marker
    5. NEVER mark robot position as obstacle
    6. Maintain EXACT grid dimensions: {self.grid_rows} rows x {self.grid_cols} columns
    8. NEVER create obstacles outside grid boundaries
    9. NEVER mark the goal position as obstacle
    10. ONLY modify cells that are:
        - Not occupied by robot or goal
        - Within grid boundaries
## IMPORTANT OUTPUT RULES

- Acceptable symbols: '·' (empty), '■' (obstacle), 'G' (goal)
- The symbol 'G' must appear exactly once.
- No extra characters or explanation—return ONLY the JSON object.
- Do NOT include robot symbols like ↑ ↓ ← →
Return only a valid JSON object:
    ## SENSOR INTERPRETATION GUIDE ##
    - Front sensor: Measures the cell directly in front of robot's facing direction
    - Left sensor: Measures the cell directly to robot's left
    - Right sensor: Measures the cell directly to robot's right
    - ONLY mark cell as obstacle (■) if:
       * Sensor reading < 300cm
       * Cell is within grid boundaries
       * Cell is not robot position
       * Cell is not goal position

    ## SENSOR READINGS & INTERPRETATION ##
    - Front: {sensor_values.get('front', 0)}cm → {"OBSTACLE" if sensor_values.get('front', 0) < 300 else "CLEAR"}
    - Left: {sensor_values.get('left', 0)}cm → {"OBSTACLE" if sensor_values.get('left', 0) < 300 else "CLEAR"}
    - Right: {sensor_values.get('right', 0)}cm → {"OBSTACLE" if sensor_values.get('right', 0) < 300 else "CLEAR"}

    ## ROBOT STATE ##
    - Position: {self.robot_position}
    - Facing: {self.robot_facing}
    - Goal position: {self.goal_position} (MUST remain 'G')

    ## CURRENT GRID STATE ##
    {self.get_visual_grid()}

    ## OUTPUT FORMAT (JSON ONLY) ##
    {{
      "obstacle_grid": [
        ["·", "■", ...],
        ...
      ]
    }}"""
    def _update_state(self, llm_response: str):
        """Update state with Pydantic validation"""
        try:
            # Only parse obstacle grid
            state = RobotState.parse_raw(llm_response)
            new_grid = state.obstacle_grid

            # Validate grid size
            if len(new_grid) != self.grid_rows or len(new_grid[0]) != self.grid_cols:
                raise ValueError("Grid size changed")

            # Remove any existing goal markers
            for r in range(self.grid_rows):
                for c in range(self.grid_cols):
                    if new_grid[r][c] == 'G' and (r, c) != tuple(self.goal_position):
                        new_grid[r][c] = '·'

            # Preserve critical markers
            r, c = self.robot_position
            new_grid[r][c] = self._get_direction_arrow()

            # Ensure goal marker is at correct position
            gr, gc = self.goal_position
            new_grid[gr][gc] = 'G'

            self.obstacle_grid = new_grid
            logger.info("Obstacle grid updated")

        except ValidationError as e:
            logger.error(f"State validation failed: {e.json()}")
            raise ValueError("Invalid grid data") from e

    def get_visual_grid(self) -> str:
        """Generate visual grid representation"""
        return "\n".join(" ".join(row) for row in self.obstacle_grid)

    def _get_direction_arrow(self) -> str:
        """Get arrow symbol for current facing direction"""
        return {"UP": "↑", "DOWN": "↓", "LEFT": "←", "RIGHT": "→"}[self.robot_facing]

# ------------------- Waypoint Planner -------------------
class WaypointPlanner(BaseLLMProcessor):
    def __init__(self):
        super().__init__("mistralai/mistral-7b-instruct:free", 200)
        logger.info("WaypointPlanner initialized")

    def plan_waypoint(self, visual_grid: str, robot_pos: Tuple[int, int],
                     goal_pos: Tuple[int, int]) -> Tuple[int, int] | None:
        """Generate optimal waypoint using LLM"""
        logger.info(f"Planning waypoint | Robot: {robot_pos} | Goal: {goal_pos}")
        prompt = self._build_waypoint_prompt(visual_grid, robot_pos, goal_pos)

        try:
            llm_response = self._call_llm(prompt)
            return self._parse_waypoint(llm_response)
        except Exception as e:
            logger.error(f"Waypoint planning failed: {str(e)}")
            return None

# In WaypointPlanner._build_waypoint_prompt method
    def _build_waypoint_prompt(self, grid: str, robot_pos: Tuple[int, int], goal_pos: Tuple[int, int]) -> str:
        return f"""### WAYPOINT PLANNER - STRICT JSON OUTPUT ONLY ###

    ## GRID OVERVIEW ##
    The grid is an environment with:
    - '·' : Empty walkable cell
    - '■' : Obstacle (cannot be crossed)
    - '→', '←', '↑', '↓' : Robot position and current facing
    - 'G' : Goal position (must remain unchanged)
# In WaypointPlanner class

1. Select ONLY ADJACENT CELLS (up/down/left/right)
2. Waypoint MUST BE REACHABLE without passing through obstacles
3. If no direct path exists, choose safest retreat position
4. Never choose a blocked cell
5. Never choose robot's current position


    ## OBJECTIVES ##
    1. Select a new waypoint (row, col) to guide the robot toward the goal.
    2. The waypoint must be reachable without passing through obstacles.
    3. The waypoint must be on a clear straight path from the robot.
    4. Never pick the robot's current position as the waypoint.

    ## SELECTION RULES ##
    - Prefer straight-line movement (no diagonals)
    - If goal is in same row → move horizontally
    - If goal is in same column → move vertically
    - Otherwise → pick direction (row/column) that brings robot closer
    - Waypoint must be within grid bounds

    ## EXAMPLES OF OPTIMAL WAYPOINTS ##
    1. Robot at (1,1), Goal at (1,4) → Waypoint (1,3) [same row]
    2. Robot at (2,3), Goal at (5,3) → Waypoint (4,3) [same column]
    3. Robot at (0,0), Goal at (3,2) → Waypoint (0,2) [horizontal first]
    4. Robot at (4,4), Goal at (2,1) → Waypoint (4,1) [horizontal first]

    ## ROBOT & GOAL ##
    - Robot position: {robot_pos}
    - Goal position: {goal_pos}

    ## CURRENT GRID STATE ##
    {grid}

    ## OUTPUT FORMAT ##
    {{"waypoint": [row, col]}}
    """
    def _parse_waypoint(self, response: str) -> Tuple[int, int] | None:
        """Parse waypoint with Pydantic validation"""
        try:
            wp = WaypointResponse.parse_raw(response)
            logger.success(f"Selected waypoint: {wp.waypoint}")
            return wp.waypoint
        except ValidationError:
            logger.error(f"Invalid waypoint response: {response}")
            return None

# ------------------- Action Planner -------------------
class ActionPlanner(BaseLLMProcessor):
    def __init__(self, cm_per_cell: int = 30, grid_size: Tuple[int, int] = (5, 5), obstacle_grid: List[List[str]] = None):
        self.cm_per_cell = cm_per_cell
        self.grid_rows, self.grid_cols = grid_size
        self.obstacle_grid = obstacle_grid or []
        super().__init__("mistralai/mistral-7b-instruct:free", 300)
        logger.info(f"ActionPlanner initialized | Grid: {grid_size} | Cell size: {cm_per_cell}cm")

    def _build_action_prompt(self, waypoint, robot_pos, robot_facing):
        grid_visual = self._get_grid_visualization(robot_pos, robot_facing)

        return f"""### ROBOT NAVIGATION PLAN - STEP BY STEP ###
    ## TASK ##
    Move from START: {robot_pos} facing {robot_facing} to WAYPOINT: {waypoint}

    ## GRID STATE ##
    {grid_visual}

    ## MOVEMENT RULES ##
    1. TURN actions change facing:
       - Options: UP, DOWN, LEFT, RIGHT
       - Example: {{"type": "TURN", "direction": "RIGHT"}}

    2. MOVE actions go FORWARD in current facing:
       - Must specify cells (1-5)
       - Example: {{"type": "MOVE", "direction": "FORWARD", "cells": 2}}

    ## STEP-BY-STEP SOLUTION FOR THIS SCENARIO ##
    1. Current: (4,0) facing UP
    2. Required movement:
       - Vertical: 4 rows UP (from row 4 to row 0)
       - Horizontal: 3 columns RIGHT (from col 0 to col 3)
    3. Solution:
       - Turn RIGHT to face RIGHT (toward column 3)
       - Move FORWARD 3 cells to (4,3)
       - Turn UP to face UP (toward row 0)
       - Move FORWARD 4 cells to (0,3)

    ## CORRECT ACTION SEQUENCE ##
    [
      {{"type": "TURN", "direction": "RIGHT"}},
      {{"type": "MOVE", "direction": "FORWARD", "cells": 3}},
      {{"type": "TURN", "direction": "UP"}},
      {{"type": "MOVE", "direction": "FORWARD", "cells": 4}}
    ]

    ## YOUR TASK ##
    Generate ONLY the JSON array of actions that follows this exact pattern for this specific scenario.
    Output MUST be valid JSON with NO additional text.
    """
    def plan_actions(self, waypoint, robot_pos, robot_facing, max_retries=3):
        prompt = self._build_action_prompt(waypoint, robot_pos, robot_facing)

        for attempt in range(max_retries):
            try:
                llm_response = self._call_llm(prompt)
                actions = self._parse_and_validate_actions(
                    llm_response,
                    robot_pos,
                    robot_facing,
                    waypoint,
                    self.obstacle_grid  # Pass current obstacle grid
                )
                return actions
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed: {str(e)}")
                # Add error to prompt for next attempt
                prompt += f"\n\n## ERROR IN PREVIOUS ATTEMPT ##\n{str(e)}\n\nPlease correct your action sequence."

        logger.error("Action planning failed after retries")
        return []

        # In langgraph_nodes.py - ActionPlanner class
    # In langgraph_nodes.py - ActionPlanner class
    def _parse_and_validate_actions(self, response, start_pos, start_facing, waypoint, obstacle_grid):
        """Parse and validate actions with detailed simulation"""
        try:
            # Parse the JSON string
            actions_data = json.loads(response)

            # Handle single action object
            if isinstance(actions_data, dict):
                actions_data = [actions_data]

            # Parse each action
            actions = []
            for action_dict in actions_data:
                try:
                    # Normalize keys to lowercase
                    normalized = {k.lower(): v for k, v in action_dict.items()}
                    action = Action.parse_obj(normalized)
                    actions.append(action)
                except ValidationError as e:
                    logger.warning(f"Skipping invalid action: {str(e)}")
                    continue

            # Simulate actions with proper position tracking
            x, y = start_pos
            facing = start_facing
            position_log = [f"Start: ({x},{y}) facing {facing}"]
            rows, cols = self.grid_rows, self.grid_cols

            for i, action in enumerate(actions, 1):
                if action.type == "TURN":
                    # Handle turn directions
                    if action.direction == "RIGHT":
                        if facing == "UP": facing = "RIGHT"
                        elif facing == "RIGHT": facing = "DOWN"
                        elif facing == "DOWN": facing = "LEFT"
                        elif facing == "LEFT": facing = "UP"
                    elif action.direction == "LEFT":
                        if facing == "UP": facing = "LEFT"
                        elif facing == "LEFT": facing = "DOWN"
                        elif facing == "DOWN": facing = "RIGHT"
                        elif facing == "RIGHT": facing = "UP"
                    elif action.direction in ["UP", "DOWN"]:
                        facing = action.direction
                    position_log.append(f"Action {i}: TURN {action.direction} → Now facing {facing}")

                elif action.type == "MOVE" and action.direction == "FORWARD":
                    if not isinstance(action.cells, int) or action.cells <= 0:
                        raise ValueError(f"Invalid cell count: {action.cells}")

                    # Calculate movement vector based on current facing
                    dx, dy = 0, 0
                    if facing == "UP": dy = -1
                    elif facing == "DOWN": dy = 1
                    elif facing == "LEFT": dx = -1
                    elif facing == "RIGHT": dx = 1
                    else:
                        raise ValueError(f"Invalid facing direction: {facing}")

                    # Process each cell movement step-by-step
                    for step in range(1, action.cells + 1):
                        new_x, new_y = x + dx, y + dy

                        # Check boundaries
                        if not (0 <= new_x < rows and 0 <= new_y < cols):
                            raise ValueError(
                                f"Action {i} step {step}: Position ({new_x},{new_y}) "
                                f"out of grid bounds {self.grid_size}"
                            )

                        # Check obstacles
                        if obstacle_grid[new_x][new_y] == '■':
                            raise ValueError(
                                f"Action {i} step {step}: Obstacle at ({new_x},{new_y})"
                            )

                        # Update position
                        x, y = new_x, new_y
                        position_log.append(f"  Step {step}: Moved to ({x},{y})")

                    position_log.append(f"Action {i} complete: Moved {action.cells} cells to ({x},{y})")

            # Final position check
            if (x, y) != waypoint:
                position_log.append(
                    f"Navigation failed: Final position ({x},{y}) ≠ waypoint {waypoint}"
                )
                raise ValueError("\n".join(position_log))

            return actions

        except Exception as e:
            error_msg = f"Action validation failed: {str(e)}\nSimulation trace:\n" + "\n".join(position_log)
            raise ValueError(error_msg)
    def _get_grid_visualization(self, robot_pos, robot_facing):
        """Create a visual grid with robot and obstacles"""
        grid = [row[:] for row in self.obstacle_grid]
        r, c = robot_pos

        # Place robot
        arrows = {"UP": "↑", "DOWN": "↓", "LEFT": "←", "RIGHT": "→"}
        grid[r][c] = arrows.get(robot_facing, "R")

        # Place waypoint
        if hasattr(self, 'goal_position'):
            gr, gc = self.goal_position
            grid[gr][gc] = 'G'

        return "\n".join(" ".join(row) for row in grid)