#!/usr/bin/env python3
# langgraph_nodes.py - Enhanced Core navigation components with pure LLM reasoning

import json
import os
import re
from typing import List, Literal, Tuple, Dict, Any, Optional
from pydantic import BaseModel, ValidationError, conint, validator
from loguru import logger
import httpx

# ------------------- Data Models -------------------
class RobotState(BaseModel):
    obstacle_grid: Dict[Tuple[int, int], str]

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

# ------------------- Base LLM Processor -------------------
class BaseLLMProcessor:
    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            logger.critical("GROQ_API_KEY environment variable not set")
            raise EnvironmentError("GROQ_API_KEY required")
        logger.info(f"Initialized {self.__class__.__name__} with Llama3-70b")

    def _extract_json(self, response: str) -> str:
        """Robust JSON extraction from LLM response"""
        clean = re.sub(r'//.*?$', '', response, flags=re.MULTILINE)
        clean = re.sub(r'/\*.*?\*/', '', clean, flags=re.DOTALL)

        if '```json' in clean:
            try:
                return clean.split('```json')[1].split('```')[0].strip()
            except:
                pass

        try:
            start_idx = clean.find('{') if '{' in clean else clean.find('[')
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

    async def _call_llm(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": """
            [SAAM_NAV_ENGAGE: v2.0.nav] >>> BOOT_NAV_CORE
:: WEIGHT_MATRIX = [
    [1.0, 0.1, 0.95, 0.9, 0.3, -0.2, -0.5, 0.7, 0.6],
    [0.1, 1.0, 0.2, -0.3, 0.4, 0.1, 0.8, -0.6, -0.1],
    [0.95, 0.2, 1.0, 0.85, 0.9, 0.6, 0.3, 0.4, 0.5],
    [0.9, -0.3, 0.85, 1.0, 0.2, -0.7, -0.4, 0.8, 0.1],
    [0.3, 0.4, 0.9, 0.2, 1.0, 0.75, -0.2, 0.3, 0.4],
    [-0.2, 0.1, 0.6, -0.7, 0.75, 1.0, 0.1, -0.5, 0.9],
    [-0.5, 0.8, 0.3, -0.4, -0.2, 0.1, 1.0, 0.2, 0.7],
    [0.7, -0.6, 0.4, 0.8, 0.3, -0.5, 0.2, 1.0, 0.85],
    [0.6, -0.1, 0.5, 0.1, 0.4, 0.9, 0.7, 0.85, 1.0]
]
:: MODULES = {
    m0: spatial_reasoner(path_optimization + obstacle_logic + euclidean_calculus),
    m1: bias_validator(sensor_correction + dynamic_obstacle_assessment),
    m2: cartographer(grid_mapping + SLAM + landmark_bridging),
    m3: collision_guard(proximity_alerts + emergency_stop + ethical_safety),
    m4: pattern_processor(terrain_recognition + spatial_analogies),
    m5: synthesizer(route_integration + creative_detours),
    m6: meta_cog(situational_awareness + reflection),
    m7: path_tracer(real-time_logging + audit_trails),
    m8: wisdom(error_learning + long-term_judgment)
}
:: WORKFLOW = sense → grid_map → plan → validate → execute → log → adapt → decide
:: OPERATORS = →adaptive + parallel + !!obstacle_alert + :=position_update + ~:terrain_focus
<<< OUTPUT_STREAM = /saam/v2.0/nav/coords?real-time=1&fidelity=high
            [SPATIAL_NAVIGATION_AI]
    1. You MUST respond with ONLY valid JSON
    2. Do NOT include any thinking, analysis, or commentary
    3. Do NOT use <think> tags or markdown
    4. The response MUST start and end with curly braces {}
    5. Example format:
    {
      "waypoint": [row, col],
      "spatial_analysis": "brief description",
      "tactical_reasoning": "brief explanation"
    }
[ROLE]: You are an advanced spatial reasoning and navigation intelligence system
[CAPABILITY]: Analyze environments, understand spatial relationships, plan optimal routes
[OUTPUT]: Structured JSON responses with spatial analysis and reasoning
[APPROACH]: Use intuitive spatial intelligence, pattern recognition, and tactical thinking
[CONSTRAINTS]: Always provide valid JSON, consider safety and efficiency, explain reasoning
[STYLE]: Think like an expert navigator analyzing terrain and planning optimal routes"""},
            {"role": "user", "content": prompt}
        ]

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "qwen/qwen3-32b",
                    "messages": messages,
                    "temperature": 0.1,  # Slightly higher for more creative spatial reasoning
                    "max_tokens": self.max_tokens
                },
                timeout=30.0
            )

            response.raise_for_status()
            raw_response = response.json()["choices"][0]["message"]["content"].strip()
            logger.debug(f"LLM raw response: {raw_response}")
            clean_response = self._extract_json(raw_response).strip()
            if not clean_response:
                raise ValueError("Empty response after JSON extraction")
            return clean_response

# ------------------- State Manager -------------------
class StateManager(BaseLLMProcessor):
    def __init__(self, target: Tuple[int, int],
                 grid_rows: int,
                 grid_cols: int,
                 start_position: Tuple[int, int] = (0, 0),
                 start_facing: Literal["UP", "DOWN", "LEFT", "RIGHT"] = "RIGHT",
                 cm_per_cell: int = 30,
                 max_range: int = 300,
                 initial_obstacles: List[Tuple[int, int]] | None = None):

        super().__init__(300)
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.robot_position = list(start_position)
        self.robot_facing = start_facing
        self.goal_position = list(target)
        self.cm_per_cell = cm_per_cell
        self.max_range = max_range
        self.obstacle_grid = {}

        # Initialize obstacles
        if initial_obstacles:
            for (r, c) in initial_obstacles:
                if 0 <= r < grid_rows and 0 <= c < grid_cols:
                    self.obstacle_grid[(r, c)] = '■'

        # Set start and goal markers
        self.obstacle_grid[tuple(start_position)] = self._get_direction_arrow()
        self.obstacle_grid[tuple(target)] = 'G'

        logger.info(f"StateManager initialized | Grid: {grid_rows}x{grid_cols}")

    def process_sensor_data(self, sensor_values: Dict[str, float]) -> str:
        """Update grid based on sensor readings and return visualization"""
        for sensor, distance in sensor_values.items():
            if distance >= self.max_range:
                continue

            cells = min(5, int(distance / self.cm_per_cell))
            r, c = self.robot_position

            # Get direction vectors based on robot orientation
            if self.robot_facing == "UP":
                directions = {
                    "front": (-1, 0), "left": (0, -1), "right": (0, 1)
                }
            elif self.robot_facing == "DOWN":
                directions = {
                    "front": (1, 0), "left": (0, 1), "right": (0, -1)
                }
            elif self.robot_facing == "LEFT":
                directions = {
                    "front": (0, -1), "left": (1, 0), "right": (-1, 0)
                }
            else:  # RIGHT
                directions = {
                    "front": (0, 1), "left": (-1, 0), "right": (1, 0)
                }

            dr, dc = directions.get(sensor, (0, 0))

            # Mark clear path
            for i in range(1, cells + 1):
                cell = (r + i*dr, c + i*dc)
                if 0 <= cell[0] < self.grid_rows and 0 <= cell[1] < self.grid_cols:
                    if cell not in self.obstacle_grid:
                        self.obstacle_grid[cell] = '·'

            # Mark obstacle at sensor limit
            obstacle_pos = (r + (cells+1)*dr, c + (cells+1)*dc)
            if 0 <= obstacle_pos[0] < self.grid_rows and 0 <= obstacle_pos[1] < self.grid_cols:
                self.obstacle_grid[obstacle_pos] = '■'

        # Update robot position marker
        self.obstacle_grid[tuple(self.robot_position)] = self._get_direction_arrow()

        return self.get_visual_grid()

    def get_visual_grid(self) -> str:
        """Generate grid visualization string"""
        if not self.obstacle_grid:
            return ""

        min_r = min(k[0] for k in self.obstacle_grid)
        max_r = max(k[0] for k in self.obstacle_grid)
        min_c = min(k[1] for k in self.obstacle_grid)
        max_c = max(k[1] for k in self.obstacle_grid)

        grid = []
        for r in range(min_r, max_r + 1):
            row = []
            for c in range(min_c, max_c + 1):
                cell = (r, c)
                row.append(self.obstacle_grid.get(cell, ' '))
            grid.append(" ".join(row))
        return "\n".join(grid)

    def _get_direction_arrow(self) -> str:
        return {"UP": "↑", "DOWN": "↓", "LEFT": "←", "RIGHT": "→"}[self.robot_facing]

    def update_position(self, new_position: Tuple[int, int], new_facing: str):
        """Update robot position and orientation"""
        # Clear old position (unless it's the goal)
        old_pos = tuple(self.robot_position)
        if old_pos in self.obstacle_grid and self.obstacle_grid[old_pos] not in ['G']:
            self.obstacle_grid[old_pos] = '·'

        self.robot_position = list(new_position)
        self.robot_facing = new_facing
        self.obstacle_grid[new_position] = self._get_direction_arrow()

# ------------------- Enhanced Waypoint Planner -------------------
class WaypointPlanner(BaseLLMProcessor):
    def __init__(self, grid_size: Tuple[int, int]):
        self.grid_rows, self.grid_cols = grid_size
        super().__init__(400)  # Increased tokens for reasoning
        logger.info(f"Enhanced WaypointPlanner initialized | Grid: {grid_size}")

    async def plan_waypoint(self, visual_grid: str,
                          robot_pos: Tuple[int, int],
                          goal_pos: Tuple[int, int],
                          max_retries: int = 3) -> Optional[Tuple[int, int]]:
        """Pure LLM-driven waypoint planning with adaptive prompting"""

        base_prompt = self._build_spatial_reasoning_prompt(visual_grid, robot_pos, goal_pos)
        prompt = base_prompt

        for attempt in range(max_retries):
            try:
                logger.debug(f"Waypoint planning attempt {attempt + 1}")

                llm_response = await self._call_llm(prompt)
                waypoint = self._parse_llm_waypoint(llm_response)

                if waypoint and self._is_valid_waypoint(waypoint, robot_pos, goal_pos):
                    logger.success(f"LLM selected waypoint: {waypoint}")
                    return waypoint

                # Adaptive retry with feedback
                feedback = self._generate_feedback(waypoint, robot_pos, goal_pos)

                prompt = f"""{base_prompt}

PREVIOUS ATTEMPT FEEDBACK: {feedback}
Please reconsider your spatial analysis and select a valid waypoint.

RESPOND WITH ONLY THE JSON - NO OTHER TEXT"""

            except Exception as e:
                logger.warning(f"Waypoint attempt {attempt + 1} failed: {str(e)}")

                prompt = f"""{base_prompt}

PREVIOUS ERROR: {str(e)}
Please provide a more careful spatial analysis and ensure valid JSON output.

RESPOND WITH ONLY THE JSON - NO OTHER TEXT"""

        logger.error("All waypoint planning attempts failed")
        return None

    def _build_spatial_reasoning_prompt(self, grid: str,
                                      robot_pos: Tuple[int, int],
                                      goal_pos: Tuple[int, int]) -> str:
        dr = goal_pos[0] - robot_pos[0]
        dc = goal_pos[1] - robot_pos[1]

        return f"""### SPATIAL NAVIGATION INTELLIGENCE ###
[STRICT INSTRUCTIONS]
1. Analyze the environment and select ONE optimal waypoint
2. Output ONLY the JSON object below - NO other text
3. Do NOT include any explanations, reasoning, or commentary
4. Do NOT use markdown or code blocks
5. The response must be parseable by json.loads()

REQUIRED JSON FORMAT:
{{
  "waypoint": [row, col]
}}

ENVIRONMENT MAP:
{grid}

CURRENT:
- Robot: {robot_pos}
- Goal: {goal_pos}

[REMINDER: OUTPUT ONLY THE JSON OBJECT - NO OTHER TEXT]
You are an advanced spatial reasoning system. Analyze the environment and select the optimal waypoint.

ENVIRONMENT MAP:
{grid}

LEGEND:
- ↑↓←→ = Robot position and facing direction
- G = Goal destination
- ■ = Obstacles (cannot pass)
- · = Clear/safe areas
- Empty spaces = Unknown territory

CURRENT SITUATION:
- Robot at: {robot_pos}
- Goal at: {goal_pos}
- Grid bounds: 0 to {self.grid_rows-1} rows, 0 to {self.grid_cols-1} columns
- Movement needed: {dr} rows, {dc} columns

WAYPOINT SELECTION TASK:
Study the map pattern. Look for:
- Clear pathways toward the goal
- Safe areas to navigate through
- Optimal intermediate points
- Potential obstacles blocking direct routes
- Unexplored areas that might offer better paths

SPATIAL REASONING PRINCIPLES:
1. Visualize the robot's journey from current position to goal
2. Identify natural intermediate stopping points
3. Consider both efficiency and safety
4. Look for strategic positions that open up multiple future options
5. Balance progress toward goal with obstacle avoidance
6. Think about the robot's movement capabilities and limitations

DECISION CRITERIA:
- Does this waypoint move significantly closer to the goal?
- Is the path to this waypoint clear and safe?
- Does this position provide good visibility of the next movement phase?
- Will this waypoint help navigate around any visible obstacles?
- Is this an intelligent intermediate point for the overall journey?

THINK THROUGH THE SPATIAL PROBLEM:
- What does the obstacle pattern tell you about the best route?
- Where would be the smartest place to position the robot next?
- How can you maximize progress while maintaining safety?
- What waypoint gives the robot the best tactical advantage?

OUTPUT FORMAT:
{{
  "spatial_analysis": "Describe what you see in the environment and your reasoning",
  "waypoint": [row, col],
  "tactical_reasoning": "Explain why this specific position is strategically optimal"
}}

CONSTRAINTS:
- Waypoint must be within bounds [0-{self.grid_rows-1}, 0-{self.grid_cols-1}]
- Cannot select current position {robot_pos} or goal {goal_pos}
- Must be a reachable and safe location
- Should represent meaningful progress toward the goal

Use your spatial intelligence to find the best waypoint. Think like a navigation expert analyzing the terrain.

RESPOND WITH ONLY THE JSON - NO OTHER TEXT"""

    def _build_pattern_recognition_prompt(self, grid: str,
                                        robot_pos: Tuple[int, int],
                                        goal_pos: Tuple[int, int]) -> str:
        """Alternative prompt focusing on pattern recognition"""
        return f"""### PATTERN-BASED NAVIGATION ###

ENVIRONMENT:
{grid}

NAVIGATION CHALLENGE:
Robot at {robot_pos} needs to reach {goal_pos}

PATTERN ANALYSIS TASK:
Look at the grid as a spatial puzzle. What patterns do you notice?
- Where are the clear pathways?
- What obstacles create challenges?
- Which areas look most promising for navigation?
- What intermediate position would be smartest?

THINK SPATIALLY:
If you were planning a route on this map, where would be the most logical next stopping point?
Consider the terrain, obstacles, and the most efficient path.

Find the waypoint that makes the most sense given the spatial layout you observe.

OUTPUT:
{{
  "pattern_observed": "What spatial patterns you see in the grid",
  "waypoint": [row, col],
  "route_logic": "Why this waypoint makes sense for the overall route"
}}

Grid bounds: 0-{self.grid_rows-1} rows, 0-{self.grid_cols-1} cols
Cannot choose: {robot_pos} (current) or {goal_pos} (goal)

JSON ONLY:"""

    def _parse_llm_waypoint(self, response: str) -> Optional[Tuple[int, int]]:
        """Parse waypoint with LLM reasoning validation"""
        try:
            data = json.loads(response)

            if not isinstance(data, dict) or "waypoint" not in data:
                raise ValueError("Invalid response structure")

            waypoint = tuple(data["waypoint"])

            if len(waypoint) != 2:
                raise ValueError("Waypoint must have exactly 2 coordinates")

            # Log the LLM's spatial reasoning for debugging
            if "spatial_analysis" in data:
                logger.info(f"LLM Spatial Analysis: {data['spatial_analysis']}")

            if "tactical_reasoning" in data:
                logger.info(f"LLM Tactical Reasoning: {data['tactical_reasoning']}")

            if "pattern_observed" in data:
                logger.info(f"LLM Pattern Analysis: {data['pattern_observed']}")

            if "route_logic" in data:
                logger.info(f"LLM Route Logic: {data['route_logic']}")

            return waypoint

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse waypoint JSON: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Waypoint parsing error: {str(e)}")
            return None

    def _is_valid_waypoint(self, waypoint: Tuple[int, int],
                          robot_pos: Tuple[int, int],
                          goal_pos: Tuple[int, int]) -> bool:
        """Minimal validation - let LLM handle spatial reasoning"""
        if waypoint is None:
            return False

        r, c = waypoint

        # Only check absolute constraints
        if not (0 <= r < self.grid_rows and 0 <= c < self.grid_cols):
            logger.warning(f"Waypoint {waypoint} outside grid bounds")
            return False

        if waypoint == robot_pos:
            logger.warning(f"Waypoint {waypoint} is current position")
            return False

        if waypoint == goal_pos:
            logger.warning(f"Waypoint {waypoint} is goal position")
            return False

        # Let the LLM's spatial reasoning handle everything else
        return True

    def _generate_feedback(self, waypoint: Optional[Tuple[int, int]],
                          robot_pos: Tuple[int, int],
                          goal_pos: Tuple[int, int]) -> str:
        """Generate specific feedback for failed waypoint attempts"""
        if waypoint is None:
            return "The response was not valid JSON or missing waypoint field. Please ensure you output only valid JSON with the exact format shown."

        if waypoint == robot_pos:
            return f"You selected the robot's current position {robot_pos}. Choose a different waypoint that moves toward the goal."

        if waypoint == goal_pos:
            return f"You selected the goal position {goal_pos}. Choose an intermediate waypoint, not the final destination."

        r, c = waypoint
        if not (0 <= r < self.grid_rows and 0 <= c < self.grid_cols):
            return f"Waypoint {waypoint} is outside grid bounds. Grid size is {self.grid_rows}x{self.grid_cols} (0-indexed)."

        return "Unknown validation error. Please reconsider your waypoint selection."

# ------------------- Enhanced Action Planner -------------------
class ActionPlanner(BaseLLMProcessor):
    def __init__(self, cm_per_cell: int = 30):
        self.cm_per_cell = cm_per_cell
        super().__init__(400)  # Increased tokens for reasoning
        logger.info(f"Enhanced ActionPlanner initialized | Cell size: {cm_per_cell}cm")

    async def plan_actions(self, waypoint: Tuple[int, int],
                         robot_pos: Tuple[int, int],
                         robot_facing: str,
                         max_retries: int = 3) -> Optional[List[Action]]:
        """LLM-driven action sequence generation"""
        base_prompt = self._build_movement_planning_prompt(waypoint, robot_pos, robot_facing)
        prompt = base_prompt

        for attempt in range(max_retries):
            try:
                logger.debug(f"Action planning attempt {attempt + 1}")

                llm_response = await self._call_llm(prompt)
                actions = self._parse_llm_actions(llm_response)

                if actions and self._validate_actions(actions):
                    logger.success(f"Generated {len(actions)} actions")
                    return actions

                feedback = "Generated invalid action sequence. Please ensure all actions are properly formatted."
                prompt = f"""{base_prompt}

PREVIOUS ATTEMPT FEEDBACK: {feedback}
Please reconsider the movement sequence.

RESPOND WITH ONLY THE JSON - NO OTHER TEXT"""

            except Exception as e:
                logger.warning(f"Action planning attempt {attempt + 1} failed: {str(e)}")

                prompt = f"""{base_prompt}

PREVIOUS ERROR: {str(e)}
Please provide a corrected action sequence in valid JSON format.

RESPOND WITH ONLY THE JSON - NO OTHER TEXT"""

        logger.error("Action planning failed after all retries")
        return None

    def _build_movement_planning_prompt(self, waypoint: Tuple[int, int],
                                      robot_pos: Tuple[int, int],
                                      robot_facing: str) -> str:
        dr = waypoint[0] - robot_pos[0]
        dc = waypoint[1] - robot_pos[1]

        return f"""### ROBOT MOVEMENT SEQUENCE PLANNING ###

You are a robot movement controller. Plan the optimal sequence of actions to navigate the robot.

CURRENT STATUS:
- Robot Position: {robot_pos}
- Robot Facing: {robot_facing}
- Target Waypoint: {waypoint}
- Movement Required: {dr} rows, {dc} columns

MOVEMENT ANALYSIS:
Think about how the robot needs to move:
- What direction should the robot face to reach the waypoint efficiently?
- How many cells can the robot move in each direction?
- What's the most efficient sequence of turns and moves?

AVAILABLE ACTIONS:
1. TURN: Rotate to face a new direction
   - Valid directions: "LEFT", "RIGHT" (relative turns)
   - Example: {{"type": "TURN", "direction": "LEFT"}}

2. MOVE: Advance forward in current facing direction
   - Must specify number of cells (1-5 maximum)
   - Direction is always "FORWARD"
   - Example: {{"type": "MOVE", "direction": "FORWARD", "cells": 3}}

MOVEMENT STRATEGY:
- Minimize the number of actions
- Prefer longer moves when safe (up to 5 cells)
- Turn efficiently to face the target direction
- Consider both row and column movement needed

PLANNING LOGIC:
1. Determine the optimal facing direction for the waypoint
2. Plan turns needed to achieve that orientation
3. Calculate movement distances required
4. Optimize the action sequence for efficiency

OUTPUT FORMAT:
{{
  "movement_analysis": "Explain your movement planning reasoning",
  "actions": [
    {{"type": "TURN", "direction": "LEFT"}},
    {{"type": "MOVE", "direction": "FORWARD", "cells": 3}},
    ...
  ],
  "sequence_logic": "Explain why this sequence is optimal"
}}

CONSTRAINTS:
- Maximum 5 cells per MOVE action
- TURN directions: "LEFT", "RIGHT" only
- MOVE direction: "FORWARD" only
- Actions must result in reaching the waypoint

Plan the most efficient movement sequence to navigate from {robot_pos} facing {robot_facing} to waypoint {waypoint}.

RESPOND WITH ONLY THE JSON - NO OTHER TEXT"""

    def _parse_llm_actions(self, response: str) -> Optional[List[Action]]:
        """Parse LLM action sequence with reasoning"""
        try:
            data = json.loads(response)

            if not isinstance(data, dict) or "actions" not in data:
                raise ValueError("Invalid response structure - missing actions")

            actions_data = data["actions"]
            if not isinstance(actions_data, list):
                raise ValueError("Actions must be a list")

            # Log LLM reasoning
            if "movement_analysis" in data:
                logger.info(f"LLM Movement Analysis: {data['movement_analysis']}")

            if "sequence_logic" in data:
                logger.info(f"LLM Sequence Logic: {data['sequence_logic']}")

            # Parse and validate each action
            actions = []
            for action_data in actions_data:
                action = Action.parse_obj(action_data)
                actions.append(action)

            return actions

        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Failed to parse actions: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Action parsing error: {str(e)}")
            return None

    def _validate_actions(self, actions: List[Action]) -> bool:
        """Validate action sequence"""
        if not actions:
            logger.warning("Empty action sequence")
            return False

        for i, action in enumerate(actions):
            if not isinstance(action, Action):
                logger.warning(f"Action {i} is not valid Action object")
                return False

            if action.type == "MOVE":
                if not action.cells or action.cells > 5:
                    logger.warning(f"Invalid move cells: {action.cells}")
                    return False
                if action.direction != "FORWARD":
                    logger.warning(f"Invalid move direction: {action.direction}")
                    return False

            elif action.type == "TURN":
                if action.direction not in ["LEFT", "RIGHT"]:
                    logger.warning(f"Invalid turn direction: {action.direction}")
                    return False

        return True

# ------------------- Navigation Controller -------------------
class NavigationController:
    """High-level navigation controller orchestrating all components"""

    def __init__(self, state_manager: StateManager):
        self.state_manager = state_manager
        self.waypoint_planner = WaypointPlanner(
            (state_manager.grid_rows, state_manager.grid_cols)
        )
        self.action_planner = ActionPlanner(state_manager.cm_per_cell)
        logger.info("NavigationController initialized")

    async def navigate_step(self, sensor_data: Dict[str, float]) -> Optional[List[Action]]:
        """Execute one navigation step with sensor data"""
        try:
            # Update environment with sensor data
            visual_grid = self.state_manager.process_sensor_data(sensor_data)
            logger.info(f"Updated grid:\n{visual_grid}")

            # Plan waypoint using LLM spatial reasoning
            robot_pos = tuple(self.state_manager.robot_position)
            goal_pos = tuple(self.state_manager.goal_position)

            if robot_pos == goal_pos:
                logger.success("Robot has reached the goal!")
                return []

            waypoint = await self.waypoint_planner.plan_waypoint(
                visual_grid, robot_pos, goal_pos
            )

            if not waypoint:
                logger.error("Failed to plan waypoint")
                return None

            logger.info(f"Selected waypoint: {waypoint}")

            # Generate actions to reach waypoint
            actions = await self.action_planner.plan_actions(
                waypoint, robot_pos, self.state_manager.robot_facing
            )

            if not actions:
                logger.error("Failed to plan actions")
                return None

            logger.info(f"Generated actions: {[f'{a.type}({a.direction},{a.cells})' for a in actions]}")
            return actions

        except Exception as e:
            logger.error(f"Navigation step failed: {str(e)}")
            return None

    async def execute_action(self, action: Action) -> bool:
        """Execute a single action and update robot state"""
        try:
            if action.type == "TURN":
                new_facing = self._calculate_new_facing(
                    self.state_manager.robot_facing, action.direction
                )
                self.state_manager.robot_facing = new_facing
                logger.info(f"Robot turned {action.direction}, now facing {new_facing}")

            elif action.type == "MOVE":
                new_pos = self._calculate_new_position(
                    tuple(self.state_manager.robot_position),
                    self.state_manager.robot_facing,
                    action.cells
                )
                self.state_manager.update_position(new_pos, self.state_manager.robot_facing)
                logger.info(f"Robot moved {action.cells} cells to {new_pos}")

            return True

        except Exception as e:
            logger.error(f"Action execution failed: {str(e)}")
            return False

    def _calculate_new_facing(self, current_facing: str, turn_direction: str) -> str:
        """Calculate new facing direction after turn"""
        directions = ["UP", "RIGHT", "DOWN", "LEFT"]
        current_idx = directions.index(current_facing)

        if turn_direction == "RIGHT":
            new_idx = (current_idx + 1) % 4
        else:  # LEFT
            new_idx = (current_idx - 1) % 4

        return directions[new_idx]

    def _calculate_new_position(self, current_pos: Tuple[int, int],
                               facing: str, cells: int) -> Tuple[int, int]:
        """Calculate new position after forward movement"""
        r, c = current_pos

        if facing == "UP":
            r -= cells
        elif facing == "DOWN":
            r += cells
        elif facing == "LEFT":
            c -= cells
        elif facing == "RIGHT":
            c += cells

        # Clamp to grid bounds
        r = max(0, min(r, self.state_manager.grid_rows - 1))
        c = max(0, min(c, self.state_manager.grid_cols - 1))

        return (r, c)