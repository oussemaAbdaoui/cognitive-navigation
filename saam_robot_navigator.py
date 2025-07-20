# ----- saam_robot_navigator.py -----
#!/usr/bin/env python3
import json
import os
import time
import hashlib
from functools import lru_cache
from typing import List, Tuple, Dict, Any, Optional
from pydantic import BaseModel, ValidationError, conint, Field, validator
from loguru import logger
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from enum import Enum
import numpy as np

class Direction(Enum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"

    @classmethod
    def turn_right(cls, current):
        return {
            cls.UP: cls.RIGHT,
            cls.RIGHT: cls.DOWN,
            cls.DOWN: cls.LEFT,
            cls.LEFT: cls.UP
        }[current]

    @classmethod
    def turn_left(cls, current):
        return {
            cls.UP: cls.LEFT,
            cls.LEFT: cls.DOWN,
            cls.DOWN: cls.RIGHT,
            cls.RIGHT: cls.UP
        }[current]

class SAAMModule(Enum):
    REASONER = "m0:reasoner(pathfinding)"
    VALIDATOR = "m1:validator(rule_enforcement)"
    MAPPER = "m2:mapper(topology)"
    GUARD = "m3:guard(safety)"
    PROCESSOR = "m4:processor(sensor_fusion)"
    SYNTHESIZER = "m5:synthesizer(action_planning)"
    META_COG = "m6:meta_cog(reflection)"
    TRACER = "m7:tracer(logging)"
    WISDOM = "m8:wisdom(recovery)"

class SAAMController:
    def __init__(self):
        self.modules = {m.name: m.value for m in SAAMModule}
        self.weight_matrix = np.eye(len(SAAMModule))
        logger.info("SAAM Controller initialized")

    def generate_prompt(self, task: str, priority_modules: List[SAAMModule],
                      input_data: Dict[str, Any], escape_trigger: Optional[str] = None) -> str:
        if "grid" in input_data:
            rows = len(input_data["grid"])
            cols = len(input_data["grid"][0]) if rows > 0 else 0
            input_data["boundaries"] = {
                "min_row": 0,
                "max_row": rows - 1,
                "min_col": 0,
                "max_col": cols - 1
            }

        prompt = f"""=== SAAM NAVIGATION PROMPT ===
TASK: {task}

SAFETY CONSTRAINTS:
1. ALL movements must stay within grid boundaries
2. NEVER output positions outside these boundaries:
   - Rows: 0 to {input_data.get('boundaries', {}).get('max_row', '?')}
   - Columns: 0 to {input_data.get('boundaries', {}).get('max_col', '?')}
3. MUST validate all positions in your response

CURRENT STATE:
{json.dumps(input_data, indent=2)}

RESPONSE REQUIREMENTS:
- MUST be valid JSON
- MUST include ALL requested fields
- MUST respect ALL safety constraints"""

        if escape_trigger:
            prompt += f"\n\nSAFETY PROTOCOL: If {escape_trigger} occurs, activate recovery procedures"

        return prompt

class RobotState(BaseModel):
    robot_position: Tuple[conint(ge=0), conint(ge=0)] = Field(..., description="(row, col) coordinates")
    robot_facing: Direction
    obstacle_grid: List[List[str]]

    @validator('obstacle_grid')
    def validate_grid_size(cls, v):
        if not v or not all(len(row) == len(v[0]) for row in v):
            raise ValueError("Grid must be rectangular")
        return v

class NavigationStep(BaseModel):
    waypoint: Tuple[conint(ge=0), conint(ge=0)]
    actions: List[Dict[str, Any]]
    confidence: float = Field(..., ge=0, le=1)

class EmergencyStop(Exception):
    pass

class SAAMNavigator:
    def __init__(self, model: str = "cognitivecomputations/dolphin3.0-mistral-24b:free"):
        self.saam = SAAMController()
        self.llm = ChatOpenAI(
            model=model,
            temperature=0.0,
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            timeout=30.0,
            max_retries=3,
            request_timeout=45.0
        )
        self.grid = None
        self.robot_pos = (0, 0)
        self.robot_facing = Direction.RIGHT
        self.goal_pos = (0, 0)
        self.failure_history = []
        logger.info("SAAM Navigator initialized")

    def reset_state(self, grid, robot_pos, robot_facing, goal_pos):
        self.grid = grid
        self.robot_pos = robot_pos
        self.robot_facing = Direction(robot_facing)
        self.goal_pos = goal_pos
        self.failure_history = []

    def _hash_prompt(self, prompt: str) -> str:
        return hashlib.md5(prompt.encode()).hexdigest()

    @lru_cache(maxsize=128)
    def _call_saam_llm_cached(self, prompt_hash: str, prompt: str) -> Dict[str, Any]:
        messages = [
            SystemMessage(content="You are a SAAM robotic controller. Respond ONLY with valid JSON that respects ALL safety constraints."),
            HumanMessage(content=prompt)
        ]

        try:
            response = self.llm.invoke(messages).content
            logger.debug(f"LLM Response: {response}")
            result = json.loads(response)

            if not isinstance(result, dict):
                raise ValueError("Response must be a JSON object")
            return result

        except json.JSONDecodeError:
            logger.error("LLM returned invalid JSON")
            raise
        except Exception as e:
            logger.error(f"LLM error: {str(e)}")
            raise

    def _call_saam_llm(self, prompt: str, retries: int = 2) -> Dict[str, Any]:
        prompt_hash = self._hash_prompt(prompt)

        for attempt in range(retries + 1):
            try:
                return self._call_saam_llm_cached(prompt_hash, prompt)
            except Exception as e:
                if attempt < retries:
                    logger.warning(f"Retrying LLM call ({attempt+1}/{retries})")
                    time.sleep(1.5 ** attempt)  # Exponential backoff
                else:
                    logger.error("LLM call failed after retries")
                    raise

    def update_grid(self, sensor_data: Dict[str, float]) -> str:
        prompt = self.saam.generate_prompt(
            task="Update robot state from sensor data",
            priority_modules=[SAAMModule.PROCESSOR, SAAMModule.MAPPER],
            input_data={
                "sensors": sensor_data,
                "current_state": {
                    "position": list(self.robot_pos),
                    "facing": self.robot_facing.value,
                    "grid": self.grid
                },
                "grid_dimensions": {
                    "rows": len(self.grid),
                    "cols": len(self.grid[0])
                },
                "required_fields": ["robot_position", "robot_facing", "obstacle_grid"],
                "example_response": {
                    "robot_position": list(self.robot_pos),
                    "robot_facing": self.robot_facing.value,
                    "obstacle_grid": self.grid,
                    "boundary_check": "Position validated within grid"
                }
            }
        )

        try:
            result = self._call_saam_llm(prompt)
            new_pos = tuple(result["robot_position"])
            new_facing = Direction(result["robot_facing"])

            # Validate new position
            if not (0 <= new_pos[0] < len(self.grid) and
                    0 <= new_pos[1] < len(self.grid[0])):
                raise ValidationError(
                    f"Position {new_pos} out of grid bounds"
                )

            # Validate obstacle grid
            if "obstacle_grid" in result:
                new_grid = result["obstacle_grid"]
                if (len(new_grid) != len(self.grid) or
                    any(len(row) != len(self.grid[0]) for row in new_grid)):
                    raise ValidationError("Invalid grid dimensions in update")
                self.grid = new_grid

            self.robot_pos = new_pos
            self.robot_facing = new_facing
            return self._visualize_grid()

        except ValidationError as ve:
            logger.warning(f"Validation failed: {str(ve)}")
            return self._visualize_grid()
        except Exception as e:
            logger.error(f"Grid update failed: {str(e)}")
            return self._visualize_grid()

    def plan_route(self) -> NavigationStep:
        prompt = self.saam.generate_prompt(
            task="Plan path from current to goal position",
            priority_modules=[SAAMModule.REASONER, SAAMModule.SYNTHESIZER],
            input_data={
                "current": list(self.robot_pos),
                "facing": self.robot_facing.value,
                "goal": list(self.goal_pos),
                "grid": self.grid,
                "required_fields": ["waypoint", "actions"],
                "example_response": {
                    "waypoint": [1, 1],
                    "actions": [
                        {"type": "TURN", "direction": "RIGHT"},
                        {"type": "MOVE", "direction": "FORWARD", "cells": 1}
                    ],
                    "safety_check": "All movements stay within boundaries"
                }
            }
        )

        try:
            result = self._call_saam_llm(prompt)
            actions = []
            for action in result.get("actions", []):
                # Validate action structure
                if not isinstance(action, dict):
                    continue

                action_type = action.get("type")
                direction = action.get("direction")

                if action_type == "TURN" and direction in ["LEFT", "RIGHT"]:
                    actions.append({
                        "type": "TURN",
                        "direction": direction
                    })
                elif (action_type == "MOVE" and
                      direction == "FORWARD" and
                      isinstance(action.get("cells", 1), (int, float))):
                    actions.append({
                        "type": "MOVE",
                        "direction": "FORWARD",
                        "cells": min(max(1, int(action.get("cells", 1))), 2)

                    })

            waypoint = tuple(result["waypoint"])

            # Validate waypoint
            if not (0 <= waypoint[0] < len(self.grid) and
                    0 <= waypoint[1] < len(self.grid[0])):
                raise ValidationError(f"Waypoint {waypoint} out of bounds")

            return NavigationStep(
                waypoint=waypoint,
                actions=actions,
                confidence=self._calculate_confidence(
                    waypoint,
                    actions,
                    len(self.failure_history)
                ))

        except ValidationError as ve:
            logger.warning(f"Route validation failed: {str(ve)}")
            return self._fallback_plan()
        except Exception as e:
            logger.error(f"Route planning failed: {str(e)}")
            return self._fallback_plan()

    def _fallback_plan(self) -> NavigationStep:
        return NavigationStep(
            waypoint=self.robot_pos,
            actions=[{"type": "TURN", "direction": "RIGHT"}],
            confidence=0.1
        )

    def adjust_for_recovery(self):
        """Adjust weights based on failure history"""
        if not self.failure_history:
            return

        last_failure = self.failure_history[-1][1].lower()
        boundary_violations = "boundary" in last_failure or "violation" in last_failure
        obstacle_hits = "obstacle" in last_failure

        if boundary_violations:
            self.saam.weight_matrix[SAAMModule.GUARD.value] *= 1.5
            self.saam.weight_matrix[SAAMModule.VALIDATOR.value] *= 1.3
            logger.info("Increased weights for GUARD and VALIDATOR modules")
        if obstacle_hits:
            self.saam.weight_matrix[SAAMModule.MAPPER.value] *= 1.5
            self.saam.weight_matrix[SAAMModule.PROCESSOR.value] *= 1.3
            logger.info("Increased weights for MAPPER and PROCESSOR modules")

    def _visualize_grid(self) -> str:
        if not self.grid:
            return "Grid not initialized"

        grid = [row.copy() for row in self.grid]
        r, c = self.robot_pos
        if 0 <= r < len(grid) and 0 <= c < len(grid[0]):
            direction_symbols = {
                Direction.UP: "↑",
                Direction.DOWN: "↓",
                Direction.LEFT: "←",
                Direction.RIGHT: "→"
            }
            grid[r][c] = direction_symbols.get(self.robot_facing, "?")
        return "\n".join(" ".join(row) for row in grid)
    def _calculate_confidence(self, waypoint: Tuple[int, int],
                            actions: List[Dict], failures: int = 0) -> float:
        # Manhattan distance
        dist = abs(waypoint[0] - self.robot_pos[0]) + abs(waypoint[1] - self.robot_pos[1])

        # Action complexity penalty
        turns = sum(1 for a in actions if a["type"] == "TURN")
        moves = sum(a.get("cells", 1) for a in actions if a["type"] == "MOVE")
        base_conf = max(0.1, 1.0 - (turns * 0.1 + moves * 0.05 + dist * 0.03))

        # Failure history reduction
        return base_conf * (0.85 ** failures)  # 15% reduction per failure