import json
import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


class StateManager:
    def __init__(self, target, grid_size, start_position=(0, 0), start_facing="RIGHT",
                 cm_per_cell=30, max_range=300, initial_obstacles=None):
        self.robot_position = list(start_position)
        self.robot_facing = start_facing
        self.goal_position = list(target)

        self.grid_rows, self.grid_cols = grid_size
        self.cm_per_cell = cm_per_cell
        self.max_range = max_range

        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")

        self.llm = ChatOpenAI(
            model="mistralai/mistral-7b-instruct:free",
            temperature=0.0,
            max_tokens=500,
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=self.openrouter_api_key,
            timeout=30.0
        )

        self.obstacle_grid = self._initialize_grid(initial_obstacles)

    def _initialize_grid(self, obstacles):
        grid = [['·'] * self.grid_cols for _ in range(self.grid_rows)]
        grid[self.robot_position[0]][self.robot_position[1]] = self._get_direction_arrow()
        grid[self.goal_position[0]][self.goal_position[1]] = 'G'

        if obstacles:
            for r, c in obstacles:
                if 0 <= r < self.grid_rows and 0 <= c < self.grid_cols:
                    grid[r][c] = '■'
        return grid

    def process_sensor_data(self, sensor_values):
        sensor_prompt = self._build_sensor_prompt(sensor_values)
        llm_response = self._call_llm(sensor_prompt)
        self._parse_state_update(llm_response)
        return self.get_visual_grid()

    def _build_sensor_prompt(self, sensor_values):
        return (
            "ROBOT STATE MANAGER - PROCESS SENSOR DATA\n\n"
            "**TASK**: Update robot state and obstacle grid based on sensor readings\n"
            "**SENSOR VALUES**:\n"
            f"- Front: {sensor_values.get('front', 0)} cm\n"
            f"- Left: {sensor_values.get('left', 0)} cm\n"
            f"- Right: {sensor_values.get('right', 0)} cm\n\n"
            "**CURRENT STATE**\n"
            f"- Position: {self.robot_position}\n"
            f"- Facing: {self.robot_facing}\n"
            f"- Goal: {self.goal_position}\n"
            "**CURRENT GRID**\n"
            f"{self.get_visual_grid()}\n\n"
            "**INSTRUCTIONS**\n"
            "1. Calculate obstacle positions based on sensor readings\n"
            "2. Update grid with new obstacles (mark as '■')\n"
            "3. Output updated grid and robot state in JSON format\n"
            "4. Maintain grid dimensions\n\n"
            "**OUTPUT FORMAT**:\n"
            "{\n"
            '  "robot_position": [row, col],\n'
            '  "robot_facing": "DIRECTION",\n'
            '  "obstacle_grid": [\n'
            '    ["·", "■", ...],\n'
            '    ...\n'
            '  ]\n'
            "}\n\n"
            "**DIRECTIONS**:\n"
            "- Output ONLY valid JSON with NO additional text\n"
            "- Use '·' for empty, '■' for obstacles, arrows for robot, 'G' for goal\n"
            f"- Maintain grid size exactly: {self.grid_rows}x{self.grid_cols}"
        )

    def _call_llm(self, prompt):
        messages = [
            SystemMessage(content="Output ONLY valid JSON with NO additional text"),
            HumanMessage(content=prompt)
        ]
        response = self.llm.invoke(messages).content.strip()

        print("\n📤 [StateManager LLM Response]:\n", response)
        if "```json" in response:
            return response.split("```json")[1].split("```")[0].strip()
        return response

    def _parse_state_update(self, llm_response):
        try:
            state_data = json.loads(llm_response)
        except json.JSONDecodeError:
            print("❌ JSON decode failed.")
            return

        try:
            self.robot_position = state_data["robot_position"]
            self.robot_facing = state_data["robot_facing"]
            self.obstacle_grid = state_data["obstacle_grid"]
            r, c = self.robot_position
            if 0 <= r < self.grid_rows and 0 <= c < self.grid_cols:
                if self.obstacle_grid[r][c] != '■':
                    self.obstacle_grid[r][c] = self._get_direction_arrow()
        except KeyError as e:
            print(f"❌ State update key error: {e}")

    def get_visual_grid(self):
        return "\n".join(" ".join(row) for row in self.obstacle_grid)

    def _get_direction_arrow(self):
        return {"UP": "↑", "DOWN": "↓", "LEFT": "←", "RIGHT": "→"}[self.robot_facing]


class WaypointPlanner:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="cognitivecomputations/dolphin3.0-mistral-24b:free",
            temperature=0.0,
            max_tokens=200,
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            timeout=30.0
        )

    def plan_waypoint(self, visual_grid, robot_pos, goal_pos):
        prompt = self._build_waypoint_prompt(visual_grid, robot_pos, goal_pos)
        llm_response = self._call_llm(prompt)
        return self._parse_waypoint(llm_response)

    def _build_waypoint_prompt(self, grid, robot_pos, goal_pos):
        return (
            "OUTPUT ONLY a single JSON object with this format:\n"
            '{"waypoint": [row, col]}\n'
            "No explanations, no markdown, no extra text.\n"
            "\n"
            f"Grid:\n{grid}\n"
            f"Robot position: {robot_pos}\n"
            f"Goal position: {goal_pos}\n"
            "\n"
            "TASK: Choose the best waypoint in the same row or column as the robot,\n"
            "which has no obstacles between robot and waypoint, is not the goal,\n"
            "and is closest to the goal by Manhattan distance.\n"
            "If ties, choose the waypoint farthest from the robot.\n"
        )


    def _call_llm(self, prompt):
        result = self.llm.invoke([HumanMessage(content=prompt)]).content.strip()
        print("\n📤 [WaypointPlanner LLM Response]:\n", result)
        return result

    def _parse_waypoint(self, response):
        try:
            data = json.loads(response)
            return data["waypoint"]
        except (json.JSONDecodeError, KeyError):
            return None


class ActionPlanner:
    def __init__(self, cm_per_cell=30):
        self.cm_per_cell = cm_per_cell
        self.llm = ChatOpenAI(
            model="cognitivecomputations/dolphin3.0-mistral-24b:free",
            temperature=0.0,
            max_tokens=300,
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            timeout=30.0
        )

    def plan_actions(self, waypoint, robot_pos, robot_facing):
        prompt = self._build_action_prompt(waypoint, robot_pos, robot_facing)
        print("\n🧠 [ActionPlanner Prompt]:\n", prompt)
        llm_response = self._call_llm(prompt)
        return self._parse_actions(llm_response)

    def _build_action_prompt(self, waypoint, robot_pos, facing):
        return (
            "ROBOT ACTION PLANNER\n\n"
            "**TASK**: Generate a sequence of movement actions to reach the given waypoint.\n\n"
            "**RULES**:\n"
            "1. Use ONLY two action types:\n"
            '   - {"type": "TURN", "direction": "LEFT|RIGHT"}\n'
            '   - {"type": "MOVE", "direction": "FORWARD", "cells": N}\n'
            "2. Turns must be 90° (LEFT or RIGHT), relative to current FACING.\n"
            "3. Turning changes robot's FACING direction immediately:\n"
            "   - LEFT turn: Rotates 90° counter-clockwise\n"
            "   - RIGHT turn: Rotates 90° clockwise\n"
            "4. Move FORWARD only, no backward or sideways moves.\n"
            f"5. Robot starts at {robot_pos} facing {facing}.\n"
            f"6. Waypoint is at {waypoint}.\n"
            "7. Cell size: 30cm.\n"
            "8. Grid coordinate system:\n"
            "   - (0,0) is TOP-LEFT corner\n"
            "   - Increasing row index = moving DOWN\n"
            "   - Increasing column index = moving RIGHT\n"
            "9. Movement effects:\n"
            "   - UP: row decreases\n"
            "   - DOWN: row increases\n"
            "   - LEFT: column decreases\n"
            "   - RIGHT: column increases\n\n"
            "**IMPORTANT OUTPUT INSTRUCTIONS**:\n"
            "- OUTPUT A VALID JSON ARRAY ONLY.\n"
            "- DO NOT return Python code or explanations.\n"
            "- DO NOT include any markdown or comments.\n"
            "- Just return the array like this:\n"
            '[{"type": "TURN", "direction": "RIGHT"}, {"type": "MOVE", "direction": "FORWARD", "cells": 2}]\n'
        )
    def apply_actions(self, actions, start_pos, start_facing):
        facing_order = ["UP", "RIGHT", "DOWN", "LEFT"]  # clockwise order
        x, y = start_pos
        facing = start_facing

        for action in actions:
            if action["type"] == "TURN":
                current_idx = facing_order.index(facing)
                if action["direction"] == "RIGHT":
                    facing = facing_order[(current_idx + 1) % 4]
                else:  # LEFT turn
                    facing = facing_order[(current_idx - 1) % 4]

            elif action["type"] == "MOVE":
                cells = action["cells"]
                if facing == "UP":
                    x -= cells
                elif facing == "DOWN":
                    x += cells
                elif facing == "LEFT":
                    y -= cells
                elif facing == "RIGHT":
                    y += cells

        return [x, y], facing


    def _call_llm(self, prompt):
        messages = [
        SystemMessage(content="Output ONLY valid JSON array, no markdown, no explanation."),
        HumanMessage(content=prompt)
    ]
        result = self.llm.invoke(messages).content.strip()
        print("\n📤 [LLM ActionPlanner Raw Output]:\n", result)
        return result

    def _parse_actions(self, response):
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return []