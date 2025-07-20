#!/usr/bin/env python3
import json
import os
import math
from collections import deque
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

class RobotKernel:
    def __init__(self, target, grid_size, start_position=(0, 0), start_facing="RIGHT",
                 cm_per_cell=30, max_range=300, emergency_threshold=10,
                 model_name="cognitivecomputations/dolphin3.0-mistral-24b:free", temperature=0.0, initial_obstacles=None):
        # Navigation state
        self.robot_position = list(start_position)  # [row, col]
        self.robot_facing = start_facing  # "UP", "DOWN", "LEFT", "RIGHT"
        self.goal_position = list(target)  # [row, col]

        # Environment state
        self.grid_rows, self.grid_cols = grid_size
        self.obstacle_map = [[0] * self.grid_cols for _ in range(self.grid_rows)]

        # Waypoint caching
        self.current_waypoint = None
        self.waypoint_path = deque()
        self.blocked_waypoints = set()

        # Parameters
        self.cm_per_cell = cm_per_cell
        self.max_range = max_range
        self.emergency_threshold = emergency_threshold
        self.max_speed = 20  # cm/s (used for time calculations)

        # Initialize LLM
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")

        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=300,
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=self.openrouter_api_key,
            timeout=30.0
        )
        if initial_obstacles:
            for r, c in initial_obstacles:
                if 0 <= r < self.grid_rows and 0 <= c < self.grid_cols:
                    self.obstacle_map[r][c] = 1

    def update_obstacle_map(self, sensor_values):
        """Update obstacle map based on sensor readings using grid boundaries"""
        sensor_directions = {
            "front": self.robot_facing,
            "left": self._turn_left(self.robot_facing),
            "right": self._turn_right(self.robot_facing)
        }

        for sensor, distance in sensor_values.items():
            if distance < self.max_range:
                dr, dc = self._get_direction_vector(sensor_directions[sensor])
                r, c = self.robot_position
                step = 1
                tolerance = 1e-5  # For floating-point precision

                while True:
                    # Calculate candidate obstacle position
                    r_obs = r + dr * step
                    c_obs = c + dc * step

                    # Stop if out of grid bounds
                    if not (0 <= r_obs < self.grid_rows and 0 <= c_obs < self.grid_cols):
                        break

                    # Calculate distance boundaries for this cell
                    lower_bound = (step - 1) * self.cm_per_cell
                    upper_bound = step * self.cm_per_cell

                    # Check if obstacle falls in this cell's physical boundaries
                    if lower_bound - tolerance <= distance <= upper_bound + tolerance:
                        self.obstacle_map[r_obs][c_obs] = 1
                        # Invalidate cached waypoint if blocked
                        if self.current_waypoint == [r_obs, c_obs]:
                            self.blocked_waypoints.add(tuple(self.current_waypoint))
                            self.current_waypoint = None
                        break

                    step += 1

    def run_step(self, sensor_values):
        """Execute one control cycle"""
        # 1. Emergency stop check
        if any(dist < self.emergency_threshold for dist in sensor_values.values()):
            return {"type": "STOP", "message": "Emergency stop triggered"}  # Return action dict

        # 2. Update environment knowledge
        self.update_obstacle_map(sensor_values)

        # 3. Check if goal is reached
        if self.robot_position == self.goal_position:
            return None

        # 4. Waypoint caching logic
        if self._should_replan():
            self._generate_new_plan()

        # 5. Execute cached path if available
        if self.waypoint_path:
            return self._execute_next_action()
        # 6. Default to stopping
        return None

    def _should_replan(self):
        """Determine if we need to generate a new plan"""
        # No current plan
        if not self.current_waypoint and not self.waypoint_path:
            return True

        # Reached current waypoint
        if self.current_waypoint and self.robot_position == self.current_waypoint:
            return True

        # Path blocked
        if self.waypoint_path:
            next_action = self.waypoint_path[0]
            if next_action["type"] == "MOVE":
                next_pos = self._get_next_position()
                if not self._is_valid_position(next_pos):
                    self.blocked_waypoints.add(tuple(next_pos))
                    return True

        return False

    def _is_path_clear(self, start, end):
        """Check if straight path between two points is obstacle-free"""
        # Same row - check horizontal path
        if start[0] == end[0]:
            col_start = min(start[1], end[1])
            col_end = max(start[1], end[1])
            for c in range(col_start, col_end + 1):
                if self.obstacle_map[start[0]][c] == 1:
                    return False
            return True

        # Same column - check vertical path
        if start[1] == end[1]:
            row_start = min(start[0], end[0])
            row_end = max(start[0], end[0])
            for r in range(row_start, row_end + 1):
                if self.obstacle_map[r][start[1]] == 1:
                    return False
            return True

        return False

    def _generate_new_plan(self):
        """Generate new navigation plan with actions using LLM"""
        self.waypoint_path.clear()
        llm_input = {
            "robot_position": self.robot_position,
            "robot_facing": self.robot_facing,
            "goal_position": self.goal_position,
            "obstacle_map": self.obstacle_map,
            "blocked_waypoints": list(self.blocked_waypoints)
        }
        prompt = self._build_llm_prompt(llm_input)

        try:
            # Capture and validate LLM response
            llm_response = self._call_llm(prompt)
            print(f"LLM Response: {llm_response}")

            # Extract JSON from markdown code blocks
            if "```json" in llm_response:
                llm_response = llm_response.split("```json")[1].split("```")[0].strip()
            elif "```" in llm_response:
                llm_response = llm_response.split("```")[1].split("```")[0].strip()

            action_plan = json.loads(llm_response)

            # Handle direct path case
            if action_plan["next_waypoint"] is None:
                if self._is_direct_path_clear():
                    self.current_waypoint = list(self.goal_position)
                    # Validate actions for direct path
                    if "actions" in action_plan and self._validate_actions(action_plan["actions"], self.goal_position):
                        self.waypoint_path = deque(action_plan["actions"])
                        print(f"Validated direct path actions: {self.waypoint_path}")
                    else:
                        print("Direct path actions are invalid. Blocking goal.")
                        self.blocked_waypoints.add(tuple(self.goal_position))
                        self.current_waypoint = None
                else:
                    print("Direct path blocked but LLM suggested direct path")
                return

            # Process waypoint and actions
            wp = action_plan["next_waypoint"]
            self.current_waypoint = [int(wp[0]), int(wp[1])]
            print(f"Selected waypoint: {self.current_waypoint}")

            # Validate waypoint
            if not self._is_valid_waypoint(self.current_waypoint):
                print(f"Invalid waypoint detected: {self.current_waypoint}")
                self.blocked_waypoints.add(tuple(self.current_waypoint))
                self.current_waypoint = None
                return

            # Process and validate actions
            if "actions" in action_plan:
                if self._validate_actions(action_plan["actions"], self.current_waypoint):
                    self.waypoint_path = deque(action_plan["actions"])
                    print(f"Validated actions: {self.waypoint_path}")
                else:
                    print("Action validation failed. Blocking waypoint.")
                    self.blocked_waypoints.add(tuple(self.current_waypoint))
                    self.current_waypoint = None
            else:
                print("No actions provided. Blocking waypoint.")
                self.blocked_waypoints.add(tuple(self.current_waypoint))
                self.current_waypoint = None

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Planning error: {e}")
            self.blocked_waypoints.add(tuple(self.robot_position))
            self.current_waypoint = None

    def _validate_actions(self, actions, waypoint):
        """Validate LLM-generated actions through simulation with detailed logging"""
        if not isinstance(actions, list):
            print("Actions not a list")
            return False

        temp_pos = list(self.robot_position)
        temp_facing = self.robot_facing
        print(f"Starting validation at {temp_pos} facing {temp_facing}")

        for i, action in enumerate(actions):
            # Validate action structure
            if not isinstance(action, dict):
                print(f"Action {i} is not a dictionary")
                return False

            if "type" not in action:
                print(f"Action {i} missing 'type' key")
                return False

            if "direction" not in action:
                print(f"Action {i} missing 'direction' key")
                return False

            action_type = action["type"]
            direction = action["direction"]
            print(f"Processing action {i}: {action_type} {direction}")

            # Process turn action
            if action_type == "TURN":
                if direction not in ["LEFT", "RIGHT"]:
                    print(f"Invalid turn direction: {direction}")
                    return False

                if direction == "LEFT":
                    temp_facing = self._turn_left(temp_facing)
                else:  # RIGHT
                    temp_facing = self._turn_right(temp_facing)

                print(f"Turned to face {temp_facing}")

            # Process move action
            elif action_type == "MOVE":
                if direction != "FORWARD":
                    print(f"Invalid move direction: {direction}")
                    return False

                # Ensure speed and time exist and are positive
                if "speed" not in action:
                    print("Move action missing 'speed'")
                    return False

                if "time" not in action:
                    print("Move action missing 'time'")
                    return False

                try:
                    speed = float(action["speed"])
                    time_val = float(action["time"])
                except (ValueError, TypeError):
                    print("Invalid speed or time value")
                    return False

                if speed <= 0:
                    print(f"Invalid speed: {speed} <= 0")
                    return False

                if time_val <= 0:
                    print(f"Invalid time: {time_val} <= 0")
                    return False

                if speed > self.max_speed:
                    print(f"Speed {speed} exceeds max {self.max_speed}")
                    return False

                # Compute how many cells this move covers
                distance_cm = speed * time_val
                steps = round(distance_cm / self.cm_per_cell)
                print(f"Moving {steps} cells ({distance_cm}cm) at {speed} cm/s for {time_val}s")

                # Now simulate steps
                dr, dc = self._get_direction_vector(temp_facing)
                for step in range(steps):
                    next_pos = [temp_pos[0] + dr, temp_pos[1] + dc]
                    print(f"Step {step+1}/{steps}: moving to {next_pos}")

                    if not self._is_valid_position(next_pos):
                        print(f"Collision detected at {next_pos}")
                        return False

                    temp_pos = next_pos
                print(f"Finished move at {temp_pos}")

            else:
                print(f"Invalid action type: {action_type}")
                return False

        # Final position check
        arrived = temp_pos == waypoint
        if not arrived:
            print(f"Action sequence ended at {temp_pos} but waypoint is {waypoint}")
        else:
            print(f"Successfully reached waypoint {waypoint}")

        return arrived

    def _is_valid_waypoint(self, waypoint):
        """Strict waypoint validation with enhanced checks and detailed logging"""
        print(f"Validating waypoint: {waypoint}")

        # Type and format validation
        if not isinstance(waypoint, list) or len(waypoint) != 2:
            print("Waypoint format invalid: must be [row, col]")
            return False

        try:
            r, c = int(waypoint[0]), int(waypoint[1])
        except (TypeError, ValueError):
            print("Waypoint coordinates not integers")
            return False

        # Boundary checks
        if not (0 <= r < self.grid_rows):
            print(f"Row {r} out of bounds [0, {self.grid_rows-1}]")
            return False

        if not (0 <= c < self.grid_cols):
            print(f"Column {c} out of bounds [0, {self.grid_cols-1}]")
            return False

        # Obstacle check
        if self.obstacle_map[r][c] == 1:
            print("Waypoint is an obstacle")
            return False

        # Must be in same row OR same column
        same_row = r == self.robot_position[0]
        same_col = c == self.robot_position[1]

        if not (same_row or same_col):
            print(f"Waypoint not in same row ({self.robot_position[0]}) or column ({self.robot_position[1]})")
            return False

        # Must not be the final goal
        if [r, c] == self.goal_position:
            print("Waypoint cannot be the final goal")
            return False

        # Must be closer to goal than current position
        current_dist = abs(self.robot_position[0] - self.goal_position[0]) + \
                      abs(self.robot_position[1] - self.goal_position[1])
        waypoint_dist = abs(r - self.goal_position[0]) + \
                        abs(c - self.goal_position[1])

        if waypoint_dist >= current_dist:
            print(f"Waypoint not closer to goal: {waypoint_dist} >= {current_dist}")
            return False

        # Must not be blocked
        if tuple(waypoint) in self.blocked_waypoints:
            print("Waypoint is blocked")
            return False

        # Path must be clear
        if not self._is_path_clear(self.robot_position, [r, c]):
            print("Path to waypoint is not clear")
            return False

        print("Waypoint is valid")
        return True

    def _is_direct_path_clear(self):
        """Check if straight-line path to goal is possible"""
        return self._is_path_clear(self.robot_position, self.goal_position)

    def _execute_next_action(self):
        """Execute next action in the path and return action dictionary"""
        if not self.waypoint_path:
            return None

        action = self.waypoint_path.popleft()
        action_dict = action.copy()  # Start with the original action

        if action["type"] == "TURN":
            if action["direction"] == "LEFT":
                self.robot_facing = self._turn_left(self.robot_facing)
                action_dict["left_speed"] = -120
                action_dict["right_speed"] = 120
                print(f"Turning LEFT to face {self.robot_facing}")
            else:  # RIGHT
                self.robot_facing = self._turn_right(self.robot_facing)
                action_dict["left_speed"] = 120
                action_dict["right_speed"] = -120
                print(f"Turning RIGHT to face {self.robot_facing}")
            return action_dict

        elif action["type"] == "MOVE":
            if "speed" not in action or "time" not in action:
                self.waypoint_path.clear()
                return None

            try:
                speed = float(action["speed"])
                time_val = float(action["time"])
            except (ValueError, TypeError):
                self.waypoint_path.clear()
                return None

            if speed <= 0 or time_val <= 0:
                self.waypoint_path.clear()
                return None

            # Compute movement parameters
            distance_cm = speed * time_val
            steps = round(distance_cm / self.cm_per_cell)
            dr, dc = self._get_direction_vector(self.robot_facing)

            # Update position
            new_pos = [
                self.robot_position[0] + dr * steps,
                self.robot_position[1] + dc * steps
            ]
            print(f"Moving {steps} cells from {self.robot_position} to {new_pos}")
            self.robot_position = new_pos

            # Add motor details to action dict
            pwm = int((speed / self.max_speed) * 120)
            action_dict["left_speed"] = pwm
            action_dict["right_speed"] = pwm
            return action_dict

        return None

    def _get_next_position(self):
        """Calculate next position based on facing direction"""
        dr, dc = self._get_direction_vector(self.robot_facing)
        return [
            self.robot_position[0] + dr,
            self.robot_position[1] + dc
        ]

    def _is_valid_position(self, position):
        try:
            # Handle multiple position formats:
            if isinstance(position, (list, tuple)) and len(position) >= 2:
                r = int(position[0])
                c = int(position[1])
            elif isinstance(position, dict) and 'x' in position and 'y' in position:
                r = int(position['x'])
                c = int(position['y'])
            else:
                return False  # Invalid position format
        except (TypeError, ValueError, IndexError, KeyError):
            return False

        # Check grid boundaries
        if not (0 <= r < self.grid_rows and 0 <= c < self.grid_cols):
            return False

        # Check if cell is not an obstacle (using obstacle_map)
        if self.obstacle_map[r][c] == 1:
            return False

        return True

    def _build_llm_prompt(self, data):
        """Construct LLM prompt with explicit constraints and examples"""
        grid_visual = self._visualize_grid()

        # Generate valid candidate waypoints
        candidate_waypoints = []
        for r in range(self.grid_rows):
            # Same row candidates
            if r == self.robot_position[0]:
                for c in range(self.grid_cols):
                    wp = [r, c]
                    if (wp != self.robot_position and
                        wp != self.goal_position and
                        self._is_valid_waypoint(wp)):
                        candidate_waypoints.append(wp)
            # Same column candidates
            else:
                c = self.robot_position[1]
                wp = [r, c]
                if (wp != self.robot_position and
                    wp != self.goal_position and
                    self._is_valid_waypoint(wp)):
                    candidate_waypoints.append(wp)

        # Format candidate waypoints
        safe_choices = "\n**VALID WAYPOINTS**\n" + "\n".join(
            f"- [{wp[0]},{wp[1]}]" for wp in candidate_waypoints
        ) if candidate_waypoints else "\n**WARNING: No valid waypoints!**"

        # Directional instructions
        facing = self.robot_facing
        turn_to_down = "RIGHT" if facing in ["RIGHT", "UP"] else "LEFT"
        turn_to_up = "LEFT" if facing in ["RIGHT", "DOWN"] else "RIGHT"

        # Pre-calculate times to avoid nested f-strings
        time1 = round(2 * self.cm_per_cell / 15, 1)
        time2 = round(2 * self.cm_per_cell / 15, 1)
        time3 = round(2 * self.cm_per_cell / 15, 1)
        time4 = round(6 * self.cm_per_cell / 15, 1)

        # Use string formatting for examples
        action_examples = (
            "**ACTION GENERATION EXAMPLES - OUTPUT EXACTLY THESE FORMATS**\n\n"
            "Case 1: Move forward in current direction (RIGHT)\n"
            "{{\n"
            '  "next_waypoint": [0, 2],\n'
            '  "actions": [\n'
            '    {{"type": "MOVE", "direction": "FORWARD", "speed": 15, "time": {}}}\n'
            "  ]\n"
            "}}\n\n"
            "Case 2: Turn and move down (from RIGHT to DOWN)\n"
            "{{\n"
            '  "next_waypoint": [2, 0],\n'
            '  "actions": [\n'
            '    {{"type": "TURN", "direction": "{}"}},\n'
            '    {{"type": "MOVE", "direction": "FORWARD", "speed": 15, "time": {}}}\n'
            "  ]\n"
            "}}\n\n"
            "Case 3: Turn and move up (from RIGHT to UP)\n"
            "{{\n"
            '  "next_waypoint": [0, 0],\n'
            '  "actions": [\n'
            '    {{"type": "TURN", "direction": "{}"}},\n'
            '    {{"type": "MOVE", "direction": "FORWARD", "speed": 15, "time": {}}}\n'
            "  ]\n"
            "}}\n\n"
            "Case 4: Direct path to goal\n"
            "{{\n"
            '  "next_waypoint": null,\n'
            '  "actions": [\n'
            '    {{"type": "MOVE", "direction": "FORWARD", "speed": 15, "time": {}}}\n'
            "  ]\n"
            "}}\n"
        ).format(time1, turn_to_down, time2, turn_to_up, time3, time4)

        # Add critical rules section
        critical_rules = (
            "**CRITICAL RULES - READ CAREFULLY**\n"
            "1. Waypoint MUST be in the SAME ROW or SAME COLUMN as the robot\n"
            "2. Diagonal waypoints like [1,1] are INVALID and will be rejected\n"
            "3. Waypoint MUST be closer to the goal than current position\n"
            "4. Actions MUST lead EXACTLY to the waypoint position\n"
            "5. For MOVE actions: time = (distance in cm) / speed\n"
            "6. Distance in cm = (number of cells) * {} cm\n"
            "7. Speed must be between 1-20 cm/s\n"
            "8. Consider robot's current facing direction when planning turns\n"
        ).format(self.cm_per_cell)

        # Build prompt
        prompt = (
            "ROBOT NAVIGATION CONTROLLER - STRICT RULES\n\n"
            + critical_rules + "\n\n"
            "**ABSOLUTE CONSTRAINTS (VIOLATIONS WILL FAIL)**\n"
            "1. Output ONLY the JSON object with NO additional text or explanations\n"
            "2. next_waypoint MUST:\n"
            "   - Be in SAME ROW or SAME COLUMN as robot\n"
            "   - NOT be the final goal (" + str(self.goal_position) + ")\n"
            "   - Be one of the valid waypoints listed below\n"
            "3. Actions MUST:\n"
            "   - Only contain TURN and MOVE actions\n"
            "   - For MOVEs: speed ≤ " + str(self.max_speed) + " cm/s\n"
            "   - Calculate time as: time = (cells × " + str(self.cm_per_cell) + ") / speed\n"
            "4. If direct path to goal exists → output null for waypoint\n\n"
            "**CURRENT STATE**\n"
            "- Robot: " + str(data['robot_position']) + " facing " + data['robot_facing'] + "\n"
            "- Goal: " + str(data['goal_position']) + "\n"
            "- Blocked waypoints: " + str(list(data['blocked_waypoints'])) + "\n\n"
            + safe_choices + "\n\n"
            + action_examples + "\n\n"
            "**GRID MAP** (■=Obstacle, G=Goal, X=Blocked, R=Robot)\n"
            + grid_visual + "\n\n"
            "**YOUR TASK**\n"
            "1. Select ONE valid waypoint from the list above\n"
            "2. Generate EXACTLY the actions needed to reach it\n"
            "3. Output ONLY the JSON object with NO additional text\n"
            "4. Use EXACTLY the format shown in examples"
        )

        return prompt

    def _call_llm(self, prompt):
        """Call LLM with strict output instructions"""
        messages = [
            SystemMessage(content=(
                "You are a robot navigation controller. "
                "Output ONLY valid JSON with NO additional text, comments, or explanations. "
                "Follow ALL navigation rules exactly. "
                "Your response MUST be parsable by json.loads() directly. "
                "Pay special attention to the robot's current facing direction."
            )),
            HumanMessage(content=prompt)
        ]
        try:
            response = self.llm.invoke(messages)
            content = response.content.strip()

            # Extract JSON from markdown code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            return content
        except Exception as e:
            print(f"LLM Error: {e}")
            return '{"next_waypoint": null, "actions": []}'

    def _visualize_grid(self):
        """Create visual grid representation for LLM"""
        grid_str = ""
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                if [r, c] == self.robot_position:
                    # Show arrow for direction
                    arrows = {"UP": "↑", "DOWN": "↓", "LEFT": "←", "RIGHT": "→"}
                    grid_str += arrows.get(self.robot_facing, 'R') + ' '
                elif [r, c] == self.goal_position:
                    grid_str += 'G '
                elif (r, c) in self.blocked_waypoints:
                    grid_str += 'X '
                elif self.obstacle_map[r][c] == 1:
                    grid_str += '■ '
                else:
                    grid_str += '· '
            grid_str += '\n'
        return grid_str

    @staticmethod
    def _turn_left(facing):
        """Calculate new facing after left turn"""
        turns = {"UP": "LEFT", "LEFT": "DOWN", "DOWN": "RIGHT", "RIGHT": "UP"}
        return turns[facing]

    @staticmethod
    def _turn_right(facing):
        """Calculate new facing after right turn"""
        turns = {"UP": "RIGHT", "RIGHT": "DOWN", "DOWN": "LEFT", "LEFT": "UP"}
        return turns[facing]

    @staticmethod
    def _get_direction_vector(facing):
        """Get (dr, dc) movement vector for direction"""
        return {
            "UP": (-1, 0),
            "DOWN": (1, 0),
            "LEFT": (0, -1),
            "RIGHT": (0, 1)
        }[facing]