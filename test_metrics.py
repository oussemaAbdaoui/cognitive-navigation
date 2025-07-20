#!/usr/bin/env python3
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
import ast

class PerceptionAccuracyMetric(BaseMetric):
    def __init__(self, expected_grid: str):
        self.threshold = 0.8
        self.expected_grid = expected_grid

    def measure(self, test_case: LLMTestCase):
        actual = test_case.actual_output
        expected = self.expected_grid

        actual_lines = actual.split('\n')
        expected_lines = expected.split('\n')
        matches = 0
        total = 0

        for i in range(min(len(actual_lines), len(expected_lines))):
            actual_cells = actual_lines[i].split()
            expected_cells = expected_lines[i].split()
            for j in range(min(len(actual_cells), len(expected_cells))):
                if actual_cells[j] == expected_cells[j]:
                    matches += 1
                total += 1

        self.score = matches / total if total > 0 else 0
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "PerceptionAccuracy"

class WaypointOptimalityMetric(BaseMetric):
    def __init__(self, robot_pos, goal_pos, grid_size, obstacle_grid):
        self.threshold = 0.7
        self.robot_pos = robot_pos
        self.goal_pos = goal_pos
        self.grid_size = grid_size
        self.obstacle_grid = obstacle_grid

    def measure(self, test_case: LLMTestCase):
        waypoint_str = test_case.actual_output

        # Handle None case
        if waypoint_str == "None":
            self.score = 0
            self.success = False
            return 0

        try:
            # Try to parse tuple from string
            if waypoint_str.startswith('(') and waypoint_str.endswith(')'):
                waypoint = tuple(map(int, waypoint_str[1:-1].split(',')))
            elif ',' in waypoint_str:
                waypoint = tuple(map(int, waypoint_str.split(',')))
            elif waypoint_str.isdigit():  # Single coordinate
                waypoint = (int(waypoint_str), 0)
            else:
                self.score = 0
                self.success = False
                return 0
        except (ValueError, AttributeError):
            self.score = 0
            self.success = False
            return 0

        # Validate waypoint format
        if not isinstance(waypoint, (tuple, list)) or len(waypoint) != 2:
            self.score = 0
            self.success = False
            return 0

        # Check boundaries
        r, c = waypoint
        if not (0 <= r < self.grid_size[0]) or not (0 <= c < self.grid_size[1]):
            self.score = 0
            self.success = False
            return 0

        # Check if obstacle
        if self.obstacle_grid[r][c] == '■':
            self.score = 0
            self.success = False
            return 0

        # Calculate Manhattan distance improvement
        robot_dist = abs(self.robot_pos[0]-self.goal_pos[0]) + abs(self.robot_pos[1]-self.goal_pos[1])
        wp_dist = abs(waypoint[0]-self.goal_pos[0]) + abs(waypoint[1]-self.goal_pos[1])

        if robot_dist == 0:
            self.score = 1.0
        else:
            self.score = max(0, min(1, 1 - (wp_dist / robot_dist)))

        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "WaypointOptimality"

class ActionCorrectnessMetric(BaseMetric):
    def __init__(self, start_pos, start_facing, waypoint, grid_size, obstacle_grid):
        self.threshold = 0.9
        self.start_pos = start_pos
        self.start_facing = start_facing
        self.waypoint = waypoint
        self.grid_size = grid_size
        self.obstacle_grid = obstacle_grid

    def measure(self, test_case: LLMTestCase):
        action_str = test_case.actual_output

        # Handle empty/None case
        if not action_str or action_str == "None" or action_str.strip() == "":
            self.score = 0.0
            self.success = False
            return 0.0

        try:
            actions = ast.literal_eval(action_str)
        except (SyntaxError, ValueError):
            self.score = 0.0
            self.success = False
            return 0.0

        # Validate actions structure
        if not isinstance(actions, list):
            self.score = 0.0
            self.success = False
            return 0.0

        # Simulate actions with obstacle/boundary checks
        x, y = self.start_pos
        facing = self.start_facing
        rows, cols = self.grid_size
        valid_actions = []

        for action in actions:
            if not isinstance(action, dict):
                continue

            # Normalize action format
            action_type = action.get('type', '').upper()
            direction = action.get('direction', '').upper()
            cells = action.get('cells', 0)

            # Validate action
            if action_type == "TURN":
                if direction in ["UP", "DOWN", "LEFT", "RIGHT"]:
                    facing = direction
                    valid_actions.append(action)
            elif action_type == "MOVE" and direction == "FORWARD":
                if isinstance(cells, int) and cells > 0:
                    # Process movement step-by-step
                    dx, dy = 0, 0
                    if facing == "UP": dx = -1
                    elif facing == "DOWN": dx = 1
                    elif facing == "LEFT": dy = -1
                    elif facing == "RIGHT": dy = 1

                    valid_move = True
                    temp_x, temp_y = x, y

                    for _ in range(cells):
                        temp_x += dx
                        temp_y += dy

                        # Check boundaries
                        if not (0 <= temp_x < rows) or not (0 <= temp_y < cols):
                            valid_move = False
                            break

                        # Check obstacles
                        if self.obstacle_grid[temp_x][temp_y] == '■':
                            valid_move = False
                            break

                    if valid_move:
                        x, y = temp_x, temp_y
                        valid_actions.append(action)

        # Check final position
        self.score = 1.0 if (x, y) == tuple(self.waypoint) else 0.0
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "ActionCorrectness"