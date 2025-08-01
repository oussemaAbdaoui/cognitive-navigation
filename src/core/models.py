# models.py - Data models and Pydantic schemas

from typing import Dict, List, Tuple, Optional, Literal
from pydantic import BaseModel, conint
from dataclasses import dataclass, field

class SensorReading(BaseModel):
    """Sensor distance readings"""
    front: float  # cm
    left: float   # cm
    right: float  # cm

class Action(BaseModel):
    """Robot action commands"""
    type: Literal["TURN", "MOVE"]
    direction: Optional[Literal["LEFT", "RIGHT", "FORWARD"]] = None
    cells: Optional[conint(gt=0, le=5)] = None

@dataclass
class GridCell:
    """Grid cell state"""
    type: str = "UNKNOWN"  # UNKNOWN, CLEAR, OBSTACLE, ROBOT, GOAL
    confidence: float = 0.0
    last_updated: int = 0

@dataclass
class RobotState:
    """Robot state with trajectory memory"""
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