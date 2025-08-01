from typing import List, Tuple
from ..core.models import SensorReading

class NavigationEnvironment:
    """Simulated environment for navigation testing"""

    def __init__(self, obstacles: List[Tuple[int, int]], bounds: Tuple[int, int, int, int]):
        self.obstacles = set(obstacles)
        self.bounds = bounds  # min_row, max_row, min_col, max_col
        self.cell_size_cm = 30

    def get_sensor_readings(self, position: Tuple[int, int], facing: str) -> SensorReading:
        """Simulate sensor readings from current position"""

        # Direction vectors for sensor orientations
        direction_vectors = {
            "UP": {"front": (-1, 0), "left": (0, -1), "right": (0, 1)},
            "DOWN": {"front": (1, 0), "left": (0, 1), "right": (0, -1)},
            "LEFT": {"front": (0, -1), "left": (1, 0), "right": (-1, 0)},
            "RIGHT": {"front": (0, 1), "left": (-1, 0), "right": (1, 0)}
        }

        vectors = direction_vectors[facing]
        max_range_cells = 10  # ~300cm range

        readings = {}
        for sensor_dir, (dr, dc) in vectors.items():
            distance_cm = self._cast_ray(position, dr, dc, max_range_cells)
            readings[sensor_dir] = distance_cm

        return SensorReading(
            front=readings["front"],
            left=readings["left"],
            right=readings["right"]
        )

# In NavigationEnvironment._cast_ray()
    def _cast_ray(self, start_pos: Tuple[int, int], dr: int, dc: int, max_cells: int) -> float:
        """Cast a ray and return distance to first obstacle"""
        r, c = start_pos

        # Check current position first (important for obstacle right in front)
        if (r, c) in self.obstacles:
            return 0.0  # Already in obstacle

        for i in range(1, max_cells + 1):
            check_pos = (r + i * dr, c + i * dc)

            if check_pos in self.obstacles:
                return i * self.cell_size_cm

        return max_cells * self.cell_size_cm

    def is_valid_position(self, position: Tuple[int, int]) -> bool:
        """Check if position is valid (within bounds and not obstacle)"""
        r, c = position
        return (self.bounds[0] <= r <= self.bounds[1] and
                self.bounds[2] <= c <= self.bounds[3] and
                position not in self.obstacles)