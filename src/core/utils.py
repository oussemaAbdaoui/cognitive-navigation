# src/core/utils.py
import math
from typing import Tuple, List, Dict, Optional
from pathlib import Path
import json
from dataclasses import asdict
import numpy as np

# ------------------- Coordinate Utilities -------------------

def calculate_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """Calculate Manhattan distance between two positions"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def calculate_euclidean_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """Calculate Euclidean distance between two points"""
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def get_relative_direction(current_pos: Tuple[int, int],
                         target_pos: Tuple[int, int],
                         facing: str) -> Dict[str, int]:
    """
    Calculate goal position relative to current facing direction
    Returns: {'left': X, 'right': X, 'forward': X, 'backward': X}
    """
    dx = target_pos[0] - current_pos[0]
    dy = target_pos[1] - current_pos[1]

    if facing == "UP":
        return {'forward': dy, 'backward': -dy, 'left': -dx, 'right': dx}
    elif facing == "DOWN":
        return {'forward': -dy, 'backward': dy, 'left': dx, 'right': -dx}
    elif facing == "LEFT":
        return {'forward': -dx, 'backward': dx, 'left': -dy, 'right': dy}
    else:  # RIGHT
        return {'forward': dx, 'backward': -dx, 'left': dy, 'right': -dy}

# ------------------- Data Serialization -------------------

def save_navigation_data(data: Dict, filename: str, directory: str = "data") -> Path:
    """Save navigation data to JSON file with automatic directory creation"""
    output_path = Path(directory)
    output_path.mkdir(exist_ok=True)
    filepath = output_path / f"{filename}.json"

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)

    return filepath

def load_navigation_data(filename: str, directory: str = "data") -> Dict:
    """Load navigation data from JSON file"""
    filepath = Path(directory) / f"{filename}.json"
    with open(filepath, 'r') as f:
        return json.load(f)

# ------------------- Map Utilities -------------------

def generate_grid_coordinates(center: Tuple[int, int],
                            radius: int) -> List[Tuple[int, int]]:
    """Generate coordinates in a square grid around center point"""
    return [
        (center[0] + dx, center[1] + dy)
        for dx in range(-radius, radius + 1)
        for dy in range(-radius, radius + 1)
    ]

def calculate_coverage(map_data: Dict[Tuple[int, int], str]) -> float:
    """Calculate percentage of explored area"""
    total = len(map_data)
    if total == 0:
        return 0.0
    explored = sum(1 for v in map_data.values() if v != "UNKNOWN")
    return explored / total * 100

# ------------------- Movement Validation -------------------

def validate_movement(start_pos: Tuple[int, int],
                     end_pos: Tuple[int, int],
                     facing: str) -> bool:
    """Validate if movement between positions matches facing direction"""
    dx = end_pos[0] - start_pos[0]
    dy = end_pos[1] - start_pos[1]

    if facing == "UP" and dy <= 0:
        return False
    elif facing == "DOWN" and dy >= 0:
        return False
    elif facing == "LEFT" and dx >= 0:
        return False
    elif facing == "RIGHT" and dx <= 0:
        return False

    return True

# ------------------- Sensor Processing -------------------

def normalize_sensor_readings(readings: Dict[str, float],
                            max_range: float) -> Dict[str, float]:
    """Normalize sensor readings to 0-1 range"""
    return {k: min(v / max_range, 1.0) for k, v in readings.items()}

def detect_obstacle_pattern(readings: Dict[str, float],
                           threshold: float = 0.3) -> Optional[str]:
    """Detect obstacle pattern from sensor readings"""
    if readings['front'] < threshold and readings['left'] < threshold and readings['right'] < threshold:
        return "dead_end"
    elif readings['front'] < threshold and readings['left'] < threshold:
        return "right_open"
    elif readings['front'] < threshold and readings['right'] < threshold:
        return "left_open"
    elif readings['front'] < threshold:
        return "front_blocked"
    return None

# ------------------- Path Smoothing -------------------

def smooth_path(path: List[Tuple[int, int]],
               window_size: int = 3) -> List[Tuple[int, int]]:
    """Apply simple moving average smoothing to path"""
    if len(path) < window_size:
        return path

    smoothed = []
    for i in range(len(path)):
        x_vals = []
        y_vals = []
        for j in range(max(0, i - window_size // 2),
                      min(len(path), i + window_size // 2 + 1)):
            x_vals.append(path[j][0])
            y_vals.append(path[j][1])
        smoothed.append((int(np.mean(x_vals)), int(np.mean(y_vals))))

    return smoothed

# ------------------- Benchmark Utilities -------------------

def generate_timestamped_dir(base_dir: str = "results") -> Path:
    """Create timestamped directory for benchmark results"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def calculate_path_efficiency(actual_path: List[Tuple[int, int]],
                            optimal_length: int) -> float:
    """Calculate efficiency ratio (optimal/actual)"""
    actual_length = len(actual_path) - 1  # Subtract start position
    if actual_length <= 0:
        return 0.0
    return min(1.0, optimal_length / actual_length)