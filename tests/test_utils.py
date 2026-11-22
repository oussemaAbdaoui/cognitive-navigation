import math

import pytest

from src.core import utils


def test_distances():
    assert utils.calculate_distance((0, 0), (3, -4)) == 7
    assert utils.calculate_euclidean_distance((0, 0), (3, -4)) == 5.0


# Goal at (2, 3) seen from the origin for each heading.
@pytest.mark.parametrize(
    "facing, expected",
    [
        ("UP", {"forward": 3, "backward": -3, "left": -2, "right": 2}),
        ("DOWN", {"forward": -3, "backward": 3, "left": 2, "right": -2}),
        ("LEFT", {"forward": -2, "backward": 2, "left": -3, "right": 3}),
        ("RIGHT", {"forward": 2, "backward": -2, "left": 3, "right": -3}),
    ],
)
def test_relative_direction(facing, expected):
    assert utils.get_relative_direction((0, 0), (2, 3), facing) == expected


@pytest.mark.parametrize(
    "facing, end, ok",
    [
        ("UP", (0, 2), True),
        ("UP", (0, -1), False),
        ("DOWN", (0, -2), True),
        ("LEFT", (-1, 0), True),
        ("RIGHT", (-1, 0), False),
    ],
)
def test_validate_movement(facing, end, ok):
    assert utils.validate_movement((0, 0), end, facing) is ok


def test_normalize_sensor_readings_clips_to_one():
    out = utils.normalize_sensor_readings({"front": 150, "left": 600, "right": 0}, 300)
    assert out == {"front": 0.5, "left": 1.0, "right": 0.0}


@pytest.mark.parametrize(
    "readings, pattern",
    [
        ({"front": 0.1, "left": 0.1, "right": 0.1}, "dead_end"),
        ({"front": 0.1, "left": 0.1, "right": 0.9}, "right_open"),
        ({"front": 0.1, "left": 0.9, "right": 0.1}, "left_open"),
        ({"front": 0.1, "left": 0.9, "right": 0.9}, "front_blocked"),
        ({"front": 0.9, "left": 0.1, "right": 0.1}, None),
    ],
)
def test_detect_obstacle_pattern(readings, pattern):
    assert utils.detect_obstacle_pattern(readings) == pattern


def test_coverage():
    assert utils.calculate_coverage({}) == 0.0
    grid = {(0, 0): "CLEAR", (0, 1): "UNKNOWN", (1, 0): "OBSTACLE", (1, 1): "UNKNOWN"}
    assert utils.calculate_coverage(grid) == 50.0


def test_generate_grid_coordinates():
    coords = utils.generate_grid_coordinates((1, 1), 1)
    assert len(coords) == 9
    assert (0, 0) in coords and (2, 2) in coords


def test_smooth_path():
    short = [(0, 0), (1, 0)]
    assert utils.smooth_path(short) == short
    assert utils.smooth_path([(0, 0), (0, 3), (0, 6)]) == [(0, 1), (0, 3), (0, 4)]


def test_path_efficiency():
    assert utils.calculate_path_efficiency([(0, 0)], 5) == 0.0
    assert utils.calculate_path_efficiency([(0, i) for i in range(11)], 5) == 0.5
    assert utils.calculate_path_efficiency([(0, 0), (0, 1)], 5) == 1.0


def test_save_and_load_roundtrip(tmp_path):
    data = {"position": [1, 2], "value": math.pi}
    path = utils.save_navigation_data(data, "run", directory=str(tmp_path / "out"))
    assert path.exists()
    assert utils.load_navigation_data("run", directory=str(tmp_path / "out")) == data
