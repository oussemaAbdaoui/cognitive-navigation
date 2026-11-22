import pytest

from src.benchmarks.environment import NavigationEnvironment
from src.core.models import SensorReading

BOUNDS = (-10, 10, -10, 10)


def test_open_space_reads_max_range():
    env = NavigationEnvironment([], BOUNDS)
    assert env.get_sensor_readings((0, 0), "UP") == SensorReading(front=300, left=300, right=300)


@pytest.mark.parametrize(
    "facing, front, left, right",
    [
        ("UP", (0, 2), (-3, 0), (4, 0)),
        ("DOWN", (0, -2), (3, 0), (-4, 0)),
        ("LEFT", (-2, 0), (0, -3), (0, 4)),
        ("RIGHT", (2, 0), (0, 3), (0, -4)),
    ],
)
def test_rays_follow_cartesian_heading(facing, front, left, right):
    env = NavigationEnvironment([front, left, right], BOUNDS)
    reading = env.get_sensor_readings((0, 0), facing)
    assert (reading.front, reading.left, reading.right) == (60, 90, 120)


def test_ray_reports_nearest_obstacle():
    env = NavigationEnvironment([(0, 5), (0, 2)], BOUNDS)
    assert env.get_sensor_readings((0, 0), "UP").front == 60


def test_ray_from_inside_obstacle_is_zero():
    env = NavigationEnvironment([(0, 0)], BOUNDS)
    assert env.get_sensor_readings((0, 0), "UP").front == 0.0


def test_is_valid_position():
    env = NavigationEnvironment([(1, 1)], BOUNDS)
    assert env.is_valid_position((0, 0))
    assert env.is_valid_position((10, -10))
    assert not env.is_valid_position((1, 1))
    assert not env.is_valid_position((11, 0))
    assert not env.is_valid_position((0, -11))


@pytest.mark.parametrize("facing", ["UP", "DOWN", "LEFT", "RIGHT"])
@pytest.mark.parametrize("sensor", ["front", "left", "right"])
def test_simulated_sensors_agree_with_navigation_map(make_nav, facing, sensor):
    """An obstacle the simulator reports must land on the same cell in the
    robot's internal map (regression test for mismatched conventions)."""
    nav = make_nav(target=(9, 9))
    nav.robot_state.facing = facing
    nav.robot_state.position = (1, 1)

    # Find which cell the map attributes to this sensor at distance 3.
    probe = make_nav(target=(9, 9))
    probe.robot_state.facing = facing
    probe.robot_state.position = (1, 1)
    far = {"front": 300.0, "left": 300.0, "right": 300.0}
    probe._update_map_from_sensors(SensorReading(**{**far, sensor: 90.0}))
    (cell,) = [p for p, c in probe.internal_map.grid.items() if c.type == "OBSTACLE"]

    env = NavigationEnvironment([cell], BOUNDS)
    reading = env.get_sensor_readings((1, 1), facing)
    assert getattr(reading, sensor) == 90
    others = [s for s in far if s != sensor]
    assert all(getattr(reading, s) == 300 for s in others)

    nav._update_map_from_sensors(reading)
    assert nav.internal_map.get_cell(cell).type == "OBSTACLE"


def test_moving_forward_approaches_the_obstacle_the_front_sensor_sees(make_nav):
    env = NavigationEnvironment([(0, 3)], BOUNDS)
    nav = make_nav(target=(9, 9))
    assert env.get_sensor_readings(nav.robot_state.position, nav.robot_state.facing).front == 90
    nav.execute_action({"type": "MOVE", "direction": "FORWARD", "cells": 2})
    assert env.get_sensor_readings(nav.robot_state.position, nav.robot_state.facing).front == 30
