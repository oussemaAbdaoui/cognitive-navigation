import pytest
from pydantic import ValidationError

from src.core.models import Action, GridCell, InternalMap, RobotState, SensorReading


def test_action_accepts_turn_and_bounded_moves():
    assert Action(type="TURN", direction="LEFT").cells is None
    assert Action(type="MOVE", direction="FORWARD", cells=5).cells == 5


@pytest.mark.parametrize("cells", [0, -1, 6, 10])
def test_action_rejects_out_of_range_cells(cells):
    with pytest.raises(ValidationError):
        Action(type="MOVE", direction="FORWARD", cells=cells)


@pytest.mark.parametrize("bad", [{"type": "JUMP"}, {"type": "TURN", "direction": "UP"}])
def test_action_rejects_unknown_type_or_direction(bad):
    with pytest.raises(ValidationError):
        Action(**bad)


def test_sensor_reading_coerces_numbers():
    reading = SensorReading(front=30, left="60.5", right=300)
    assert (reading.front, reading.left, reading.right) == (30.0, 60.5, 300.0)


def test_unknown_cells_default_and_are_not_stored():
    m = InternalMap()
    cell = m.get_cell((3, 4))
    assert cell == GridCell(type="UNKNOWN", confidence=0.0)
    assert (3, 4) not in m.grid


def test_set_cell_updates_type_confidence_and_bounds():
    m = InternalMap()
    m.set_cell((2, -3), "OBSTACLE", 0.9)
    m.set_cell((-1, 5), "CLEAR", 0.8)
    assert m.get_cell((2, -3)).type == "OBSTACLE"
    assert m.get_cell((2, -3)).confidence == 0.9
    assert m.bounds == {"min_row": -1, "max_row": 2, "min_col": -3, "max_col": 5}

    m.set_cell((2, -3), "CLEAR", 0.5)
    assert m.get_cell((2, -3)).type == "CLEAR"
    assert m.get_cell((2, -3)).confidence == 0.5


def test_robot_state_instances_do_not_share_history():
    a, b = RobotState(), RobotState()
    a.recent_positions.append((1, 1))
    assert b.recent_positions == []
