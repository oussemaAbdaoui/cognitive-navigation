import asyncio
import json
import re

import httpx
import pytest

from conftest import FAKE_KEY, llm_reply
from src.core.models import Action, SensorReading
from src.core.navigation import LLMNavigationSystem
from src.kernels import native, saamv1, saamv2

FAR = 300.0  # at max_sensor_range_cm: "nothing in range"


# ---------------------------------------------------------------- setup


def test_requires_api_key():
    with pytest.raises(EnvironmentError):
        LLMNavigationSystem(target_position=(1, 1))


def test_api_key_from_environment(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", FAKE_KEY)
    assert LLMNavigationSystem(target_position=(1, 1)).api_key == FAKE_KEY


def test_rejects_unknown_kernel():
    with pytest.raises(ValueError):
        LLMNavigationSystem(target_position=(1, 1), kernel="gpt", api_key=FAKE_KEY)


@pytest.mark.parametrize(
    "kernel, prompt",
    [
        ("saamv1", saamv1.SAAM_V1_SYSTEM_PROMPT),
        ("saamv2", saamv2.SAAM_V2_SYSTEM_PROMPT),
        ("native", native.NATIVE_SYSTEM_PROMPT),
    ],
)
def test_kernel_selects_system_prompt_prefix(make_nav, kernel, prompt):
    nav = make_nav(kernel=kernel)
    system_prompt = nav._get_system_prompt()
    assert system_prompt.startswith(prompt)
    assert "OUTPUT FORMAT" in system_prompt


def test_native_kernel_adds_no_kernel_prompt():
    # The "native" kernel is the plain LLM planner without a SAAM prefix,
    # not a non-LLM baseline.
    assert native.NATIVE_SYSTEM_PROMPT.strip() == ""


def test_initial_state(make_nav):
    nav = make_nav(target=(3, 4))
    assert nav.robot_state.position == (0, 0)
    assert nav.robot_state.facing == "UP"
    assert nav.robot_state.last_goal_distance == 7
    assert nav.internal_map.get_cell((0, 0)).type == "ROBOT"
    assert nav.internal_map.get_cell((3, 4)).type == "GOAL"
    assert not nav.is_goal_reached()


# -------------------------------------------------------- motion model


def test_turns_cycle_clockwise_and_counterclockwise(make_nav):
    nav = make_nav()
    seen = []
    for _ in range(4):
        assert nav.execute_action({"type": "TURN", "direction": "RIGHT"})
        seen.append(nav.robot_state.facing)
    assert seen == ["RIGHT", "DOWN", "LEFT", "UP"]
    nav.execute_action({"type": "TURN", "direction": "LEFT"})
    assert nav.robot_state.facing == "LEFT"
    assert nav.robot_state.position == (0, 0)


def test_invalid_turn_direction_fails(make_nav):
    nav = make_nav()
    assert nav.execute_action({"type": "TURN", "direction": "FORWARD"}) is False
    assert nav.robot_state.facing == "UP"


@pytest.mark.parametrize(
    "facing, expected",
    [("UP", (0, 2)), ("DOWN", (0, -2)), ("LEFT", (-2, 0)), ("RIGHT", (2, 0))],
)
def test_move_follows_cartesian_heading(make_nav, facing, expected):
    nav = make_nav(target=(9, 9))
    nav.robot_state.facing = facing
    action = {"type": "MOVE", "direction": "FORWARD", "cells": 2}
    assert nav._calculate_next_waypoint(Action(**action)) == expected
    assert nav.execute_action(action)
    assert nav.robot_state.position == expected
    assert nav.internal_map.get_cell(expected).type == "ROBOT"
    assert nav.internal_map.get_cell((0, 0)).type == "CLEAR"


def test_turn_does_not_change_waypoint(make_nav):
    nav = make_nav()
    assert nav._calculate_next_waypoint(Action(type="TURN", direction="LEFT")) == (0, 0)


def test_trajectory_memory_keeps_last_five_positions(make_nav):
    nav = make_nav(target=(0, 20))
    for _ in range(7):
        nav.execute_action({"type": "MOVE", "direction": "FORWARD", "cells": 1})
    assert nav.robot_state.recent_positions == [(0, 2), (0, 3), (0, 4), (0, 5), (0, 6)]
    assert nav.robot_state.position == (0, 7)


def test_goal_reached(make_nav):
    nav = make_nav(target=(0, 3))
    nav.execute_action({"type": "MOVE", "direction": "FORWARD", "cells": 3})
    assert nav.is_goal_reached()
    status = nav.get_robot_status()
    assert status["goal_reached"] is True
    assert status["distance_to_goal"] == 0


# ------------------------------------------------------ sensor mapping


def test_front_sensor_marks_clear_cells_then_obstacle(make_nav):
    nav = make_nav(target=(5, 5))
    nav._update_map_from_sensors(SensorReading(front=90.0, left=FAR, right=FAR))
    assert nav.robot_state.step_count == 1
    assert nav.internal_map.get_cell((0, 1)).type == "CLEAR"
    assert nav.internal_map.get_cell((0, 2)).type == "CLEAR"
    assert nav.internal_map.get_cell((0, 3)).type == "OBSTACLE"
    assert nav.internal_map.get_cell((0, 3)).confidence == 0.9


@pytest.mark.parametrize(
    "facing, front, left, right",
    [
        ("UP", (0, 1), (-1, 0), (1, 0)),
        ("DOWN", (0, -1), (1, 0), (-1, 0)),
        ("LEFT", (-1, 0), (0, -1), (0, 1)),
        ("RIGHT", (1, 0), (0, 1), (0, -1)),
    ],
)
def test_side_sensors_rotate_with_heading(make_nav, facing, front, left, right):
    nav = make_nav(target=(9, 9))
    nav.robot_state.facing = facing
    nav._update_map_from_sensors(SensorReading(front=30.0, left=30.0, right=30.0))
    for cell in (front, left, right):
        assert nav.internal_map.get_cell(cell).type == "OBSTACLE"


def test_readings_at_max_range_leave_map_unknown(make_nav):
    nav = make_nav(target=(9, 9))
    before = dict(nav.internal_map.grid)
    nav._update_map_from_sensors(SensorReading(front=FAR, left=FAR, right=400.0))
    assert nav.internal_map.grid.keys() == before.keys()


def test_sensor_never_overwrites_goal(make_nav):
    nav = make_nav(target=(0, 2))
    nav._update_map_from_sensors(SensorReading(front=60.0, left=FAR, right=FAR))
    assert nav.internal_map.get_cell((0, 2)).type == "GOAL"


def test_zero_distance_marks_adjacent_cell(make_nav):
    nav = make_nav(target=(9, 9))
    nav._update_map_from_sensors(SensorReading(front=0.0, left=FAR, right=FAR))
    assert nav.internal_map.get_cell((0, 1)).type == "OBSTACLE"


# ------------------------------------------------- progress monitoring


def test_moving_away_counts_consecutive_increases(make_nav):
    nav = make_nav(target=(0, 5))
    nav.robot_state.facing = "DOWN"
    for expected in (1, 2, 3):
        nav.execute_action({"type": "MOVE", "direction": "FORWARD", "cells": 1})
        assert nav.robot_state.consecutive_distance_increases == expected
    assert nav.robot_state.last_goal_distance == 8


def test_progress_resets_the_streak(make_nav):
    nav = make_nav(target=(0, 5))
    nav.robot_state.facing = "DOWN"
    nav.execute_action({"type": "MOVE", "direction": "FORWARD", "cells": 1})
    nav.execute_action({"type": "TURN", "direction": "RIGHT"})
    nav.execute_action({"type": "TURN", "direction": "RIGHT"})
    nav.execute_action({"type": "MOVE", "direction": "FORWARD", "cells": 1})
    assert nav.robot_state.consecutive_distance_increases == 0


def test_check_progress_keeps_streak_when_distance_unchanged(make_nav):
    # Regression test: _check_progress used to reset the streak whenever
    # execute_action had already recorded the latest distance.
    nav = make_nav(target=(0, 5))
    nav.robot_state.facing = "DOWN"
    for _ in range(3):
        nav.execute_action({"type": "MOVE", "direction": "FORWARD", "cells": 1})
    nav._check_progress()
    assert nav.robot_state.consecutive_distance_increases == 3


def test_check_progress_detects_unrecorded_regression(make_nav):
    nav = make_nav(target=(0, 5))
    nav.robot_state.position = (0, -2)  # moved without execute_action
    nav._check_progress()
    assert nav.robot_state.consecutive_distance_increases == 1
    assert nav.robot_state.last_goal_distance == 7
    nav.robot_state.position = (0, 0)
    nav._check_progress()
    assert nav.robot_state.consecutive_distance_increases == 0


def test_prompt_contains_progress_warning(make_nav):
    nav = make_nav(target=(0, 5))
    nav.robot_state.consecutive_distance_increases = 2
    prompt = nav._build_navigation_prompt(SensorReading(front=30, left=FAR, right=FAR), "map")
    assert "PROGRESS WARNING" in prompt
    assert "Front: 30.0 (BLOCKED)" in prompt
    assert "Left: 300.0 (CLEAR)" in prompt


@pytest.mark.parametrize(
    "facing, text",
    [
        ("UP", "Goal is 2 cells RIGHT and 3 cells UP"),
        ("RIGHT", "Goal is 3 cells LEFT and 2 cells UP"),
        ("DOWN", "Goal is 2 cells LEFT and 3 cells DOWN"),
        ("LEFT", "Goal is 3 cells RIGHT and 2 cells DOWN"),
    ],
)
def test_relative_goal_description(make_nav, facing, text):
    nav = make_nav(target=(2, 3))
    nav.robot_state.facing = facing
    assert nav._get_relative_goal_description().startswith(text)


# ----------------------------------------------------------- ASCII map


def _map_rows(nav):
    lines = nav.get_current_map().split("\n")
    return {int(line[:3]): line[4:] for line in lines[1:] if re.match(r"^\s*-?\d+ ", line)}


def test_map_draws_x_as_columns_and_up_on_top(make_nav):
    # Regression test: the map used to be transposed relative to its legend.
    nav = make_nav(target=(2, 0))
    nav.internal_map.set_cell((0, 1), "OBSTACLE", 1.0)
    rows = _map_rows(nav)
    assert list(rows) == [1, 0]  # highest y first
    # Goal (2, 0) is on the robot's row, two columns to the right.
    assert rows[0].split() == ["↑", "G"]
    assert rows[0].index("G") - rows[0].index("↑") == 6  # 3 chars per column
    # Obstacle (0, 1) is directly above the robot.
    assert rows[1].index("■") == rows[0].index("↑")
    assert "X-axis → Columns" in nav.get_current_map()


# ---------------------------------------------------- response parsing


def test_parse_fenced_json_move(make_nav):
    nav = make_nav()
    reply = llm_reply({"type": "MOVE", "direction": "FORWARD", "cells": 2})
    content = reply["choices"][0]["message"]["content"]
    assert nav._parse_navigation_actions(content) == {
        "next_waypoint": (0, 2),
        "type": "MOVE",
        "direction": "FORWARD",
        "cells": 2,
    }


def test_parse_turn_has_no_cells(make_nav):
    nav = make_nav()
    out = nav._parse_navigation_actions(json.dumps({"action": {"type": "TURN", "direction": "LEFT"}}))
    assert out == {"next_waypoint": (0, 0), "type": "TURN", "direction": "LEFT"}


def test_parse_move_defaults_to_one_cell(make_nav):
    nav = make_nav()
    out = nav._parse_navigation_actions('{"action": {"type": "MOVE", "direction": "FORWARD"}}')
    assert out["cells"] == 1


@pytest.mark.parametrize(
    "text",
    [
        "I think we should go left.",
        "{not json}",
        '{"spatial_analysis": "no action here"}',
        '{"action": {"type": "MOVE", "direction": "FORWARD", "cells": 7}}',
        '{"action": {"type": "FLY"}}',
    ],
)
def test_parse_rejects_bad_responses(make_nav, text):
    assert make_nav()._parse_navigation_actions(text) is None


def test_prompt_move_range_matches_action_schema(make_nav):
    # Regression test: the prompt used to advertise 1-10 cells while the
    # Action schema only accepts 1-5.
    prompt = make_nav()._get_system_prompt()
    assert '"cells": 1-5' in prompt
    assert "1-10" not in prompt


# ------------------------------------------- navigate_step (mocked LLM)


def test_navigate_step_calls_llm_and_returns_action(make_nav, fake_llm):
    nav = make_nav(target=(0, 5), kernel="saamv2")
    fake_llm.replies.append(llm_reply({"type": "MOVE", "direction": "FORWARD", "cells": 3}))

    action = asyncio.run(nav.navigate_step(SensorReading(front=FAR, left=FAR, right=FAR)))

    assert action == {"next_waypoint": (0, 3), "type": "MOVE", "direction": "FORWARD", "cells": 3}
    (request,) = fake_llm.requests
    assert request.url == "https://api.groq.com/openai/v1/chat/completions"
    assert request.headers["Authorization"] == f"Bearer {FAKE_KEY}"
    body = json.loads(request.content)
    assert body["messages"][0]["content"].startswith(saamv2.SAAM_V2_SYSTEM_PROMPT)
    assert "Goal: (0, 5)" in body["messages"][1]["content"]
    assert nav._tokens_used_today == 42
    assert nav._daily_requests == 1
    assert nav.robot_state.step_count == 1


@pytest.mark.parametrize(
    "reply",
    [
        httpx.Response(500, json={"error": "server"}),
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"choices": []}),
        httpx.Response(200, json={"choices": [{"message": {"content": "no json at all"}}]}),
    ],
)
def test_navigate_step_returns_none_on_bad_llm_output(make_nav, fake_llm, reply):
    nav = make_nav()
    fake_llm.replies.append(reply)
    assert asyncio.run(nav.navigate_step(SensorReading(front=FAR, left=FAR, right=FAR))) is None


def test_daily_request_limit_blocks_calls(make_nav, fake_llm):
    nav = make_nav()
    nav._daily_requests = 1000
    assert asyncio.run(nav.navigate_step(SensorReading(front=FAR, left=FAR, right=FAR))) is None
    assert fake_llm.requests == []


def test_scripted_llm_drives_robot_to_goal(make_nav, fake_llm):
    nav = make_nav(target=(2, 3))
    fake_llm.replies += [
        llm_reply({"type": "MOVE", "direction": "FORWARD", "cells": 3}),
        llm_reply({"type": "TURN", "direction": "RIGHT"}),
        llm_reply({"type": "MOVE", "direction": "FORWARD", "cells": 2}),
    ]
    sensors = SensorReading(front=FAR, left=FAR, right=FAR)
    while not nav.is_goal_reached():
        action = asyncio.run(nav.navigate_step(sensors))
        assert action is not None
        assert nav.execute_action(action)
    assert nav.robot_state.position == (2, 3)
    assert nav.robot_state.facing == "RIGHT"
    assert len(fake_llm.requests) == 3
