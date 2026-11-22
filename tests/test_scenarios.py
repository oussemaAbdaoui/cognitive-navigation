from collections import deque

import pytest

from src.benchmarks.scenarios import BenchmarkScenario, ScenarioType, create_benchmark_scenarios

SCENARIOS = create_benchmark_scenarios()
BOUNDS = (-10, 10, -10, 10)  # DEFAULT_BOUNDS used by the benchmark runner


def shortest_path_cells(scenario: BenchmarkScenario):
    """4-connected BFS distance from start to goal, or None if unreachable."""
    blocked = set(scenario.obstacles)
    start, goal = scenario.start_position, scenario.goal_position
    queue, seen = deque([(start, 0)]), {start}
    while queue:
        (x, y), d = queue.popleft()
        if (x, y) == goal:
            return d
        for nxt in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            in_bounds = BOUNDS[0] <= nxt[0] <= BOUNDS[1] and BOUNDS[2] <= nxt[1] <= BOUNDS[3]
            if in_bounds and nxt not in blocked and nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, d + 1))
    return None


def test_ten_distinct_scenarios():
    names = [s.name for s in SCENARIOS]
    assert len(SCENARIOS) == 10
    assert len(set(names)) == len(names), "duplicate scenario names"


def test_every_scenario_type_is_used():
    assert {s.scenario_type for s in SCENARIOS} == set(ScenarioType)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_scenario_is_well_formed(scenario):
    assert scenario.difficulty in {"easy", "medium", "hard", "expert"}
    assert scenario.start_position not in scenario.obstacles
    assert scenario.goal_position not in scenario.obstacles
    assert scenario.start_position != scenario.goal_position
    low, high = scenario.expected_path_length_range
    assert 0 < low <= high
    assert scenario.max_steps >= low
    for x, y in scenario.obstacles + [scenario.start_position, scenario.goal_position]:
        assert BOUNDS[0] <= x <= BOUNDS[1] and BOUNDS[2] <= y <= BOUNDS[3]


UNREACHABLE = {"Spiral Navigation"}


@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param(
            s,
            marks=pytest.mark.xfail(
                strict=True,
                reason="Known issue: the goal cell is fully enclosed by obstacles",
            ),
        )
        if s.name in UNREACHABLE
        else s
        for s in SCENARIOS
    ],
    ids=lambda s: s.name,
)
def test_goal_is_reachable(scenario):
    assert shortest_path_cells(scenario) is not None


def test_known_shortest_paths():
    # Pin the geometry so accidental edits to obstacle layouts are noticed.
    by_name = {s.name: shortest_path_cells(s) for s in SCENARIOS}
    assert by_name["Simple Path"] == 5
    assert by_name["Single Obstacle"] == 6
    assert by_name["Narrow Corridor"] == 6
    assert by_name["Open Space Efficiency"] == 16
