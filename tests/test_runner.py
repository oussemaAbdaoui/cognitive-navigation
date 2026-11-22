"""Benchmark runner tests. The LLM is replaced by a scripted fake, so these
check the runner's bookkeeping (collisions, success, reports), not planning."""

import asyncio
import json
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

from conftest import FAKE_KEY, llm_reply  # noqa: E402
from src.benchmarks import runner  # noqa: E402
from src.benchmarks.environment import NavigationEnvironment  # noqa: E402
from src.benchmarks.runner import NavigationBenchmark, NavigationResult  # noqa: E402
from src.benchmarks.scenarios import create_benchmark_scenarios  # noqa: E402
from src.core.navigation import LLMNavigationSystem  # noqa: E402

SCENARIOS = {s.name: s for s in create_benchmark_scenarios()}
MOVE = lambda n: llm_reply({"type": "MOVE", "direction": "FORWARD", "cells": n})  # noqa: E731
TURN = lambda d: llm_reply({"type": "TURN", "direction": d})  # noqa: E731


@pytest.fixture(autouse=True)
def no_delays(monkeypatch):
    monkeypatch.setattr(runner, "API_RETRY_DELAY", 0.0)


def run(name, fake_llm, replies, kernel="native"):
    scenario = SCENARIOS[name]
    fake_llm.replies += replies
    bench = NavigationBenchmark([scenario], FAKE_KEY)
    nav = LLMNavigationSystem(scenario.goal_position, kernel=kernel, api_key=FAKE_KEY)
    return asyncio.run(bench._run_single_scenario(scenario, nav))


def test_requires_scenarios_and_api_key():
    with pytest.raises(ValueError):
        NavigationBenchmark([], FAKE_KEY)
    with pytest.raises(ValueError):
        NavigationBenchmark(list(SCENARIOS.values()), "")


def test_path_collides_checks_every_cell_of_a_move():
    env = NavigationEnvironment([(0, 2)], runner.DEFAULT_BOUNDS)
    assert not NavigationBenchmark._path_collides(env, (0, 0), (0, 1))
    assert NavigationBenchmark._path_collides(env, (0, 0), (0, 2))
    assert NavigationBenchmark._path_collides(env, (0, 0), (0, 4))  # passes through
    assert not NavigationBenchmark._path_collides(env, (0, 0), (0, 0))  # turn
    assert NavigationBenchmark._path_collides(env, (0, 9), (0, 11))  # leaves bounds


def test_straight_run_succeeds(fake_llm):
    result = run("Simple Path", fake_llm, [MOVE(5)])
    assert result.success and result.goal_reached
    assert not result.collision_occurred and not result.timeout
    assert result.steps_taken == 1
    assert result.path_trace == [(0, 0), (0, 5)]
    assert result.path_efficiency == 1.0
    assert result.error_message is None


def test_detour_around_single_obstacle(fake_llm):
    replies = [TURN("RIGHT"), MOVE(1), TURN("LEFT"), MOVE(4), TURN("LEFT"), MOVE(1)]
    result = run("Single Obstacle", fake_llm, replies)
    assert result.success
    assert result.steps_taken == 6
    assert result.path_trace[-1] == (0, 4)
    assert result.path_efficiency == 1.0  # 6 steps vs. lower bound 6


def test_moving_through_an_obstacle_is_a_collision(fake_llm):
    # Regression test: a 4-cell move used to jump over the obstacle at (0, 2).
    result = run("Single Obstacle", fake_llm, [MOVE(4)])
    assert result.collision_occurred
    assert not result.success
    assert result.error_message == "Collision detected"


def test_no_action_stops_the_run(fake_llm):
    result = run("Simple Path", fake_llm, [{"choices": []}])
    assert not result.success
    assert result.error_message == "No action returned from navigation system"


def test_step_budget_exhaustion_is_a_timeout(fake_llm):
    scenario = SCENARIOS["Simple Path"]
    replies = [TURN("LEFT")] * scenario.max_steps
    result = run("Simple Path", fake_llm, replies)
    assert result.timeout and not result.success
    assert result.steps_taken == scenario.max_steps


def test_run_all_scenarios_writes_results_per_kernel(fake_llm, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    scenario = SCENARIOS["Simple Path"]
    bench = NavigationBenchmark([scenario], FAKE_KEY)
    fake_llm.replies += [MOVE(5)] * len(bench.kernels)

    results = asyncio.run(bench.run_all_scenarios())

    assert [r.kernel for r in results] == ["saamv1", "saamv2", "native"]
    assert all(r.success for r in results)
    (run_dir,) = (tmp_path / "benchmark_results").iterdir()
    for kernel in bench.kernels:
        out = run_dir / "Simple_Path" / kernel
        assert json.loads((out / "result.json").read_text())["kernel"] == kernel
        assert (out / "path_visualization.png").stat().st_size > 0


def _result(name, kernel, success):
    return NavigationResult(
        scenario_name=name, kernel=kernel, success=success, steps_taken=5,
        path_efficiency=1.0 if success else 0.5, goal_reached=success,
        timeout=not success, collision_occurred=False, final_position=(0, 0),
        execution_time=1.0,
    )


def test_report_pairs_results_with_their_own_scenario():
    # Regression test: results (one per scenario x kernel) used to be
    # zipped against the scenario list, mislabelling and dropping results.
    easy, hard = SCENARIOS["Simple Path"], SCENARIOS["Dead End Escape"]
    bench = NavigationBenchmark([easy, hard], FAKE_KEY)
    bench.results = [_result(easy.name, k, True) for k in bench.kernels] + [
        _result(hard.name, k, False) for k in bench.kernels
    ]

    report = bench.generate_detailed_report()

    assert report["summary"]["total_scenarios"] == 6
    assert report["difficulty_breakdown"]["easy"]["total"] == 3
    assert report["difficulty_breakdown"]["easy"]["success_rate"] == 1.0
    assert report["difficulty_breakdown"]["hard"]["total"] == 3
    assert report["difficulty_breakdown"]["hard"]["success_rate"] == 0.0
    assert {k: v["total"] for k, v in report["kernel_breakdown"].items()} == {
        "saamv1": 2, "saamv2": 2, "native": 2,
    }
    details = report["scenario_details"]
    assert len(details) == 6
    assert all(d["difficulty"] == ("easy" if d["name"] == easy.name else "hard") for d in details)
    assert report["failure_analysis"]["timeout"] == 3

    dataset = bench.create_deepeval_dataset()
    assert len(dataset.test_cases) == 6
    expected = json.loads(dataset.test_cases[-1].expected_output)
    assert expected["difficulty"] == "hard"


def test_save_results_writes_report_and_raw_data(tmp_path):
    bench = NavigationBenchmark([SCENARIOS["Simple Path"]], FAKE_KEY)
    bench.results = [_result("Simple Path", "native", True)]
    out = Path(bench.save_results(base_dir=str(tmp_path)))
    assert out.parent == tmp_path
    report = json.loads((out / "benchmark_report.json").read_text())
    assert report["summary"]["success_rate"] == 1.0
    raw = json.loads((out / "raw_results.json").read_text())
    assert raw[0]["scenario_name"] == "Simple Path"
