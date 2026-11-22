"""The DeepEval metrics are deterministic (no LLM judge): they only parse the
JSON the benchmark runner writes into each test case."""

import json

import pytest
from deepeval.test_case import LLMTestCase

from src.benchmarks.metrics import (
    CollisionAvoidanceMetric,
    NavigationSuccessMetric,
    PathEfficiencyMetric,
    RobustnessMetric,
)


def case(actual, expected=None):
    return LLMTestCase(
        input="navigate",
        actual_output=actual if isinstance(actual, str) else json.dumps(actual),
        expected_output=json.dumps(expected or {"expected_path_length_range": [5, 8]}),
    )


@pytest.mark.parametrize("reached, score", [(True, 1.0), (False, 0.0)])
def test_navigation_success(reached, score):
    metric = NavigationSuccessMetric()
    assert metric.measure(case({"goal_reached": reached})) == score
    assert metric.is_successful() is reached
    assert metric.__name__ == "Navigation Success"


@pytest.mark.parametrize(
    "steps, score",
    [(5, 1.0), (4, 1.0), (10, 0.5), (0, 0.0)],
)
def test_path_efficiency_uses_lower_bound_of_expected_range(steps, score):
    metric = PathEfficiencyMetric(threshold=0.7)
    assert metric.measure(case({"steps_taken": steps})) == pytest.approx(score)
    assert metric.is_successful() is (score >= 0.7)


def test_path_efficiency_without_steps_scores_zero():
    assert PathEfficiencyMetric().measure(case({})) == 0.0


@pytest.mark.parametrize("collided, score", [(True, 0.0), (False, 1.0)])
def test_collision_avoidance(collided, score):
    metric = CollisionAvoidanceMetric()
    assert metric.measure(case({"collision_occurred": collided})) == score


@pytest.mark.parametrize(
    "payload, score",
    [
        ({"timeout": False, "error_message": None}, 1.0),
        ({"timeout": True, "error_message": None}, 0.0),
        ({"timeout": False, "error_message": "Collision detected"}, 0.0),
    ],
)
def test_robustness(payload, score):
    assert RobustnessMetric().measure(case(payload)) == score


@pytest.mark.parametrize(
    "metric_cls",
    [NavigationSuccessMetric, PathEfficiencyMetric, CollisionAvoidanceMetric, RobustnessMetric],
)
def test_metrics_score_zero_on_unparseable_output(metric_cls):
    metric = metric_cls()
    assert metric.measure(case("not json")) == 0.0
    assert "Error" in metric.reason
    assert not metric.is_successful()
