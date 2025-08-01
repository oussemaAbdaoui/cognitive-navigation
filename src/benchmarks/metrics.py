import json
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
class NavigationSuccessMetric(BaseMetric):
    """Metric to evaluate navigation success"""

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self.score = 0.0
        self.reason = ""

    def measure(self, test_case: LLMTestCase) -> float:
        """Measure navigation success (1.0 for success, 0.0 for failure)"""
        try:
            # Parse the actual output JSON string
            result = json.loads(test_case.actual_output)
            self.score = 1.0 if result.get('goal_reached', False) else 0.0
            self.reason = f"Goal reached: {result.get('goal_reached', False)}"
            return self.score
        except (json.JSONDecodeError, AttributeError) as e:
            self.score = 0.0
            self.reason = f"Error parsing output: {str(e)}"
            return self.score

    def is_successful(self) -> bool:
        return self.score >= self.threshold

    @property
    def __name__(self):
        return "Navigation Success"

class PathEfficiencyMetric(BaseMetric):
    """Metric to evaluate path efficiency"""

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self.score = 0.0
        self.reason = ""

    def measure(self, test_case: LLMTestCase) -> float:
        """Measure path efficiency (optimal_length / actual_length)"""
        try:
            result = json.loads(test_case.actual_output)
            expected = json.loads(test_case.expected_output)

            expected_range = expected.get('expected_path_length_range', [1, 100])
            actual_steps = result.get('steps_taken', float('inf'))

            if actual_steps == 0:
                self.score = 0.0
                self.reason = "No steps taken"
                return self.score

            optimal_length = expected_range[0]
            self.score = min(1.0, optimal_length / actual_steps)
            self.reason = f"Efficiency: {optimal_length}/{actual_steps} = {self.score:.3f}"
            return self.score

        except (json.JSONDecodeError, AttributeError, ZeroDivisionError) as e:
            self.score = 0.0
            self.reason = f"Error calculating efficiency: {str(e)}"
            return self.score

    def is_successful(self) -> bool:
        return self.score >= self.threshold

    @property
    def __name__(self):
        return "Path Efficiency"

class CollisionAvoidanceMetric(BaseMetric):
    """Metric to evaluate collision avoidance"""

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self.score = 0.0
        self.reason = ""

    def measure(self, test_case: LLMTestCase) -> float:
        """Measure collision avoidance (1.0 for no collisions, 0.0 for collision)"""
        try:
            result = json.loads(test_case.actual_output)
            collision_occurred = result.get('collision_occurred', False)
            self.score = 0.0 if collision_occurred else 1.0
            self.reason = f"Collision occurred: {collision_occurred}"
            return self.score
        except (json.JSONDecodeError, AttributeError) as e:
            self.score = 0.0
            self.reason = f"Error parsing collision data: {str(e)}"
            return self.score

    def is_successful(self) -> bool:
        return self.score >= self.threshold

    @property
    def __name__(self):
        return "Collision Avoidance"

class RobustnessMetric(BaseMetric):
    """Metric to evaluate robustness (no timeouts or errors)"""

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self.score = 0.0
        self.reason = ""

    def measure(self, test_case: LLMTestCase) -> float:
        """Measure robustness"""
        try:
            result = json.loads(test_case.actual_output)
            timeout = result.get('timeout', False)
            error_message = result.get('error_message', None)

            if timeout or error_message:
                self.score = 0.0
                self.reason = f"Timeout: {timeout}, Error: {bool(error_message)}"
            else:
                self.score = 1.0
                self.reason = "No timeouts or errors"
            return self.score
        except (json.JSONDecodeError, AttributeError) as e:
            self.score = 0.0
            self.reason = f"Error parsing robustness data: {str(e)}"
            return self.score

    def is_successful(self) -> bool:
        return self.score >= self.threshold

    @property
    def __name__(self):
        return "Robustness"