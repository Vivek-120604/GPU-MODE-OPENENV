"""Grader classes for BiologicalOptimizationEnv — imported directly by OpenEnv validator."""
import math


def _sigmoid_score(performance_score: float, threshold: float) -> float:
    """Sigmoid mapping → strictly in (0.001, 0.999), never 0.0 or 1.0."""
    normalised = (performance_score - threshold) / max(threshold, 0.1)
    raw = 1.0 / (1.0 + math.exp(-3.0 * normalised))
    return max(0.001, min(0.999, raw))


class EasyGrader:
    def grade(self, episode_result) -> float:
        """Grade easy task. Returns float strictly in (0, 1)."""
        if episode_result is None:
            return 0.5
        if isinstance(episode_result, dict):
            perf = float(episode_result.get("performance_score",
                         episode_result.get("final_performance_score", 0.5)))
        elif hasattr(episode_result, "performance_score"):
            perf = float(episode_result.performance_score)
        elif hasattr(episode_result, "final_performance_score"):
            perf = float(episode_result.final_performance_score)
        else:
            perf = 0.5
        return _sigmoid_score(perf, threshold=0.60)


class MediumGrader:
    def grade(self, episode_result) -> float:
        """Grade medium task. Returns float strictly in (0, 1)."""
        if episode_result is None:
            return 0.5
        if isinstance(episode_result, dict):
            perf = float(episode_result.get("performance_score",
                         episode_result.get("final_performance_score", 0.5)))
        elif hasattr(episode_result, "performance_score"):
            perf = float(episode_result.performance_score)
        elif hasattr(episode_result, "final_performance_score"):
            perf = float(episode_result.final_performance_score)
        else:
            perf = 0.5
        return _sigmoid_score(perf, threshold=0.75)


class HardGrader:
    def grade(self, episode_result) -> float:
        """Grade hard task. Returns float strictly in (0, 1)."""
        if episode_result is None:
            return 0.5
        if isinstance(episode_result, dict):
            perf = float(episode_result.get("performance_score",
                         episode_result.get("final_performance_score", 0.5)))
        elif hasattr(episode_result, "performance_score"):
            perf = float(episode_result.performance_score)
        elif hasattr(episode_result, "final_performance_score"):
            perf = float(episode_result.final_performance_score)
        else:
            perf = 0.5
        return _sigmoid_score(perf, threshold=0.85)
       
