"""Deterministic grader for Task 1: identify overprovisioned instances."""

from __future__ import annotations

TARGETS = {"i-001", "i-002", "i-003", "i-004", "i-005"}
ANCHORS = {"i-015", "i-016", "i-017", "i-018", "i-019", "i-020"}


def _strict_unit_interval(score: float) -> float:
    return min(0.9999, max(0.0001, round(score, 4)))


def _normalize_inputs(arg1, arg2=None) -> tuple[list[dict], dict | None]:
    """Support both (episode_log, final_state) and (final_state, episode_log)."""
    if isinstance(arg1, list):
        return arg1, arg2 if isinstance(arg2, dict) else None
    if isinstance(arg2, list):
        return arg2, arg1 if isinstance(arg1, dict) else None
    return [], arg1 if isinstance(arg1, dict) else (arg2 if isinstance(arg2, dict) else None)


def grade(episode_log: list[dict], final_state: dict | None = None) -> float:
    """Score 0.0-1.0 based on correct downsizes and avoiding anchor instances."""
    hit, penalized = set(), 0
    for entry in episode_log:
        action = entry.get("action", {})
        if action.get("action_type") != "resize":
            continue
        instance_id = action.get("instance_id", "")
        if instance_id in TARGETS:
            hit.add(instance_id)
        if instance_id in ANCHORS:
            penalized += 1
    return _strict_unit_interval(max(0.0, len(hit) / 5 - penalized * 0.1))


def evaluate(arg1, arg2=None) -> dict:
    episode_log, final_state = _normalize_inputs(arg1, arg2)
    score = grade(episode_log, final_state)
    return {"task_id": "task1", "score": score}


def score(arg1, arg2=None) -> float:
    episode_log, final_state = _normalize_inputs(arg1, arg2)
    return grade(episode_log, final_state)
