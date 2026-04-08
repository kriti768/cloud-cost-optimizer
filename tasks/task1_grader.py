"""Deterministic grader for Task 1: identify overprovisioned instances."""

from __future__ import annotations

TARGETS = {"i-001", "i-002", "i-003", "i-004", "i-005"}
ANCHORS = {"i-015", "i-016", "i-017", "i-018", "i-019", "i-020"}


def _strict_unit_interval(score: float) -> float:
    return min(0.9999, max(0.0001, round(score, 4)))


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


def evaluate(episode_log: list[dict], final_state: dict | None = None) -> dict:
    score = grade(episode_log, final_state)
    return {"task_id": "task1", "score": score}
