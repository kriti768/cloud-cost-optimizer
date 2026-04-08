"""Deterministic grader for Task 3: spot + reserved portfolio blending."""

from __future__ import annotations


def grade(episode_log: list[dict], final_state: dict) -> float:
    """Score workload classification, interruption response, efficiency, and budget discipline."""
    sla = final_state.get("sla_violations", 0)
    savings_pct = final_state.get("savings_pct", 0.0)
    spot_instances = final_state.get("spot_instances", 0)
    reserved_instances = final_state.get("reserved_instances", 0)

    classification_score = min((spot_instances + reserved_instances) / 8.0, 1.0)
    efficiency_score = min(savings_pct / 50.0, 1.0)
    budget_score = 1.0 if sla < 3 else 0.5

    interruptions = [
        entry for entry in episode_log
        if entry.get("info", {}).get("interruptions", 0) > 0
    ]
    restore_hours = [
        entry["hour"] for entry in episode_log
        if entry.get("action", {}).get("action_type") == "restore"
    ]

    if not interruptions:
        response_score = 0.5
    else:
        fast_restores = sum(
            1 for entry in interruptions
            if any(abs(restore_hour - entry["hour"]) <= 2 for restore_hour in restore_hours)
        )
        response_score = fast_restores / len(interruptions)

    return round(
        0.35 * classification_score
        + 0.30 * response_score
        + 0.20 * efficiency_score
        + 0.15 * budget_score,
        4,
    )


def evaluate(episode_log: list[dict], final_state: dict) -> dict:
    score = grade(episode_log, final_state)
    return {"task_id": "task3", "score": score}
