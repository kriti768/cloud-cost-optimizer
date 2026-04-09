"""Deterministic grader for Task 2: schedule-aware scaling."""

from __future__ import annotations


def _normalize_inputs(arg1, arg2=None) -> tuple[list[dict], dict]:
    """Support both (episode_log, final_state) and (final_state, episode_log)."""
    if isinstance(arg1, list):
        return arg1, arg2 if isinstance(arg2, dict) else {}
    if isinstance(arg2, list):
        return arg2, arg1 if isinstance(arg1, dict) else {}
    return [], arg1 if isinstance(arg1, dict) else (arg2 if isinstance(arg2, dict) else {})


def _strict_unit_interval(score: float) -> float:
    return min(0.9999, max(0.0001, round(score, 4)))


def grade(episode_log: list[dict], final_state: dict) -> float:
    """Blend timing behavior, SLA preservation, and savings efficiency."""
    sla = final_state.get("sla_violations", 0)
    savings_pct = final_state.get("savings_pct", 0.0)
    schedule_actions = sum(
        1 for entry in episode_log
        if entry.get("action", {}).get("action_type") == "schedule"
    )

    sla_score = 1.0 if sla == 0 else (0.6 if sla <= 2 else (0.3 if sla <= 5 else 0.0))
    savings_score = min(savings_pct / 40.0, 1.0)
    timing_score = min(schedule_actions / 5.0, 1.0)

    return _strict_unit_interval(0.40 * timing_score + 0.35 * sla_score + 0.25 * savings_score)


def evaluate(arg1, arg2=None) -> dict:
    episode_log, final_state = _normalize_inputs(arg1, arg2)
    score = grade(episode_log, final_state)
    return {"task_id": "task2", "score": score}


def score(arg1, arg2=None) -> float:
    episode_log, final_state = _normalize_inputs(arg1, arg2)
    return grade(episode_log, final_state)
