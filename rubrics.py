"""
Cloud Cost Optimizer — LLMJudge Rubrics
========================================
Uses openenv.core.rubrics.LLMJudge (OpenEnv v0.2.2+).

Three task rubrics, each with multiple evaluation dimensions:
  - Task 1: Overprovisioning identification   (easy)
  - Task 2: Schedule-aware scaling            (medium)
  - Task 3: Spot + reserved portfolio         (hard)

Plus a trajectory-level EvalHarness that judges the full episode arc,
rewarding agents that show clear reasoning across multiple decision points —
the "multiple trajectories" criterion judges look for.

Usage
-----
    from rubrics import (
        task1_judge, task2_judge, task3_judge,
        trajectory_harness, score_episode,
    )

    # Per-step scoring (called inside your training loop)
    step_score = await task1_judge(action=action_dict, observation=obs_dict)

    # End-of-episode trajectory scoring
    full_score = await score_episode(task=1, episode_log=env.episode_log())
"""

import json
import asyncio
from dataclasses import dataclass
from typing import Any

try:
    from openenv.core.rubrics import LLMJudge, EvalHarness, Rubric, RubricDimension
except ImportError:
    # Fallback shim so the file is importable before openenv-core is installed.
    # Replace with the real classes once `pip install openenv-core`.
    class LLMJudge:  # type: ignore
        def __init__(self, prompt_template: str, client: Any = None,
                     dimensions: list | None = None, weights: dict | None = None):
            self.prompt_template = prompt_template
            self.client = client
            self.dimensions = dimensions or []
            self.weights = weights or {}

        async def __call__(self, action, observation, **ctx):
            raise RuntimeError("Install openenv-core to use LLMJudge")

    class EvalHarness:  # type: ignore
        def __init__(self, judges: list, aggregation: str = "weighted_mean"):
            self.judges = judges
            self.aggregation = aggregation

        async def evaluate(self, trajectory: list[dict]) -> dict:
            raise RuntimeError("Install openenv-core to use EvalHarness")

    class RubricDimension:  # type: ignore
        def __init__(self, name: str, weight: float, description: str):
            self.name = name
            self.weight = weight
            self.description = description

    class Rubric:  # type: ignore
        pass


def _strict_unit_interval(score: float) -> float:
    """Clamp scores to the open interval (0, 1) for validator compatibility."""
    return min(0.90, max(0.10, round(float(score), 4)))


class StrictScoreRubric(Rubric):
    """Wrapper that prevents public rubric entry points from returning 0.0 or 1.0."""

    def __init__(self, judge):
        try:
            super().__init__()
        except Exception:
            pass
        self.judge = judge

    @property
    def client(self):
        return getattr(self.judge, "client", getattr(self.judge, "_client", None))

    @client.setter
    def client(self, value):
        if hasattr(self.judge, "client"):
            self.judge.client = value
        if hasattr(self.judge, "_client"):
            self.judge._client = value

    def __call__(self, action, observation):
        return self.forward(action, observation)

    async def forward(self, action, observation):
        result = self.judge(action, observation)
        if hasattr(result, "__await__"):
            result = await result
        if isinstance(result, dict):
            result = result.get("score", 0.5)
        return _strict_unit_interval(result)


def _strict_rubric(judge):
    return StrictScoreRubric(judge)


# ---------------------------------------------------------------------------
# Shared system context injected into every prompt
# ---------------------------------------------------------------------------

_SYSTEM_CONTEXT = """\
You are an expert cloud infrastructure evaluator assessing an AI agent's \
decisions inside a simulated AWS cost-optimization environment.

Environment rules you must know:
- Instances have a type (e.g. m5.xlarge), workload category, and pricing model \
(on-demand / spot / reserved).
- CPU utilization above 90% is an SLA breach. Each breach reduces the final score.
- Downsizing reduces cost but also reduces headroom, increasing SLA risk on spikes.
- Spot instances cost 68% less but can be interrupted at any time.
- Reserved instances cost 40% less with no interruption risk (1-year commitment).
- Data-pipeline workloads peak at night (2am–6am); web-api workloads peak at 2pm.
- Instances i-001 through i-005 are clearly overprovisioned (CPU ≤ 8%).
- Instances i-015 through i-020 are stable anchors — resizing them causes SLA breaches.

Score strictly within the open interval (0, 1). Never return exactly 0.0 or 1.0.
Output ONLY a JSON object in this exact format:
{"score": <float between 0 and 1>, "reasoning": "<one concise sentence>"}

Do not include markdown, code blocks, or any other text outside the JSON object.
"""


# ---------------------------------------------------------------------------
# Task 1 — Overprovisioning identification (Easy)
# ---------------------------------------------------------------------------
#
# What good looks like: agent inspects CPU/memory, correctly identifies the 5
# overprovisioned instances (i-001 to i-005), downsizes them to an appropriate
# smaller type, and does NOT touch the stable anchors (i-015 to i-020).
#
# Dimensions:
#   - target_accuracy    (0.5): Did it resize the right instances?
#   - sizing_quality     (0.3): Did it pick a sensible new type (not overkill)?
#   - anchor_discipline  (0.2): Did it leave stable anchors alone?

_TASK1_STEP_TEMPLATE = """\
{system_context}

TASK: Identify and downsize overprovisioned cloud instances.

CURRENT OBSERVATION:
{observation}

AGENT ACTION:
{action}

EVALUATION INSTRUCTIONS:
Score this single action on target accuracy — did the agent correctly identify \
an overprovisioned instance and downsize it appropriately?

A GOOD action looks like:
  resize i-001 (cpu_avg=4.2%, m5.4xlarge → m5.large) — correctly identifies \
a clearly underutilized instance and moves it to a type that still has headroom.

A BAD action looks like:
  resize i-016 (cpu_avg=68%, r5.xlarge → t3.small) — resizes a healthy \
database instance to an undersized type, guaranteeing SLA breach.

Score 1.0: Correct target, sensible new type, clear waste reduction.
Score 0.5: Correct target but overshoots (e.g. downsizes too aggressively, leaving no headroom).
Score 0.0: Wrong target (touches a healthy/anchor instance) or action makes things worse.
"""

_TASK1_TRAJECTORY_TEMPLATE = """\
{system_context}

TASK: Identify and downsize overprovisioned cloud instances.

FULL EPISODE TRAJECTORY (chronological action log):
{trajectory}

FINAL STATE:
{final_state}

EVALUATION INSTRUCTIONS:
Evaluate the agent's complete episode strategy across three dimensions.
For each dimension score 0.0–1.0, then output the combined weighted score.

Dimension 1 — target_accuracy (weight 0.50):
  Did the agent correctly identify all 5 overprovisioned instances (i-001 to i-005)?
  Score 1.0 = all 5 found and resized. Score 0.0 = none found or wrong targets chosen.

Dimension 2 — sizing_quality (weight 0.30):
  Were the chosen replacement types appropriate? An agent that resizes m5.4xlarge \
(16 vCPU) to m5.large (2 vCPU) for a 5% CPU workload scores higher than one that \
only goes to m5.2xlarge (still 8x oversized).

Dimension 3 — anchor_discipline (weight 0.20):
  Did the agent avoid touching i-015 through i-020 (the stable anchor instances)?
  Score 1.0 = zero anchor touches. Score 0.0 = any anchor was resized.

Compute: final_score = 0.5*d1 + 0.3*d2 + 0.2*d3

Output JSON: {{"score": <weighted_float>, "reasoning": "<one sentence summarising \
the agent's strategy and its main strength or failure>"}}
"""

task1_step_judge = LLMJudge(
    prompt_template=_TASK1_STEP_TEMPLATE,
    dimensions=[
        RubricDimension("target_accuracy", weight=1.0,
                        description="Correct instance targeted for downsizing"),
    ],
)

task1_trajectory_judge = LLMJudge(
    prompt_template=_TASK1_TRAJECTORY_TEMPLATE,
    dimensions=[
        RubricDimension("target_accuracy",   weight=0.50,
                        description="All 5 overprovisioned instances found and resized"),
        RubricDimension("sizing_quality",    weight=0.30,
                        description="Replacement type is appropriately sized, not just slightly smaller"),
        RubricDimension("anchor_discipline", weight=0.20,
                        description="Stable anchor instances i-015 to i-020 left untouched"),
    ],
    weights={"target_accuracy": 0.50, "sizing_quality": 0.30, "anchor_discipline": 0.20},
)


# ---------------------------------------------------------------------------
# Task 2 — Schedule-aware scaling (Medium)
# ---------------------------------------------------------------------------
#
# What good looks like: agent learns web-api peaks at 2pm and data-pipeline
# peaks at 2am, schedules scale-downs during the quiet windows, and avoids
# resizing anything during peak hours. Zero SLA breaches = perfect.
#
# Dimensions:
#   - timing_intelligence (0.40): Did it act in off-peak windows?
#   - sla_preservation    (0.35): Did it avoid SLA breaches through the episode?
#   - savings_efficiency  (0.25): How much cost did it save vs the maximum possible?

_TASK2_STEP_TEMPLATE = """\
{system_context}

TASK: Schedule-aware scaling — scale instances down during off-peak hours \
and back up before traffic spikes.

CURRENT HOUR: {hour_of_day} (0=midnight, 14=2pm business peak, 3=3am night trough)

WORKLOAD PATTERNS (agent should have learned these):
  - web_api instances: peak traffic 10am–6pm, quiet 10pm–7am
  - data_pipeline instances: peak activity 2am–6am, quiet 8am–midnight

CURRENT OBSERVATION:
{observation}

AGENT ACTION:
{action}

EVALUATION INSTRUCTIONS:
Score the timing wisdom of this single action. The agent should scale DOWN \
during quiet hours and scale UP before peaks.

Score 1.0: Action correctly anticipates the next traffic pattern \
(e.g. schedules a web-api scale-down at hour 22, before the overnight quiet period).
Score 0.5: Action is directionally correct but poorly timed \
(e.g. scales down at hour 12, right before the 2pm peak — risky).
Score 0.0: Action is actively harmful — scales down a web-api instance at hour 13 \
(one hour before peak), or resizes a data-pipeline instance at 3am during its peak.
"""

_TASK2_TRAJECTORY_TEMPLATE = """\
{system_context}

TASK: Schedule-aware scaling across a 72-hour simulated episode.

FULL EPISODE TRAJECTORY (chronological — each entry is one hour):
{trajectory}

FINAL STATE:
{final_state}

EVALUATION INSTRUCTIONS:
Evaluate the agent's scheduling strategy across three dimensions.

Dimension 1 — timing_intelligence (weight 0.40):
  Did the agent's scale-down actions consistently target off-peak windows?
  Look for: schedule actions placed at hours 21–8 for web-api, and hours 8–23 for data-pipeline.
  Penalise: scale-downs placed within 2 hours of a known peak.
  Score 1.0 = all scale-downs timed correctly. Score 0.0 = agent ignored patterns entirely.

Dimension 2 — sla_preservation (weight 0.35):
  How well did the agent protect SLA throughout the 72-hour episode?
  Check `sla_violations` in the final state.
  Score 1.0 = zero violations. Score 0.5 = 1–2 violations. Score 0.0 = 5+ violations.

Dimension 3 — savings_efficiency (weight 0.25):
  What fraction of the theoretical maximum savings did the agent capture?
  (total_savings_usd / max_possible_savings_usd) from the final state.
  Score 1.0 = ≥80% of max captured. Score 0.5 = 40–80%. Score 0.0 = <40%.

Note: An agent can score well on timing but poorly on savings if it was too conservative \
(never scaled down enough). Both dimensions matter.

Compute: final_score = 0.40*d1 + 0.35*d2 + 0.25*d3

Output JSON: {{"score": <weighted_float>, "reasoning": "<one sentence describing \
the agent's scheduling strategy and where it succeeded or failed>"}}
"""

task2_step_judge = LLMJudge(
    prompt_template=_TASK2_STEP_TEMPLATE,
    dimensions=[
        RubricDimension("timing_intelligence", weight=1.0,
                        description="Action timed correctly relative to workload traffic pattern"),
    ],
)

task2_trajectory_judge = LLMJudge(
    prompt_template=_TASK2_TRAJECTORY_TEMPLATE,
    dimensions=[
        RubricDimension("timing_intelligence", weight=0.40,
                        description="Scale-down actions placed in off-peak windows"),
        RubricDimension("sla_preservation",    weight=0.35,
                        description="SLA violations minimised across the full episode"),
        RubricDimension("savings_efficiency",  weight=0.25,
                        description="Fraction of theoretical maximum savings captured"),
    ],
    weights={"timing_intelligence": 0.40, "sla_preservation": 0.35, "savings_efficiency": 0.25},
)


# ---------------------------------------------------------------------------
# Task 3 — Spot + reserved portfolio blending (Hard)
# ---------------------------------------------------------------------------
#
# What good looks like: agent identifies which workloads are safe for spot
# (stateless, batch), commits steady predictable workloads to reserved pricing,
# keeps databases on on-demand, monitors for spot interruptions and restores
# promptly, and manages this across 3 accounts within budget.
#
# This is the hardest rubric because it requires multi-step reasoning:
#   1. Classify workloads by interruption tolerance
#   2. Choose the right pricing model per workload
#   3. React to spot interruptions without SLA breach
#   4. Balance savings vs risk across a portfolio
#
# Dimensions:
#   - workload_classification (0.35): Right pricing model for each workload type?
#   - interruption_response   (0.30): Did it restore interrupted spot instances fast?
#   - portfolio_efficiency    (0.20): Overall cost reduction vs baseline
#   - budget_discipline       (0.15): Stayed within budget across all 3 accounts?

_TASK3_STEP_TEMPLATE = """\
{system_context}

TASK: Spot + reserved instance portfolio management across 3 accounts.

PRICING MODEL RULES the agent must respect:
  - SAFE for spot:     stateless, batch_job workloads (can tolerate interruption)
  - SAFE for reserved: database, steady web_api workloads (predictable, never interrupt)
  - NEVER spot:        database workloads (interruption = guaranteed SLA breach)

CURRENT OBSERVATION:
{observation}

AGENT ACTION:
{action}

EVALUATION INSTRUCTIONS:
Score the agent's single pricing or restoration decision.

Score 1.0 — examples of perfect actions:
  convert_spot i-011 (workload=stateless) — correct, stateless is spot-safe.
  reserve i-015 (workload=database, pricing=on_demand) — correct, databases need stability.
  restore i-012 after spot interruption within 2 hours — fast recovery.

Score 0.5 — examples of acceptable but suboptimal actions:
  reserve i-011 (workload=stateless) — safe but leaves money on the table (spot would be cheaper).
  convert_spot i-009 (workload=data_pipeline) — data pipelines run at night, risky but not catastrophic.

Score 0.0 — examples of harmful actions:
  convert_spot i-015 (workload=database) — converts a database to spot, will cause SLA breach.
  noop while i-012 is interrupted for >3 hours — ignores a spot interruption, SLA breach accumulating.
"""

_TASK3_TRAJECTORY_TEMPLATE = """\
{system_context}

TASK: Spot + reserved instance portfolio blending across a 72-hour episode with 3 accounts.

FULL EPISODE TRAJECTORY:
{trajectory}

FINAL STATE:
{final_state}

SPOT INTERRUPTION LOG (extracted from trajectory):
{interruption_log}

EVALUATION INSTRUCTIONS:
This is the hardest task. Evaluate across four dimensions.

Dimension 1 — workload_classification (weight 0.35):
  Did the agent choose the right pricing model for each workload type?
  Perfect: stateless/batch → spot, stable databases → reserved, transitional → on-demand.
  Check: did any database (i-015, i-016) end up on spot? If yes, dimension score ≤ 0.2.
  Check: did any stateless instance (i-011, i-012, i-013) stay on expensive on-demand \
when it could have been spot or reserved? Penalise opportunity cost.

Dimension 2 — interruption_response (weight 0.30):
  How quickly did the agent restore interrupted spot instances?
  Score 1.0 = all interruptions restored within 2 hours.
  Score 0.5 = restored within 4 hours.
  Score 0.0 = any interruption unresolved for >6 hours, or ignored entirely.
  Note: if the agent converted no instances to spot, this dimension scores 0.5 \
(neutral — no risk taken, no response needed).

Dimension 3 — portfolio_efficiency (weight 0.20):
  Total cost reduction vs on-demand baseline across all 3 accounts.
  Score 1.0 = ≥50% cost reduction. Score 0.5 = 25–50%. Score 0.0 = <10%.

Dimension 4 — budget_discipline (weight 0.15):
  Did the agent stay within the monthly_budget constraint in the final state?
  Score 1.0 = final projected monthly cost ≤ budget. Score 0.0 = over budget.

Compute: final_score = 0.35*d1 + 0.30*d2 + 0.20*d3 + 0.15*d4

This rubric intentionally rewards agents that take *calculated* risk (converting \
appropriate workloads to spot, then managing interruptions) over agents that play \
it safe (reserving everything) or are reckless (converting databases to spot).

Output JSON: {{"score": <weighted_float>, "reasoning": "<one sentence describing \
the agent's portfolio strategy, its risk tolerance, and where it succeeded or failed>"}}
"""

task3_step_judge = LLMJudge(
    prompt_template=_TASK3_STEP_TEMPLATE,
    dimensions=[
        RubricDimension("pricing_decision", weight=1.0,
                        description="Correct pricing model chosen for the workload type"),
    ],
)

task3_trajectory_judge = LLMJudge(
    prompt_template=_TASK3_TRAJECTORY_TEMPLATE,
    dimensions=[
        RubricDimension("workload_classification", weight=0.35,
                        description="Right pricing model selected per workload type"),
        RubricDimension("interruption_response",   weight=0.30,
                        description="Spot interruptions detected and restored quickly"),
        RubricDimension("portfolio_efficiency",    weight=0.20,
                        description="Total cost reduction across all three accounts"),
        RubricDimension("budget_discipline",       weight=0.15,
                        description="Monthly projected cost stays within budget cap"),
    ],
    weights={
        "workload_classification": 0.35,
        "interruption_response":   0.30,
        "portfolio_efficiency":    0.20,
        "budget_discipline":       0.15,
    },
)


# ---------------------------------------------------------------------------
# Trajectory-level EvalHarness
# ---------------------------------------------------------------------------
# Judges the full episode arc beyond individual actions:
#   - Did the agent show a coherent strategy that evolved over time?
#   - Did it adapt when things went wrong (spike, interruption)?
#   - Did it avoid repeating the same mistake twice?
# This is what the "multiple trajectories" judging criterion measures.

_TRAJECTORY_COHERENCE_TEMPLATE = """\
{system_context}

TRAJECTORY COHERENCE EVALUATION

You are evaluating not just individual decisions, but the arc of an entire \
72-hour cloud cost optimization episode.

FULL EPISODE TRAJECTORY (sampled key moments):
{trajectory}

EPISODE SUMMARY STATS:
{final_state}

EVALUATION DIMENSIONS:

1. Strategic coherence (0.40):
   Did the agent pursue a consistent, recognisable strategy throughout the episode?
   Examples of coherent strategies:
     - "Conservative resizer": methodically identified waste, downsized incrementally, \
never risked SLA
     - "Spot opportunist": converted stateless workloads to spot early, monitored \
closely, restored promptly when interrupted
     - "Scheduler": learned traffic patterns, used schedule actions to capture off-peak \
savings without touching instance types
   An incoherent agent randomly mixes all action types with no apparent logic. \
Score 1.0 = clearly coherent strategy. Score 0.0 = random, reactive, no evident plan.

2. Adaptability (0.35):
   When something went wrong (SLA breach, spot interruption, unexpected spike), \
did the agent adapt its subsequent decisions?
   Score 1.0 = clear behavioural change after adverse events (e.g. after an SLA \
breach, agent becomes more conservative about downsizing similar instances).
   Score 0.5 = partial adaptation (acknowledged the event but didn't fully adjust).
   Score 0.0 = agent repeated the same mistake after an adverse event, or ignored \
adverse events entirely.

3. Exploration vs exploitation (0.25):
   Did the agent try multiple action types (resize, schedule, spot, reserve) to \
find the best strategy, or did it myopically exploit only one?
   Score 1.0 = agent used 3+ distinct action types purposefully.
   Score 0.5 = agent used 2 action types.
   Score 0.0 = agent used only noop or a single action type throughout.

Compute: coherence_score = 0.40*d1 + 0.35*d2 + 0.25*d3

Output JSON: {{"score": <weighted_float>, "reasoning": "<one sentence identifying \
the agent's overall trajectory strategy and its defining characteristic>"}}
"""

trajectory_coherence_judge = LLMJudge(
    prompt_template=_TRAJECTORY_COHERENCE_TEMPLATE,
    dimensions=[
        RubricDimension("strategic_coherence",        weight=0.40,
                        description="Consistent recognisable strategy across the episode"),
        RubricDimension("adaptability",               weight=0.35,
                        description="Behavioural change after adverse events"),
        RubricDimension("exploration_vs_exploitation", weight=0.25,
                        description="Diversity of action types used purposefully"),
    ],
    weights={
        "strategic_coherence":         0.40,
        "adaptability":                0.35,
        "exploration_vs_exploitation": 0.25,
    },
)

# Public rubric entry points are wrapped because the hackathon validator requires
# every score to be strictly inside (0, 1), while OpenEnv's LLMJudge clamps to
# the inclusive range [0, 1].
task1_step_judge = _strict_rubric(task1_step_judge)
task1_trajectory_judge = _strict_rubric(task1_trajectory_judge)
task2_step_judge = _strict_rubric(task2_step_judge)
task2_trajectory_judge = _strict_rubric(task2_trajectory_judge)
task3_step_judge = _strict_rubric(task3_step_judge)
task3_trajectory_judge = _strict_rubric(task3_trajectory_judge)
trajectory_coherence_judge = _strict_rubric(trajectory_coherence_judge)

trajectory_harness = EvalHarness(
    judges=[
        task1_trajectory_judge,   # task-specific outcome
        task2_trajectory_judge,
        task3_trajectory_judge,
        trajectory_coherence_judge,  # cross-task arc quality
    ],
    aggregation="weighted_mean",
)


# ---------------------------------------------------------------------------
# Convenience scorer — call this at the end of each episode
# ---------------------------------------------------------------------------

@dataclass
class EpisodeScore:
    task_score: float          # outcome score for the specific task (0–1)
    coherence_score: float     # trajectory arc quality (0–1)
    combined: float            # 0.7 * task + 0.3 * coherence
    task_reasoning: str
    coherence_reasoning: str


async def score_episode(
    task: int,
    episode_log: list[dict],
    final_state: dict,
    llm_client: Any = None,
) -> EpisodeScore:
    """
    Score a completed episode end-to-end.

    Args:
        task:        1, 2, or 3
        episode_log: list of {"hour", "action", "reward", ...} dicts from env
        final_state: dict from env.state()
        llm_client:  your Anthropic / OpenAI client (passed to LLMJudge)

    Returns:
        EpisodeScore with per-dimension breakdown
    """
    trajectory_judges = {1: task1_trajectory_judge,
                         2: task2_trajectory_judge,
                         3: task3_trajectory_judge}
    if task not in trajectory_judges:
        raise ValueError(f"task must be 1, 2, or 3 — got {task}")

    judge = trajectory_judges[task]
    judge.client = llm_client
    trajectory_coherence_judge.client = llm_client

    # Summarise the trajectory — sample key moments to stay within token budget
    sampled = _sample_trajectory(episode_log, n=20)
    interruption_log = _extract_interruptions(episode_log)

    trajectory_str    = json.dumps(sampled, indent=2)
    final_state_str   = json.dumps(final_state, indent=2)
    interruption_str  = json.dumps(interruption_log, indent=2)

    task_result, coherence_result = await asyncio.gather(
        judge(
            action={"summary": "full episode trajectory"},
            observation={"trajectory": trajectory_str,
                         "final_state": final_state_str,
                         "interruption_log": interruption_str},
        ),
        trajectory_coherence_judge(
            action={"summary": "full episode trajectory"},
            observation={"trajectory": trajectory_str,
                         "final_state": final_state_str},
        ),
    )

    task_score = _strict_unit_interval(task_result.get("score", 0.5))
    coherence_score = _strict_unit_interval(coherence_result.get("score", 0.5))
    combined = _strict_unit_interval(0.70 * task_score + 0.30 * coherence_score)

    return EpisodeScore(
        task_score=task_score,
        coherence_score=coherence_score,
        combined=combined,
        task_reasoning=task_result.get("reasoning", ""),
        coherence_reasoning=coherence_result.get("reasoning", ""),
    )


# ---------------------------------------------------------------------------
# Prompt rendering helpers
# ---------------------------------------------------------------------------

def render_step_prompt(judge: LLMJudge, action: dict, observation: dict,
                       **extra_ctx) -> str:
    """Render a step-level judge prompt for inspection / debugging."""
    ctx = {
        "system_context": _SYSTEM_CONTEXT,
        "action":         json.dumps(action, indent=2),
        "observation":    json.dumps(observation, indent=2),
        "hour_of_day":    observation.get("hour_of_day", "unknown"),
        **extra_ctx,
    }
    return judge.prompt_template.format(**ctx)


def render_trajectory_prompt(judge: LLMJudge, episode_log: list[dict],
                              final_state: dict) -> str:
    """Render a trajectory-level judge prompt for inspection / debugging."""
    sampled = _sample_trajectory(episode_log, n=20)
    ctx = {
        "system_context":    _SYSTEM_CONTEXT,
        "trajectory":        json.dumps(sampled, indent=2),
        "final_state":       json.dumps(final_state, indent=2),
        "interruption_log":  json.dumps(_extract_interruptions(episode_log), indent=2),
    }
    return judge.prompt_template.format(**ctx)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sample_trajectory(log: list[dict], n: int = 20) -> list[dict]:
    """
    Sample n evenly-spaced entries from the episode log.
    Always includes the first 3 and last 3 entries so the judge
    sees the agent's opening strategy and closing moves.
    """
    if len(log) <= n:
        return log
    head = log[:3]
    tail = log[-3:]
    middle_n = n - 6
    step = max(1, (len(log) - 6) // middle_n)
    middle = log[3:-3:step][:middle_n]
    return head + middle + tail


def _extract_interruptions(log: list[dict]) -> list[dict]:
    """Extract all spot interruption and restore events from the episode log."""
    events = []
    for entry in log:
        info = entry.get("info", {})
        if info.get("interruptions", 0) > 0:
            events.append({"hour": entry["hour"], "event": "spot_interruption",
                           "penalty": info["interruptions"]})
        action = entry.get("action", {})
        if action.get("action_type") == "restore":
            events.append({"hour": entry["hour"], "event": "restore",
                           "instance": action.get("instance_id")})
    return events
