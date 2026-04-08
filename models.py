"""
models.py — Typed Pydantic models for the Cloud Cost Optimizer environment.
Required by OpenEnv spec: all Action / Observation / State must be typed models,
not plain dicts. The framework uses these for JSON serialization, MCP tool schema
generation, and the /web UI form fields.
"""

from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field, ConfigDict

try:
    from openenv.core.env_server.types import Action as OpenEnvAction
    from openenv.core.env_server.types import Observation as OpenEnvObservation
    from openenv.core.env_server.types import State as OpenEnvState
except ImportError:
    OpenEnvAction = BaseModel
    OpenEnvObservation = BaseModel
    OpenEnvState = BaseModel


# ---------------------------------------------------------------------------
# Action model
# ---------------------------------------------------------------------------

class CloudAction(OpenEnvAction):
    """
    One agent action. action_type determines which fields are used.
    Exposed as MCP tools in server/app.py — each action_type becomes one tool.
    """
    action_type: Literal[
        "resize",        # change instance type
        "schedule",      # set scale-down / scale-up hours
        "convert_spot",  # switch to spot pricing (cheaper, interruptible)
        "reserve",       # commit to 1-year reserved pricing (40% off)
        "restore",       # bring back an interrupted spot instance
        "noop",          # do nothing this step
    ] = Field(description="The type of action to perform")

    instance_id: Optional[str] = Field(
        default=None,
        description="Target instance ID, e.g. 'i-001'. Required for all actions except noop.",
    )
    new_type: Optional[str] = Field(
        default=None,
        description="New instance type for resize actions, e.g. 'm5.large'. See INSTANCE_TYPES.",
    )
    schedule_off: Optional[int] = Field(
        default=None,
        ge=0, le=23,
        description="Hour of day (0-23) to scale the instance down. Used with action_type='schedule'.",
    )
    schedule_on: Optional[int] = Field(
        default=None,
        ge=0, le=23,
        description="Hour of day (0-23) to scale the instance back up. Used with action_type='schedule'.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"action_type": "resize",       "instance_id": "i-001", "new_type": "m5.large"},
                {"action_type": "schedule",     "instance_id": "i-006", "schedule_off": 22, "schedule_on": 8},
                {"action_type": "convert_spot", "instance_id": "i-011"},
                {"action_type": "reserve",      "instance_id": "i-015"},
                {"action_type": "restore",      "instance_id": "i-012"},
                {"action_type": "noop"},
            ]
        }
    )


# ---------------------------------------------------------------------------
# Per-instance state (nested inside Observation)
# ---------------------------------------------------------------------------

class InstanceState(BaseModel):
    id: str                = Field(description="Instance identifier, e.g. 'i-001'")
    instance_type: str     = Field(description="Current AWS instance type, e.g. 'm5.xlarge'")
    workload: str          = Field(description="Workload category: web_api / data_pipeline / batch_job / stateless / database")
    pricing: str           = Field(description="Current pricing model: on_demand / spot / reserved")
    vcpu: int              = Field(description="Number of virtual CPUs")
    memory_gb: int         = Field(description="Memory in GB")
    hourly_cost: float     = Field(description="Current hourly cost in USD after pricing model applied")
    current_cpu_pct: float = Field(description="Current CPU utilization percentage (0-100)")
    current_mem_pct: float = Field(description="Current memory utilization percentage (0-100)")
    is_interrupted: bool   = Field(description="True if this spot instance has been interrupted")
    sla_breaches: int      = Field(description="Cumulative SLA breach count for this instance this episode")
    headroom_factor: float = Field(description="Remaining capacity headroom (1.0=full, shrinks after downsizing)")


# ---------------------------------------------------------------------------
# Observation model — returned by reset() and step()
# ---------------------------------------------------------------------------

class CloudObservation(OpenEnvObservation):
    """
    Full environment snapshot returned after every step and reset.
    Contains everything an agent needs to make its next decision.
    """
    hour: int = Field(description="Simulated hour elapsed in the episode (0-72)")
    hour_of_day: int = Field(description="Hour of day in the simulation (0-23, 14=2pm peak)")
    instances: list[InstanceState] = Field(description="State of all 20 instances in the fleet")
    total_cost_so_far: float = Field(description="Cumulative cost in USD since episode start")
    sla_violations: int = Field(description="Total SLA breach events across all instances this episode")
    savings_vs_baseline: float = Field(description="USD saved vs doing nothing (on-demand, no resizing)")
    scheduled_actions: dict[str, dict] = Field(
        default_factory=dict,
        description="Active schedule rules: {instance_id: {off: hour, on: hour}}",
    )


# ---------------------------------------------------------------------------
# State model — returned by state()
# ---------------------------------------------------------------------------

class CloudState(OpenEnvState):
    """
    High-level episode state snapshot. Returned by state().
    Used by graders to compute task scores without re-running the full observation.
    """
    hour: int                = Field(description="Current simulated hour (0-72)")
    done: bool               = Field(description="True when the 72-hour episode is complete")
    total_cost_usd: float    = Field(description="Cumulative cost incurred so far this episode")
    baseline_cost_usd: float = Field(description="What cost would have been with no agent actions")
    total_savings_usd: float = Field(description="Actual USD saved vs baseline (always >= 0)")
    savings_pct: float       = Field(description="Percentage cost reduction vs baseline")
    sla_violations: int      = Field(description="Total SLA breach events this episode")
    active_instances: int    = Field(description="Number of instances currently running (not interrupted)")
    spot_instances: int      = Field(description="Number of instances on spot pricing")
    reserved_instances: int  = Field(description="Number of instances on reserved pricing")


# ---------------------------------------------------------------------------
# Step result — wraps observation + reward + done signal
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    """
    Full result returned by step(). Matches OpenEnv standard step result shape.
    """
    observation: CloudObservation
    reward: float  = Field(description="Reward signal for this step (USD savings minus penalties)")
    done: bool     = Field(description="True when the episode has ended")
    info: dict     = Field(default_factory=dict, description="Debug info: hourly_cost, new_sla_violations, etc.")
