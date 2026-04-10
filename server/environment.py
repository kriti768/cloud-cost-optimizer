"""
Cloud Cost Optimizer — OpenEnv Environment
Supports long-running episodes (72 simulated hours) with multiple
viable trajectories through the environment.

Key design decisions for trajectory diversity:
  1. Time-aware simulation: actions have different effects at different hours
  2. Stochastic utilization: CPU/memory fluctuate with realistic noise
  3. Action interdependence: resizing now affects headroom for future spikes
  4. Multiple action types: resize, schedule, spot-convert, reserve — all valid paths
"""

import random
import math
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

try:
    from ..models import CloudObservation, CloudState, StepResult
except ImportError:
    import os
    import sys

    _server_dir = os.path.dirname(os.path.abspath(__file__))
    _root_dir = os.path.dirname(_server_dir)
    sys.path.insert(0, _root_dir)
    from models import CloudObservation, CloudState, StepResult

try:
    from openenv.core.env_server import Environment as OpenEnvBase
    _BASE = OpenEnvBase
except ImportError:
    _BASE = object



# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HOURS_PER_EPISODE = 72          # long-running: 3 simulated days
SLA_CPU_THRESHOLD = 90.0        # CPU % above which SLA is breached
SLA_PENALTY_PER_BREACH = 0.15   # score penalty per SLA violation event
SPOT_INTERRUPTION_PROB = 0.04   # 4% chance per hour a spot instance is interrupted

INSTANCE_TYPES = {
    # name            : (vcpu, memory_gb, hourly_cost_usd)
    "t3.small"        : (2,    2,    0.021),
    "t3.medium"       : (2,    4,    0.042),
    "t3.large"        : (2,    8,    0.083),
    "m5.large"        : (2,    8,    0.096),
    "m5.xlarge"       : (4,   16,    0.192),
    "m5.2xlarge"      : (8,   32,    0.384),
    "m5.4xlarge"      : (16,  64,    0.768),
    "m5.8xlarge"      : (32, 128,    1.536),
    "c5.xlarge"       : (4,    8,    0.170),
    "c5.2xlarge"      : (8,   16,    0.340),
    "r5.large"        : (2,   16,    0.126),
    "r5.xlarge"       : (4,   32,    0.252),
}

SPOT_DISCOUNT = 0.68            # spot instances cost 68% less than on-demand
RESERVED_DISCOUNT = 0.40        # 1-year reserved instances cost 40% less


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class PricingModel(Enum):
    ON_DEMAND = "on_demand"
    SPOT      = "spot"
    RESERVED  = "reserved"


class WorkloadType(Enum):
    WEB_API      = "web_api"       # traffic follows business hours
    DATA_PIPELINE= "data_pipeline" # runs at night
    BATCH_JOB    = "batch_job"     # bursty, unpredictable
    STATELESS    = "stateless"     # safe for spot
    DATABASE     = "database"      # steady, never interrupt


@dataclass
class Instance:
    id: str
    instance_type: str
    workload: WorkloadType
    pricing: PricingModel = PricingModel.ON_DEMAND

    # Baseline utilization (% of capacity this workload actually needs)
    base_cpu_need: float   = 10.0
    base_mem_need: float   = 20.0

    # Runtime state (updated each simulated hour)
    current_cpu: float     = 0.0
    current_mem: float     = 0.0
    is_interrupted: bool   = False   # spot interruption flag
    sla_breaches: int      = 0
    hours_active: int      = 0

    # Headroom shrinks when instance is downsized — key for action interdependence
    headroom_factor: float = 1.0    # 1.0 = full headroom, 0.6 = tight after downsize

    @property
    def specs(self):
        return INSTANCE_TYPES[self.instance_type]

    @property
    def vcpu(self):
        return self.specs[0]

    @property
    def memory_gb(self):
        return self.specs[1]

    @property
    def hourly_cost(self) -> float:
        base = self.specs[2]
        if self.pricing == PricingModel.SPOT:
            return base * (1 - SPOT_DISCOUNT)
        if self.pricing == PricingModel.RESERVED:
            return base * (1 - RESERVED_DISCOUNT)
        return base

    def to_dict(self) -> dict:
        return {
            "id"              : self.id,
            "instance_type"   : self.instance_type,
            "workload"        : self.workload.value,
            "pricing"         : self.pricing.value,
            "vcpu"            : self.vcpu,
            "memory_gb"       : self.memory_gb,
            "hourly_cost"     : round(self.hourly_cost, 4),
            "current_cpu_pct" : round(self.current_cpu, 1),
            "current_mem_pct" : round(self.current_mem, 1),
            "is_interrupted"  : self.is_interrupted,
            "sla_breaches"    : self.sla_breaches,
            "headroom_factor" : round(self.headroom_factor, 2),
        }


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class CloudCostEnv(_BASE):
    """
    OpenEnv-compatible environment for cloud cost optimization.

    Action space (dict):
        action_type : str  — one of: "resize", "schedule", "convert_spot",
                                      "reserve", "restore", "noop"
        instance_id : str  — target instance ID
        new_type    : str  — (resize only) new instance type string
        schedule_off: int  — (schedule only) hour to scale down (0-23)
        schedule_on : int  — (schedule only) hour to scale back up (0-23)

    Observation space (dict):
        hour            : int        — current simulated hour (0-71)
        instances       : list[dict] — all instance states
        total_cost_so_far : float    — cumulative cost in USD
        sla_violations  : int        — total SLA breach events this episode
        savings_vs_baseline : float  — USD saved vs doing nothing

    Reward:
        Each step returns a float reward signal:
          + cost saved this hour vs baseline
          - SLA_PENALTY_PER_BREACH for each new SLA violation
          - small penalty for spot interruptions that weren't handled
        This gives dense, continuous reward on every step (not just at episode end).
    """

    metadata = {"name": "cloud-cost-optimizer", "version": "1.0"}

    def __init__(self, scenario: str = "default", seed: Optional[int] = None):
        try:
            super().__init__()
        except Exception:
            pass
        self.scenario = scenario
        self.seed = seed
        self._rng = random.Random(seed)
        self.instances: list[Instance] = []
        self.hour = 0
        self.done = False
        self._baseline_hourly_cost = 0.0
        self._total_cost = 0.0
        self._baseline_total = 0.0
        self._sla_violations = 0
        self._scheduled_actions: dict = {}   # instance_id -> (off_hour, on_hour)
        self._episode_log: list[dict] = []

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> CloudObservation:
        """Start a new episode. Returns initial observation."""
        if seed is not None:
            self.seed = seed
        self._rng = random.Random(self.seed)
        self.hour = 0
        self.done = False
        self._total_cost = 0.0
        self._sla_violations = 0
        self._scheduled_actions = {}
        self._episode_log = []

        self.instances = self._build_scenario(self.scenario)
        self._baseline_hourly_cost = sum(i.hourly_cost for i in self.instances)
        self._baseline_total = self._baseline_hourly_cost * HOURS_PER_EPISODE

        self._tick_utilization()  # set initial utilization values
        obs = self._observation()
        obs.reward = 0.0
        obs.done = False
        obs.metadata = {"episode_id": episode_id} if episode_id else {}
        return obs

    def step(self, action, timeout_s: Optional[float] = None, **kwargs) -> CloudObservation:
        """
        Apply one action, advance time by one hour, return (obs, reward, done, info).
        Compatible with both gym-style tuple returns and OpenEnv dict returns.
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        reward = 0.0
        info = {}

        # 1. Apply the agent's action
        action_data = action.model_dump(exclude_none=True) if hasattr(action, "model_dump") else action
        action_result = self._apply_action(action_data)
        info["action_result"] = action_result

        # 2. Advance time by one hour
        self.hour += 1
        self._apply_scheduled_actions()
        self._tick_utilization()

        # 3. Check for spot interruptions
        interruption_penalties = self._check_spot_interruptions()
        reward -= interruption_penalties
        info["interruptions"] = interruption_penalties

        # 4. Check SLA violations
        new_violations = self._check_sla()
        self._sla_violations += new_violations
        reward -= new_violations * SLA_PENALTY_PER_BREACH
        info["new_sla_violations"] = new_violations

        # 5. Calculate cost reward for this hour
        current_hourly = sum(
            i.hourly_cost for i in self.instances if not i.is_interrupted
        )
        hourly_savings = self._baseline_hourly_cost - current_hourly
        reward += hourly_savings
        self._total_cost += current_hourly

        normalized_reward = self._normalize_reward(reward)

        info["hourly_savings"] = round(hourly_savings, 4)
        info["current_hourly_cost"] = round(current_hourly, 4)
        info["hour"] = self.hour
        info["raw_reward"] = round(reward, 4)
        info["normalized_reward"] = normalized_reward

        self._episode_log.append({
            "hour": self.hour,
            "action": action_data,
            "reward": normalized_reward,
            "raw_reward": round(reward, 4),
            "sla_violations": self._sla_violations,
            "hourly_cost": round(current_hourly, 4),
        })

        if self.hour >= HOURS_PER_EPISODE:
            self.done = True

        obs = self._observation()
        obs.reward = normalized_reward
        obs.done = self.done
        obs.metadata = info
        return obs

    @property
    def state(self) -> CloudState:
        """Return current environment state snapshot (no time advance)."""
        total_savings = self._baseline_total - self._total_cost
        savings_pct = (total_savings / self._baseline_total * 100
                       if self._baseline_total > 0 else 0.0)
        return CloudState(
            hour=self.hour,
            done=self.done,
            total_cost_usd=round(self._total_cost, 2),
            baseline_cost_usd=round(self._baseline_hourly_cost * self.hour, 2),
            total_savings_usd=round(max(0, total_savings), 2),
            savings_pct=round(savings_pct, 1),
            sla_violations=self._sla_violations,
            active_instances=len([i for i in self.instances if not i.is_interrupted]),
            spot_instances=len([i for i in self.instances if i.pricing == PricingModel.SPOT]),
            reserved_instances=len([i for i in self.instances if i.pricing == PricingModel.RESERVED]),
        )

    # ------------------------------------------------------------------
    # Scenario builder — generates the starting fleet
    # ------------------------------------------------------------------

    def _build_scenario(self, scenario: str) -> list[Instance]:
        """
        Build a fleet of instances for the given scenario.
        'default' gives the full mixed fleet used for all three tasks.
        Overprovisioned instances are intentionally large relative to workload need.
        """
        fleet = []

        # --- Clearly overprovisioned (Task 1 targets) ---
        # These have very low base_cpu_need relative to their instance size.
        # Easy for an agent to spot — CPU never rises above ~12%.
        overprov = [
            ("i-001", "m5.4xlarge", WorkloadType.WEB_API,       5.0,  8.0),
            ("i-002", "m5.2xlarge", WorkloadType.STATELESS,      7.0, 10.0),
            ("i-003", "m5.4xlarge", WorkloadType.STATELESS,      4.0,  6.0),
            ("i-004", "c5.2xlarge", WorkloadType.BATCH_JOB,      6.0, 12.0),
            ("i-005", "r5.xlarge",  WorkloadType.DATA_PIPELINE,  8.0,  9.0),
        ]
        for iid, itype, wtype, cpu_need, mem_need in overprov:
            fleet.append(Instance(
                id=iid, instance_type=itype, workload=wtype,
                base_cpu_need=cpu_need, base_mem_need=mem_need,
            ))

        # --- Schedule-sensitive (Task 2 targets) ---
        # These are correctly sized but follow predictable traffic patterns.
        # Can be safely scaled down during off-hours.
        scheduled = [
            ("i-006", "m5.xlarge",  WorkloadType.WEB_API,       55.0, 40.0),
            ("i-007", "m5.2xlarge", WorkloadType.WEB_API,       60.0, 45.0),
            ("i-008", "m5.xlarge",  WorkloadType.WEB_API,       50.0, 38.0),
            ("i-009", "c5.xlarge",  WorkloadType.DATA_PIPELINE, 20.0, 30.0),
            ("i-010", "c5.xlarge",  WorkloadType.DATA_PIPELINE, 18.0, 28.0),
        ]
        for iid, itype, wtype, cpu_need, mem_need in scheduled:
            fleet.append(Instance(
                id=iid, instance_type=itype, workload=wtype,
                base_cpu_need=cpu_need, base_mem_need=mem_need,
            ))

        # --- Spot candidates (Task 3 targets) ---
        # Stateless, fault-tolerant workloads. Good spot candidates.
        spot_candidates = [
            ("i-011", "m5.2xlarge", WorkloadType.STATELESS,     45.0, 35.0),
            ("i-012", "m5.xlarge",  WorkloadType.STATELESS,     40.0, 30.0),
            ("i-013", "c5.2xlarge", WorkloadType.STATELESS,     50.0, 40.0),
            ("i-014", "m5.xlarge",  WorkloadType.BATCH_JOB,     35.0, 25.0),
        ]
        for iid, itype, wtype, cpu_need, mem_need in spot_candidates:
            fleet.append(Instance(
                id=iid, instance_type=itype, workload=wtype,
                base_cpu_need=cpu_need, base_mem_need=mem_need,
            ))

        # --- Stable anchors (should NOT be touched) ---
        # Databases and high-load services. Downsizing these causes SLA breaches.
        stable = [
            ("i-015", "r5.xlarge",  WorkloadType.DATABASE,      70.0, 75.0),
            ("i-016", "r5.xlarge",  WorkloadType.DATABASE,      68.0, 72.0),
            ("i-017", "m5.2xlarge", WorkloadType.WEB_API,       78.0, 60.0),
            ("i-018", "m5.2xlarge", WorkloadType.WEB_API,       75.0, 58.0),
            ("i-019", "c5.2xlarge", WorkloadType.BATCH_JOB,     72.0, 55.0),
            ("i-020", "m5.xlarge",  WorkloadType.DATA_PIPELINE, 65.0, 50.0),
        ]
        for iid, itype, wtype, cpu_need, mem_need in stable:
            fleet.append(Instance(
                id=iid, instance_type=itype, workload=wtype,
                base_cpu_need=cpu_need, base_mem_need=mem_need,
            ))

        return fleet

    # ------------------------------------------------------------------
    # Action handlers — this is where multiple trajectories are born
    # ------------------------------------------------------------------

    def _apply_action(self, action: dict) -> str:
        atype = action.get("action_type", "noop")

        if atype == "noop":
            return "no action taken"

        iid = action.get("instance_id")
        instance = self._get_instance(iid)
        if instance is None:
            return f"unknown instance: {iid}"

        if atype == "resize":
            return self._action_resize(instance, action.get("new_type", ""))

        if atype == "schedule":
            return self._action_schedule(
                instance,
                action.get("schedule_off", 22),
                action.get("schedule_on", 8),
            )

        if atype == "convert_spot":
            return self._action_convert_spot(instance)

        if atype == "reserve":
            return self._action_reserve(instance)

        if atype == "restore":
            return self._action_restore(instance)

        return f"unknown action_type: {atype}"

    def _action_resize(self, instance: Instance, new_type: str) -> str:
        """
        Resize instance to a different type.

        KEY MECHANIC FOR TRAJECTORY DIVERSITY:
        Downsizing reduces headroom_factor, making the instance more
        vulnerable to future traffic spikes. This creates a real tradeoff:
        aggressive early downsizing saves more money but increases SLA risk
        later when traffic patterns shift. Agents that downsize carefully
        and monitor utilization will follow a different trajectory than
        agents that resize everything immediately.
        """
        if new_type not in INSTANCE_TYPES:
            return f"invalid instance type: {new_type}"

        old_type = instance.instance_type
        old_specs = INSTANCE_TYPES[old_type]
        new_specs = INSTANCE_TYPES[new_type]

        # Calculate headroom change
        cpu_ratio = new_specs[0] / old_specs[0]   # ratio of new vcpu to old
        mem_ratio = new_specs[1] / old_specs[1]

        # Headroom shrinks proportionally when downsizing
        instance.headroom_factor = min(
            instance.headroom_factor,
            min(cpu_ratio, mem_ratio)
        )
        instance.instance_type = new_type
        return f"resized {instance.id}: {old_type} -> {new_type} (headroom now {instance.headroom_factor:.2f})"

    def _action_schedule(self, instance: Instance, off_hour: int, on_hour: int) -> str:
        """
        Schedule an instance to scale down at off_hour and back up at on_hour.

        This trajectory option rewards agents that observe workload patterns
        and time their actions to off-peak windows, rather than resizing
        immediately and risking SLA during traffic peaks.
        """
        self._scheduled_actions[instance.id] = (off_hour % 24, on_hour % 24)
        return (f"scheduled {instance.id}: down at hour {off_hour % 24}, "
                f"up at hour {on_hour % 24}")

    def _action_convert_spot(self, instance: Instance) -> str:
        """
        Convert instance to spot pricing — larger discount, interruption risk.
        Only safe for stateless/batch workloads. Databases should never be spot.
        This enables a completely different cost-saving trajectory than resizing.
        """
        if instance.workload == WorkloadType.DATABASE:
            return f"refused: databases cannot run on spot (would guarantee SLA breach)"
        instance.pricing = PricingModel.SPOT
        return f"converted {instance.id} to spot pricing (68% discount)"

    def _action_reserve(self, instance: Instance) -> str:
        """
        Commit to reserved pricing — 40% discount, no interruption risk.
        Best for stable, predictable workloads. Creates a third trajectory
        where the agent identifies stable instances and pre-commits them.
        """
        instance.pricing = PricingModel.RESERVED
        return f"reserved {instance.id} for 1-year term (40% discount)"

    def _action_restore(self, instance: Instance) -> str:
        """Restore an interrupted spot instance to on-demand."""
        if not instance.is_interrupted:
            return f"{instance.id} is not interrupted — nothing to restore"
        instance.is_interrupted = False
        instance.pricing = PricingModel.ON_DEMAND
        return f"restored {instance.id} to on-demand after interruption"

    # ------------------------------------------------------------------
    # Time simulation — where stochasticity creates trajectory divergence
    # ------------------------------------------------------------------

    def _tick_utilization(self):
        """
        Advance utilization for all instances by one simulated hour.

        Utilization depends on:
        - The hour of day (business hours pattern for WEB_API)
        - Workload type (DATA_PIPELINE peaks at night, etc.)
        - Random noise (±15% realistic fluctuation)
        - Headroom factor (tight instances spike harder under load)

        This stochasticity means two agents making identical early decisions
        will face different utilization values by hour 20+, ensuring that
        trajectories genuinely diverge rather than converging on one path.
        """
        hour_of_day = self.hour % 24

        # Traffic multiplier by hour — business hours for web workloads
        web_multiplier = self._business_hours_curve(hour_of_day)

        for inst in self.instances:
            if inst.is_interrupted:
                inst.current_cpu = 0.0
                inst.current_mem = 0.0
                continue

            # Base utilization for this workload type
            if inst.workload == WorkloadType.WEB_API:
                load_mult = web_multiplier
            elif inst.workload == WorkloadType.DATA_PIPELINE:
                # Peaks between 2am–6am
                load_mult = 1.8 if 2 <= hour_of_day <= 6 else 0.3
            elif inst.workload == WorkloadType.BATCH_JOB:
                # Bursty — random spikes
                load_mult = self._rng.uniform(0.2, 2.2)
            elif inst.workload == WorkloadType.DATABASE:
                # Steady with small variance
                load_mult = self._rng.uniform(0.85, 1.15)
            else:  # STATELESS
                load_mult = self._rng.uniform(0.4, 1.2)

            # Apply noise and headroom squeeze
            noise_cpu = self._rng.gauss(0, 0.12)
            noise_mem = self._rng.gauss(0, 0.08)

            # Tight headroom amplifies spikes — key interdependence mechanic
            spike_factor = 1.0 + (1.0 - inst.headroom_factor) * 0.5

            raw_cpu = inst.base_cpu_need * load_mult * spike_factor + noise_cpu * 20
            raw_mem = inst.base_mem_need * load_mult + noise_mem * 15

            inst.current_cpu = max(1.0, min(100.0, raw_cpu))
            inst.current_mem = max(1.0, min(100.0, raw_mem))
            inst.hours_active += 1

    def _business_hours_curve(self, hour: int) -> float:
        """Smooth sine-based traffic curve peaking at 2pm, trough at 4am."""
        # Shift so peak is at hour 14, trough at hour 4
        angle = (hour - 14) * math.pi / 12
        raw = 0.5 - 0.5 * math.cos(angle + math.pi)
        # Scale to 0.15 (4am quiet) – 1.6 (2pm peak)
        return 0.15 + raw * 1.45

    def _apply_scheduled_actions(self):
        """
        Execute any scheduled scale events for the current hour.
        Scale-down swaps to a smaller instance type; scale-up restores original.
        """
        hour_of_day = self.hour % 24
        for iid, (off_h, on_h) in list(self._scheduled_actions.items()):
            inst = self._get_instance(iid)
            if inst is None:
                continue
            if hour_of_day == off_h:
                # Downsize by one tier for the off-peak window
                smaller = self._next_smaller_type(inst.instance_type)
                if smaller:
                    inst.instance_type = smaller
                    inst.headroom_factor = min(inst.headroom_factor, 0.7)
            elif hour_of_day == on_h:
                # Restore to next larger size
                larger = self._next_larger_type(inst.instance_type)
                if larger:
                    inst.instance_type = larger
                    inst.headroom_factor = min(1.0, inst.headroom_factor + 0.3)

    def _check_spot_interruptions(self) -> float:
        """
        Randomly interrupt spot instances. Returns penalty for unhandled interruptions.
        Agents on the spot-migration trajectory must actively monitor and restore.
        """
        penalty = 0.0
        for inst in self.instances:
            if inst.pricing == PricingModel.SPOT and not inst.is_interrupted:
                if self._rng.random() < SPOT_INTERRUPTION_PROB:
                    inst.is_interrupted = True
                    # SLA breach only for workloads that shouldn't be interrupted
                    if inst.workload not in (WorkloadType.STATELESS, WorkloadType.BATCH_JOB):
                        penalty += SLA_PENALTY_PER_BREACH
                        self._sla_violations += 1
        return penalty

    def _check_sla(self) -> int:
        """
        Count new SLA violations (instances where CPU > SLA_CPU_THRESHOLD).
        Returns the number of new violations this hour.
        """
        new_violations = 0
        for inst in self.instances:
            if inst.is_interrupted:
                continue
            if inst.current_cpu > SLA_CPU_THRESHOLD:
                inst.sla_breaches += 1
                new_violations += 1
        return new_violations

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _observation(self) -> CloudObservation:
        return CloudObservation(
            hour=self.hour,
            hour_of_day=self.hour % 24,
            instances=[i.to_dict() for i in self.instances],
            total_cost_so_far=round(self._total_cost, 2),
            sla_violations=self._sla_violations,
            savings_vs_baseline=round(
                max(0, self._baseline_hourly_cost * self.hour - self._total_cost), 2
            ),
            scheduled_actions={
                iid: {"off": off, "on": on}
                for iid, (off, on) in self._scheduled_actions.items()
            },
        )

    def _normalize_reward(self, raw_reward: float) -> float:
        """
        Map the dense raw reward into the strict open interval (0, 1).

        This preserves ordering and partial progress while satisfying validators
        that require reward-like scores to avoid exact 0.0 or 1.0 values.
        """
        squashed = 1.0 / (1.0 + math.exp(-raw_reward))
        return min(0.9999, max(0.0001, round(squashed, 4)))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_instance(self, iid: str) -> Optional[Instance]:
        return next((i for i in self.instances if i.id == iid), None)

    def _next_smaller_type(self, itype: str) -> Optional[str]:
        """Return the next cheaper instance type in the same family."""
        types_by_cost = sorted(INSTANCE_TYPES.keys(), key=lambda t: INSTANCE_TYPES[t][2])
        idx = types_by_cost.index(itype) if itype in types_by_cost else -1
        return types_by_cost[idx - 1] if idx > 0 else None

    def _next_larger_type(self, itype: str) -> Optional[str]:
        """Return the next more expensive instance type in the same family."""
        types_by_cost = sorted(INSTANCE_TYPES.keys(), key=lambda t: INSTANCE_TYPES[t][2])
        idx = types_by_cost.index(itype) if itype in types_by_cost else -1
        return types_by_cost[idx + 1] if idx < len(types_by_cost) - 1 else None

    def episode_summary(self) -> dict:
        """Full episode stats — useful for logging and debugging."""
        s = self.state()
        s["episode_log"] = self._episode_log
        s["trajectory_diversity_score"] = self._compute_trajectory_diversity()
        return s

    def _compute_trajectory_diversity(self) -> float:
        """
        Rough measure of how many distinct action types were used.
        High diversity = agent explored multiple routes. Useful for analysis.
        """
        action_types_used = set(
            log["action"].get("action_type", "noop")
            for log in self._episode_log
        )
        return len(action_types_used) / 6.0   # 6 possible action types
