"""
client.py — Typed client for the Cloud Cost Optimizer environment.

Agents use this to connect to the deployed HF Space without knowing
the HTTP details. Supports both async and sync usage.

Usage (sync):
    with CloudCostClient(base_url="https://user-cloud-cost-optimizer.hf.space").sync() as env:
        env.reset()
        result = env.step(CloudAction(action_type="resize", instance_id="i-001", new_type="m5.large"))
        print(result.reward)

Usage (async):
    async with CloudCostClient(base_url="...") as env:
        await env.reset()
        result = await env.step(CloudAction(...))
"""
from __future__ import annotations
import os
import httpx
from typing import Optional

try:
    from openenv.core.client import EnvClient
    OPENENV_AVAILABLE = True
except ImportError:
    OPENENV_AVAILABLE = False

try:
    from .models import CloudAction, CloudObservation, CloudState, StepResult
except ImportError:
    from models import CloudAction, CloudObservation, CloudState, StepResult

DEFAULT_URL = os.getenv(
    "CLOUD_ENV_URL",
    "https://YOUR_HF_USERNAME-cloud-cost-optimizer.hf.space"
)


class CloudCostClient:
    """
    HTTP client for the Cloud Cost Optimizer OpenEnv environment.
    Wraps the REST API with typed request/response models.
    """

    def __init__(self, base_url: str = DEFAULT_URL, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session_id = "default"
        self._http: Optional[httpx.AsyncClient] = None

    # ── Context manager (async) ──────────────────────────────────────────
    async def __aenter__(self):
        self._http = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)
        return self

    async def __aexit__(self, *_):
        if self._http:
            await self._http.aclose()

    # ── OpenEnv API ──────────────────────────────────────────────────────
    async def reset(self) -> CloudObservation:
        r = await self._http.post(f"/reset?session_id={self._session_id}")
        r.raise_for_status()
        return CloudObservation(**r.json()["observation"])

    async def step(self, action: CloudAction) -> StepResult:
        r = await self._http.post(
            f"/step?session_id={self._session_id}",
            json=action.model_dump(exclude_none=True),
        )
        r.raise_for_status()
        d = r.json()
        return StepResult(
            observation=CloudObservation(**d["observation"]),
            reward=d["reward"],
            done=d["done"],
            info=d.get("info", {}),
        )

    async def state(self) -> CloudState:
        r = await self._http.get(f"/state?session_id={self._session_id}")
        r.raise_for_status()
        return CloudState(**r.json())

    async def health(self) -> bool:
        try:
            r = await self._http.get("/health")
            return r.status_code == 200
        except Exception:
            return False

    # ── Sync wrapper ─────────────────────────────────────────────────────
    def sync(self) -> "_SyncClient":
        return _SyncClient(self.base_url, self.timeout)


class _SyncClient:
    """Synchronous wrapper — use as a context manager: with client.sync() as env:"""

    def __init__(self, base_url: str, timeout: float):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._http: Optional[httpx.Client] = None
        self._session_id = "default"

    def __enter__(self):
        self._http = httpx.Client(base_url=self.base_url, timeout=self.timeout)
        return self

    def __exit__(self, *_):
        if self._http:
            self._http.close()

    def reset(self) -> dict:
        r = self._http.post(f"/reset?session_id={self._session_id}")
        r.raise_for_status()
        return r.json()

    def step(self, action: CloudAction) -> dict:
        r = self._http.post(
            f"/step?session_id={self._session_id}",
            json=action.model_dump(exclude_none=True),
        )
        r.raise_for_status()
        return r.json()

    def state(self) -> dict:
        r = self._http.get(f"/state?session_id={self._session_id}")
        r.raise_for_status()
        return r.json()
