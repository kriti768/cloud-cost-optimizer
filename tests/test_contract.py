from server.app import app
from server.environment import CloudCostEnv


def test_web_route_is_registered():
    assert any(getattr(route, "path", None) == "/web" for route in app.routes)


def test_environment_contract():
    env = CloudCostEnv(seed=42)
    obs = env.reset()
    assert obs.instances
    assert obs.done is False
    result = env.step({"action_type": "noop"})
    assert result.reward is not None
    assert hasattr(result, "instances")
    state = env.state
    state_dict = state.model_dump()
    assert {"hour", "done", "total_cost_usd", "sla_violations"} <= set(state_dict.keys())
