"""
server/app.py â€” Fixed for openenv-core real API signature.

Real create_fastapi_app signature (from OpenEnv source):
    create_fastapi_app(env_instance, ActionModel, ObservationModel)
    â€” takes an instantiated env, NOT a factory
    â€” NO env_name, state_model kwargs
"""
from __future__ import annotations
import os
import uuid
import json

_server_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir   = os.path.dirname(_server_dir)

try:
    from ..models import CloudAction, CloudObservation, CloudState, StepResult
    from .environment import CloudCostEnv
except ImportError:
    import sys
    sys.path.insert(0, _server_dir)
    sys.path.insert(0, _root_dir)
    from models import CloudAction, CloudObservation, CloudState, StepResult
    from environment import CloudCostEnv

try:
    from openenv.core.env_server import create_fastapi_app
    OPENENV_AVAILABLE = True
except ImportError:
    OPENENV_AVAILABLE = False

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse
import uvicorn

_sessions: dict[str, CloudCostEnv] = {}


def _new_env() -> CloudCostEnv:
    return CloudCostEnv(
        scenario=os.getenv("CLOUD_ENV_SCENARIO", "default"),
        seed=int(os.getenv("CLOUD_ENV_SEED", "42")),
    )


# â”€â”€ Build the app â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if OPENENV_AVAILABLE:
    # Pass the CLASS (callable), not an instance â€” openenv creates instances itself
    app = create_fastapi_app(CloudCostEnv, CloudAction, CloudObservation)

    @app.get("/web", response_class=HTMLResponse)
    async def web_ui():
        return HTMLResponse(_WEB_UI)
else:
    # â”€â”€ Fallback: plain FastAPI â€” identical HTTP contract â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    app = FastAPI(
        title="Cloud Cost Optimizer",
        description="OpenEnv RL environment for cloud cost optimization",
        version="1.0.0",
    )

    @app.post("/reset")
    async def reset(session_id: str = "default"):
        env = _new_env()
        _sessions[session_id] = env
        obs = env.reset()
        return {
            "observation": obs.model_dump(exclude={"reward", "done", "metadata"}),
            "reward": obs.reward,
            "done": obs.done,
            "info": obs.metadata,
        }

    @app.post("/step")
    async def step(action: CloudAction, session_id: str = "default"):
        from fastapi import HTTPException
        env = _sessions.get(session_id)
        if env is None:
            raise HTTPException(400, "Call /reset first")
        obs = env.step(action)
        return {
            "observation": obs.model_dump(exclude={"reward", "done", "metadata"}),
            "reward": obs.reward,
            "done": obs.done,
            "info": obs.metadata,
        }

    @app.get("/state")
    async def state(session_id: str = "default"):
        from fastapi import HTTPException
        env = _sessions.get(session_id)
        if env is None:
            raise HTTPException(400, "Call /reset first")
        return env.state.model_dump()

    # WebSocket for RL training loops
    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket):
        await websocket.accept()
        sid = str(uuid.uuid4())
        env = CloudCostEnv(scenario="default", seed=42)
        _sessions[sid] = env
        try:
            while True:
                data = json.loads(await websocket.receive_text())
                cmd = data.get("command")
                if cmd == "reset":
                    env = CloudCostEnv(scenario="default", seed=42)
                    _sessions[sid] = env
                    obs = env.reset()
                    await websocket.send_text(json.dumps(
                        {
                            "observation": obs.model_dump(exclude={"reward", "done", "metadata"}),
                            "reward": obs.reward,
                            "done": obs.done,
                            "info": obs.metadata,
                        }
                    ))
                elif cmd == "step":
                    action = CloudAction(**data.get("action", {}))
                    obs = env.step(action)
                    await websocket.send_text(json.dumps(
                        {
                            "observation": obs.model_dump(exclude={"reward", "done", "metadata"}),
                            "reward": obs.reward,
                            "done": obs.done,
                            "info": obs.metadata,
                        }
                    ))
                elif cmd == "state":
                    await websocket.send_text(json.dumps(env.state.model_dump()))
        except WebSocketDisconnect:
            _sessions.pop(sid, None)

    @app.get("/web", response_class=HTMLResponse)
    async def web_ui():
        return HTMLResponse(_WEB_UI)


# â”€â”€ Health â€” required by HF checklist â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.get("/")
async def root():
    return JSONResponse({"status": "ok", "env": "cloud-cost-optimizer"})

@app.get("/health")
async def health():
    return JSONResponse({"status": "healthy"})


# â”€â”€ Session-backed UI routes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.post("/ui/reset")
async def ui_reset(session_id: str = "default"):
    env = _new_env()
    _sessions[session_id] = env
    obs = env.reset()
    return {
        "observation": obs.model_dump(exclude={"reward", "done", "metadata"}),
        "reward": obs.reward,
        "done": obs.done,
        "info": obs.metadata,
    }


@app.post("/ui/step")
async def ui_step(action: CloudAction, session_id: str = "default"):
    from fastapi import HTTPException

    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(400, "Call /ui/reset first")
    obs = env.step(action)
    return {
        "observation": obs.model_dump(exclude={"reward", "done", "metadata"}),
        "reward": obs.reward,
        "done": obs.done,
        "info": obs.metadata,
    }


@app.get("/ui/state")
async def ui_state(session_id: str = "default"):
    from fastapi import HTTPException

    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(400, "Call /ui/reset first")
    return env.state.model_dump()


# â”€â”€ MCP tool discovery (RFC 004) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.get("/tools")
async def list_tools():
    return {"tools": [
        {"name": "resize",       "params": {"instance_id": "str", "new_type": "str"}},
        {"name": "schedule",     "params": {"instance_id": "str", "schedule_off": "int", "schedule_on": "int"}},
        {"name": "convert_spot", "params": {"instance_id": "str"}},
        {"name": "reserve",      "params": {"instance_id": "str"}},
        {"name": "restore",      "params": {"instance_id": "str"}},
        {"name": "noop",         "params": {}},
    ]}


# â”€â”€ Embedded web UI â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_WEB_UI = """<!DOCTYPE html>
<html><head>
<title>Cloud Cost Optimizer</title>
<style>
  :root{
    --bg:#f4f7fb;
    --panel:#ffffff;
    --panel-border:#dbe5f0;
    --text:#112031;
    --muted:#5c6b7c;
    --primary:#0b57d0;
    --primary-hover:#0847aa;
    --accent:#dbeafe;
    --output:#f7f9fc;
    --shadow:0 18px 50px rgba(17,32,49,.08);
  }
  *{box-sizing:border-box}
  body{
    margin:0;
    font-family:"Segoe UI",system-ui,sans-serif;
    color:var(--text);
    background:
      radial-gradient(circle at top left,#e3eefc 0,transparent 32%),
      linear-gradient(180deg,#f9fbfe 0%,var(--bg) 100%);
  }
  .shell{max-width:980px;margin:0 auto;padding:38px 20px 56px}
  .hero{
    background:linear-gradient(135deg,#ffffff 0%,#eef5ff 100%);
    border:1px solid var(--panel-border);
    border-radius:24px;
    box-shadow:var(--shadow);
    padding:28px 28px 22px;
    margin-bottom:18px;
  }
  .eyebrow{
    display:inline-block;
    font-size:12px;
    font-weight:700;
    letter-spacing:.08em;
    text-transform:uppercase;
    color:var(--primary);
    background:var(--accent);
    border-radius:999px;
    padding:7px 10px;
    margin-bottom:12px;
  }
  h1{font-size:34px;line-height:1.05;margin:0 0 10px;font-weight:750}
  .sub{
    color:var(--muted);
    font-size:15px;
    max-width:760px;
    margin:0 0 14px;
  }
  .links{
    display:flex;
    gap:10px;
    flex-wrap:wrap;
  }
  .links a{
    text-decoration:none;
    color:var(--text);
    font-size:13px;
    font-weight:600;
    padding:8px 12px;
    border-radius:999px;
    border:1px solid var(--panel-border);
    background:#fff;
  }
  .grid{display:grid;grid-template-columns:1fr;gap:16px}
  .card{
    background:var(--panel);
    border:1px solid var(--panel-border);
    border-radius:22px;
    box-shadow:var(--shadow);
    padding:20px;
  }
  .card h2{
    margin:0 0 6px;
    font-size:18px;
  }
  .hint{
    color:var(--muted);
    font-size:13px;
    margin:0 0 16px;
  }
  .actions{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:14px}
  button{
    background:linear-gradient(135deg,var(--primary) 0%,#2563eb 100%);
    color:#fff;
    border:none;
    border-radius:12px;
    padding:11px 16px;
    cursor:pointer;
    font-size:14px;
    font-weight:700;
    letter-spacing:.01em;
    box-shadow:0 10px 24px rgba(11,87,208,.24);
  }
  button:hover{background:linear-gradient(135deg,var(--primary-hover) 0%,#1d4ed8 100%)}
  select,input{
    border:1px solid #cdd9e5;
    border-radius:12px;
    padding:11px 12px;
    font-size:14px;
    margin-bottom:10px;
    width:100%;
    background:#fff;
    color:var(--text);
  }
  pre{
    background:var(--output);
    border:1px solid #e7edf5;
    border-radius:16px;
    padding:14px;
    font-size:12px;
    line-height:1.45;
    max-height:280px;
    overflow:auto;
    margin:0;
  }
  label{
    font-size:13px;
    font-weight:700;
    display:block;
    margin-bottom:6px;
  }
  .row{display:grid;grid-template-columns:1fr 1fr;gap:14px}
  .tag{
    display:inline-block;
    padding:5px 10px;
    border-radius:999px;
    font-size:11px;
    font-weight:800;
    letter-spacing:.04em;
    text-transform:uppercase;
  }
  .green{background:#dcfce7;color:#166534}
  .amber{background:#fef3c7;color:#92400e}
  @media (max-width: 720px){
    .shell{padding:20px 14px 40px}
    .hero,.card{border-radius:18px}
    h1{font-size:28px}
    .row{grid-template-columns:1fr}
  }
</style></head><body>
<div class="shell">
  <section class="hero">
    <div class="eyebrow">OpenEnv Demo</div>
    <h1>Cloud Cost Optimizer</h1>
    <p class="sub">Explore the environment interactively, inspect state transitions, and validate reset or step behavior before deployment.</p>
    <div class="links">
      <a href="/docs">API docs</a>
      <a href="/tools">MCP tools</a>
      <a href="/">Health check</a>
    </div>
  </section>

  <div class="grid">
    <div class="card">
      <h2>Episode Control</h2>
      <p class="hint">Start a fresh session and inspect the initial observation payload.</p>
      <div class="actions">
        <button onclick="doReset()">Reset Episode</button>
        <span id="sbadge"></span>
      </div>
      <pre id="reset-out">Click Reset to start a new episode</pre>
    </div>

    <div class="card">
      <h2>Action Runner</h2>
      <p class="hint">Choose an action, target instance, and optional parameters, then execute one environment step.</p>
      <div class="row">
        <div>
          <label>Action Type</label>
          <select id="atype" onchange="toggleFields()">
            <option>resize</option><option>schedule</option>
            <option>convert_spot</option><option>reserve</option>
            <option>restore</option><option>noop</option>
          </select>
        </div>
        <div>
          <label>Instance ID</label>
          <input id="iid" value="i-001" placeholder="e.g. i-001">
        </div>
      </div>
      <div id="resize-fields" class="row">
        <div>
          <label>New Type (resize only)</label>
          <select id="ntype">
            <option>m5.large</option><option>m5.xlarge</option><option>m5.2xlarge</option>
            <option>t3.medium</option><option>c5.xlarge</option><option>r5.large</option>
          </select>
        </div>
      </div>
      <div class="actions">
        <button onclick="doStep()">Take Step</button>
      </div>
      <pre id="step-out">Reset first, then take steps</pre>
    </div>

    <div class="card">
      <h2>State Snapshot</h2>
      <p class="hint">Inspect the current session state without advancing the environment.</p>
      <div class="actions">
        <button onclick="doState()">Get State</button>
      </div>
      <pre id="state-out">State will appear here</pre>
    </div>
  </div>
</div>

<script>
const BASE = window.location.origin;
const SID  = "web-" + Math.random().toString(36).slice(2,7);

function badge(t,c){document.getElementById("sbadge").innerHTML=`<span class="tag ${c}">${t}</span>`;}

async function doReset(){
  const r = await fetch(`${BASE}/ui/reset?session_id=${SID}`,{method:"POST"});
  const d = await r.json();
  document.getElementById("reset-out").textContent = JSON.stringify(d,null,2);
  badge("Running","green");
}
async function doStep(){
  const at = document.getElementById("atype").value;
  const body = {action_type: at};
  const iid = document.getElementById("iid").value.trim();
  if(iid && at!=="noop") body.instance_id = iid;
  if(at==="resize") body.new_type = document.getElementById("ntype").value;
  if(at==="schedule"){body.schedule_off=22;body.schedule_on=8;}
  const r = await fetch(`${BASE}/ui/step?session_id=${SID}`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
  const d = await r.json();
  document.getElementById("step-out").textContent = JSON.stringify(d,null,2);
  if(d.done) badge("Done","amber");
}
async function doState(){
  const r = await fetch(`${BASE}/ui/state?session_id=${SID}`);
  const d = await r.json();
  document.getElementById("state-out").textContent = JSON.stringify(d,null,2);
}
function toggleFields(){
  document.getElementById("resize-fields").style.display =
    document.getElementById("atype").value==="resize"?"grid":"none";
}
toggleFields();
</script></body></html>"""


def main() -> None:
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()

