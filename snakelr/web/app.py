from __future__ import annotations

from typing import Optional

import base64
import io

import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from PIL import Image

from snakelr.envs.snake_env import SnakeEnv

app = FastAPI(title="SnakeLR Web")

# Single global env for demo purposes (not multi-user safe)
ENV: Optional[SnakeEnv] = None


class StepRequest(BaseModel):
    action: int


class ResetRequest(BaseModel):
    height: int = 15
    width: int = 15


@app.get("/")
async def index():
    return HTMLResponse(CLIENT_HTML)


@app.post("/reset")
async def reset(req: ResetRequest):
    global ENV
    ENV = SnakeEnv({"board_size": (req.height, req.width)})
    obs, info = ENV.reset()
    return JSONResponse({
        "obs": image_to_base64(obs),
        "info": info,
    })


@app.post("/step")
async def step(payload: StepRequest):
    if ENV is None:
        return JSONResponse({"error": "Env not initialized"}, status_code=400)
    obs, reward, terminated, truncated, info = ENV.step(payload.action)
    done = terminated or truncated
    if done:
        ENV.reset()
    return JSONResponse({
        "obs": image_to_base64(obs),
        "reward": reward,
        "done": done,
        "info": info,
    })


@app.get("/obs")
async def get_obs():
    if ENV is None:
        return JSONResponse({"error": "Env not initialized"}, status_code=400)
    img = ENV.render()
    return JSONResponse({"obs": image_to_base64(img)})


def image_to_base64(img: np.ndarray) -> str:
    pil = Image.fromarray(img)
    buf = io.BytesIO()
    pil = pil.resize((img.shape[1] * 16, img.shape[0] * 16), Image.NEAREST)
    pil.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


CLIENT_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>SnakeLR Web</title>
  <style>
    body { font-family: sans-serif; background: #111; color: #eee; display: flex; flex-direction: column; align-items: center; }
    #board { margin-top: 16px; }
    #controls { margin-top: 8px; }
    button { padding: 8px 12px; margin: 0 4px; }
  </style>
</head>
<body>
  <h1>SnakeLR</h1>
  <div>
    <label>Height <input id="height" type="number" value="15" min="5" max="50"/></label>
    <label>Width <input id="width" type="number" value="15" min="5" max="50"/></label>
    <button id="reset">Reset</button>
  </div>
  <img id="board" width="480" height="480"/>
  <div id="controls">
    <button data-action="0">Up</button>
    <button data-action="1">Down</button>
    <button data-action="2">Left</button>
    <button data-action="3">Right</button>
  </div>
  <p id="status"></p>
  <script>
    const img = document.getElementById('board');
    const status = document.getElementById('status');
    const resetBtn = document.getElementById('reset');

    async function reset() {
      const h = parseInt(document.getElementById('height').value);
      const w = parseInt(document.getElementById('width').value);
      const res = await fetch('/reset', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({height:h, width:w})});
      const data = await res.json();
      img.src = data.obs;
      status.textContent = 'Score: ' + (data.info && data.info.score);
    }

    async function step(action) {
      const res = await fetch('/step', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({action})});
      const data = await res.json();
      if (data.error) { alert(data.error); return; }
      img.src = data.obs;
      status.textContent = `Reward: ${data.reward?.toFixed(2)} | Done: ${data.done}`;
    }

    resetBtn.addEventListener('click', reset);
    document.querySelectorAll('#controls button').forEach(btn => {
      btn.addEventListener('click', () => step(parseInt(btn.dataset.action)));
    });

    // Keyboard control
    document.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowUp') step(0);
      else if (e.key === 'ArrowDown') step(1);
      else if (e.key === 'ArrowLeft') step(2);
      else if (e.key === 'ArrowRight') step(3);
    });

    reset();
  </script>
</body>
</html>
"""
