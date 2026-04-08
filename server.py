from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import os
from environment import BioCircuitEnv

app = FastAPI(title="SynBio-RL OpenEnv", version="1.0.0")

env = BioCircuitEnv()
current_task_idx = 0

class Action(BaseModel):
    type: str = "place"
    part: Optional[str] = None

class ResetRequest(BaseModel):
    task_id: Optional[int] = 0

class Observation(BaseModel):
    task: str
    circuit: List[str]
    available_parts: List[str]
    target: float
    fluorescence: float
    math_reward: float
    steps: int
    hint: str

class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict = {}

class ResetResponse(BaseModel):
    observation: Observation

@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest = ResetRequest()):
    global current_task_idx
    current_task_idx = req.task_id or 0
    state = env.reset(current_task_idx)
    return ResetResponse(observation=Observation(**state))

@app.post("/step", response_model=StepResponse)
def step(action: Action):
    state, reward, done = env.step(action.dict())
    norm_reward = min(max(round(reward / 10.0, 4), 0.0), 1.0)
    return StepResponse(
        observation=Observation(**state),
        reward=norm_reward,
        done=done,
        info={}
    )

@app.get("/state")
def get_state():
    return env.state()

@app.get("/tasks")
def list_tasks():
    result = []
    for i, t in enumerate(env.tasks):
        result.append({
            "id": t.get("id", f"task_{i+1}"),
            "name": t.get("name", t.get("task_name", f"Task {i+1}")),
            "difficulty": "easy" if i < 5 else "medium" if i < 10 else "hard",
            "target_output": t.get("target_output", 0.0)
        })
    return {"tasks": result, "total": len(result)}

@app.get("/health")
def health():
    return {"status": "ok", "env": "SynBio-RL"}

@app.get("/")
def root():
    return {"name": "SynBio-RL", "version": "1.0.0", "status": "running"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
