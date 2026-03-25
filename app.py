import logging
from fastapi import FastAPI, HTTPException
from open_env import DataOpsEnv, Action, Observation, Reward, SimpleAgent
from pydantic import BaseModel
from typing import Dict, Any, List

# Configure application-wide logging so endpoint behavior is visible in local runs.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app metadata is shown in OpenAPI docs (`/docs`).
app = FastAPI(title="DataOps Environment API")

# Keep a single mutable environment in memory for the interactive API lifecycle.
# Default to easy so first-time users can hit `/state` without a reset call.
env = DataOpsEnv(difficulty="easy")

class StepResponse(BaseModel):
    # Full observation returned after applying an action.
    observation: Observation
    # Reward payload includes score, delta, done flag, and score breakdown.
    reward: Reward

class TaskInfo(BaseModel):
    # Stable task identifier used by clients.
    id: str
    # Difficulty level used by `/reset`.
    difficulty: str

class TasksResponse(BaseModel):
    # Enumerated task list for client selection.
    tasks: List[TaskInfo]
    # JSON schema for valid `Action` objects.
    action_schema: Dict[str, Any]

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the service is running.
    """
    return {"status": "ok"}

@app.get("/tasks", response_model=TasksResponse)
async def get_tasks():
    """
    Returns the available tasks and the action schema.
    """
    # Keep tasks explicit so they stay in sync with openenv.yaml defaults.
    tasks = [
        {"id": "easy", "difficulty": "easy"},
        {"id": "medium", "difficulty": "medium"},
        {"id": "hard", "difficulty": "hard"}
    ]
    return {
        "tasks": tasks,
        "action_schema": Action.model_json_schema()
    }

@app.post("/reset", response_model=Observation)
async def reset_env(difficulty: str = "easy"):
    """
    Resets the environment to its initial state with the specified difficulty.
    """
    global env
    logger.info(f"Resetting environment with difficulty: {difficulty}")
    try:
        # Replace the global instance so each reset starts from a pristine state.
        env = DataOpsEnv(difficulty=difficulty)
        return env.reset()
    except Exception as e:
        logger.error(f"Reset failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset environment: {str(e)}")

@app.post("/step", response_model=StepResponse)
async def step_env(action: Action):
    """
    Applies an action to the environment.
    """
    logger.info(f"Stepping environment with action: {action.action_type}")
    try:
        # Delegate full validation/scoring behavior to the environment implementation.
        obs, reward = env.step(action)
        return StepResponse(observation=obs, reward=reward)
    except Exception as e:
        logger.error(f"Step failed: {e}")
        # Ensure invalid actions do not crash the server
        raise HTTPException(status_code=400, detail=f"Action execution failed: {str(e)}")

@app.get("/state", response_model=Observation)
async def get_state():
    """
    Retrieves the current state of the environment.
    """
    try:
        return env.state()
    except Exception as e:
        logger.error(f"State retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve state: {str(e)}")

@app.post("/grader")
async def grade_env():
    """
    Returns the final score for the current environment state.
    """
    logger.info("Grading environment")
    try:
        # Compute granular metrics first, then expose the normalized total score.
        score_breakdown = env.compute_score()
        score = score_breakdown.get("total", 0.0)
        # Ensure value is between 0 and 1
        score = max(0.0, min(1.0, score))
        return {"score": score}
    except Exception as e:
        logger.error(f"Grading failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to grade environment: {str(e)}")

@app.post("/baseline")
async def run_baseline():
    """
    Runs the baseline agent for all 3 tasks and returns the scores.
    """
    logger.info("Running baseline agent for all tasks")
    results = {}
    difficulties = ["easy", "medium", "hard"]
    
    for diff in difficulties:
        logger.info(f"Running baseline for difficulty: {diff}")
        try:
            # Isolate each difficulty so actions do not leak across runs.
            task_env = DataOpsEnv(difficulty=diff)
            # Agent consumes observations and emits candidate actions.
            agent = SimpleAgent(env=task_env)
            # Keep baseline short for quick sanity checks in CI/local runs.
            agent.run(max_steps=5, k=1) 
            
            score_breakdown = task_env.compute_score()
            score = score_breakdown.get("total", 0.0)
            results[diff] = max(0.0, min(1.0, score))
        except Exception as e:
            logger.error(f"Baseline failed for {diff}: {e}")
            results[diff] = 0.0
            
    return results

if __name__ == "__main__":
    import uvicorn
    # Ready for uvicorn execution
    uvicorn.run(app, host="0.0.0.0", port=3000)
