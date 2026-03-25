import logging
import time
from typing import Any, Dict, Optional

import requests

# Simple, console-friendly logging for local benchmarking loops.
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Service URL for the running API server.
BASE_URL = "http://localhost:3000"
# Upper bound on action iterations for each task.
MAX_STEPS = 10
# Retries make local startup races less brittle.
MAX_RETRIES = 3
# Backoff between retries.
RETRY_DELAY = 2


def api_request(method: str, endpoint: str, **kwargs) -> Optional[Dict[str, Any]]:
    """
    Execute an HTTP request against the API with lightweight retry logic.
    """
    url = f"{BASE_URL}{endpoint}"
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.request(method, url, timeout=10, **kwargs)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            logger.warning(
                "API %s %s failed (attempt %s/%s): %s",
                method,
                endpoint,
                attempt + 1,
                MAX_RETRIES,
                exc,
            )
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
    return None


def choose_action(observation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Select an action deterministically from issue/constraint keywords.
    """
    # Convert lists to searchable lowercase text.
    issues = " ".join(observation.get("issues_detected", [])).lower()
    constraints = " ".join(observation.get("constraints", [])).lower()
    context = f"{issues} {constraints}"

    # Priority ordering below acts as a compact policy.
    if "missing" in context:
        return {"action_type": "fill_missing", "parameters": {"column": "region", "value": "unknown"}}
    if "duplicate" in context:
        return {"action_type": "deduplicate", "parameters": {"subset": ["id"]}}
    if "format" in context or "timestamp" in context:
        return {"action_type": "normalize_format", "parameters": {"column": "timestamp", "format": "ISO8601"}}
    if "constraint" in context or "negative" in context:
        return {"action_type": "enforce_constraint", "parameters": {"constraint": "non_negative_sales"}}

    return {"action_type": "schedule_task", "parameters": {"task": "daily_quality_check"}}


def run_baseline() -> None:
    """
    Execute all configured tasks and print a final score table.
    """
    logger.info("Connecting to DataOps API at %s", BASE_URL)
    # Discover tasks from the server rather than hard-coding assumptions here.
    tasks_data = api_request("GET", "/tasks")
    if not tasks_data:
        logger.error("Failed to retrieve tasks. Exiting.")
        return

    tasks = tasks_data.get("tasks", [])
    final_results: Dict[str, float] = {}

    for task in tasks:
        difficulty = task["difficulty"]
        logger.info("--- Starting Task: %s ---", difficulty)
        # Pre-seed output so incomplete tasks still appear in the summary.
        final_results[difficulty] = 0.0

        # Reset gives us the initial observation for the episode.
        obs = api_request("POST", "/reset", params={"difficulty": difficulty})
        if not obs:
            logger.error("Failed to reset environment for %s. Skipping.", difficulty)
            continue

        done = False
        step_count = 0
        last_score = 0.0

        while not done and step_count < MAX_STEPS:
            step_count += 1
            # Choose next action using the deterministic local policy.
            action = choose_action(obs)
            step_resp = api_request("POST", "/step", json=action)
            if not step_resp:
                logger.error("Step %s failed. Stopping task %s.", step_count, difficulty)
                break

            # Pull forward the observation for the next policy decision.
            obs = step_resp.get("observation", {})
            reward_info = step_resp.get("reward", {})
            current_score = reward_info.get("score", 0.0)
            delta = current_score - last_score
            last_score = current_score
            done = reward_info.get("done", False)

            logger.info(
                "Step %s | Action: %s | Score Delta: %+0.3f | Score: %0.3f",
                step_count,
                action.get("action_type"),
                delta,
                current_score,
            )

        # Ask grader for the normalized terminal score.
        grade_resp = api_request("POST", "/grader")
        if grade_resp:
            score = grade_resp.get("score", 0.0)
            final_results[difficulty] = score
            logger.info("Task %s complete. Final Score: %.3f", difficulty, score)

    # Human-readable final summary.
    print("\n" + "=" * 40)
    print(f"{'DIFFICULTY':<15} | {'FINAL SCORE':<15}")
    print("-" * 40)
    for diff, score in final_results.items():
        print(f"{diff.capitalize():<15} | {score:<15.3f}")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    try:
        run_baseline()
    except KeyboardInterrupt:
        logger.info("Baseline run interrupted by user.")
