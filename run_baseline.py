import os
import json
import time
import requests
import logging
from openai import OpenAI
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
BASE_URL = "http://localhost:3000"
API_KEY = os.getenv("OPENAI_API_KEY")
MAX_STEPS = 10
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# Initialize OpenAI-compatible client
client = OpenAI(api_key=API_KEY)

def api_request(method: str, endpoint: str, **kwargs) -> Optional[Dict[str, Any]]:
    """
    Makes an API request with retry logic.
    """
    url = f"{BASE_URL}{endpoint}"
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.request(method, url, timeout=10, **kwargs)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"API {method} {endpoint} failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
    return None

def validate_action(action: Any, action_schema: Dict[str, Any]) -> bool:
    """
    Validates that the action matches the expected schema structure.
    """
    if not isinstance(action, dict):
        return False
    if "action_type" not in action:
        return False
    
    # Check if action_type is valid according to schema if possible
    allowed_actions = action_schema.get("properties", {}).get("action_type", {}).get("enum", [])
    if allowed_actions and action["action_type"] not in allowed_actions:
        logger.error(f"Invalid action_type: {action['action_type']}. Expected one of {allowed_actions}")
        return False
        
    return True

def get_action_from_llm(observation: Dict[str, Any], action_schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Queries the LLM to propose a single action with retry logic and validation.
    """
    prompt = f"""
    You are a data engineering agent. Analyze the following environment observation and propose a single action.
    
    Observation:
    {json.dumps(observation, indent=2)}
    
    Action Schema:
    {json.dumps(action_schema, indent=2)}
    
    Constraints:
    1. Output ONLY a valid JSON object representing the action.
    2. The JSON object MUST match the Action schema: {{"action_type": "...", "parameters": {{...}}}}
    """
    
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            raw_content = response.choices[0].message.content or ""
            action = json.loads(raw_content)
            
            if validate_action(action, action_schema):
                return action
            else:
                logger.warning(f"LLM produced invalid action (attempt {attempt + 1}/{MAX_RETRIES})")
                
        except Exception as e:
            logger.warning(f"LLM call failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
        
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAY)
            
    return None

def run_baseline():
    """
    Executes the baseline agent for each task difficulty.
    """
    logger.info(f"Connecting to DataOps API at {BASE_URL}...")
    
    # Get available tasks and schema
    tasks_data = api_request("GET", "/tasks")
    if not tasks_data:
        logger.error("Failed to retrieve tasks. Exiting.")
        return

    tasks = tasks_data.get("tasks", [])
    action_schema = tasks_data.get("action_schema", {})
    final_results = {}

    for task in tasks:
        difficulty = task["difficulty"]
        logger.info(f"--- Starting Task: {difficulty} ---")
        final_results[difficulty] = 0.0  # Default score
        
        try:
            # Reset environment
            obs = api_request("POST", "/reset", params={"difficulty": difficulty})
            if not obs:
                logger.error(f"Failed to reset environment for {difficulty}. Skipping task.")
                continue

            done = False
            step_count = 0
            last_reward_value = 0.0

            while not done and step_count < MAX_STEPS:
                step_count += 1
                
                # Generate action
                action = get_action_from_llm(obs, action_schema)
                if not action:
                    logger.error(f"Step {step_count}: No valid action generated. Stopping task.")
                    break
                
                # Apply action
                step_resp = api_request("POST", "/step", json=action)
                if not step_resp:
                    logger.error(f"Step {step_count}: API call failed. Stopping task.")
                    break
                
                obs = step_resp.get("observation", {})
                reward_info = step_resp.get("reward", {})
                
                current_reward_value = reward_info.get("value", 0.0)
                reward_delta = current_reward_value - last_reward_value
                last_reward_value = current_reward_value
                done = reward_info.get("done", False)

                logger.info(
                    f"Step {step_count} | Action: {action.get('action_type')} | "
                    f"Reward Delta: {reward_delta:+.3f} | Total Reward: {current_reward_value:.3f}"
                )

            # Get final grade
            grade_resp = api_request("POST", "/grader")
            if grade_resp:
                score = grade_resp.get("score", 0.0)
                final_results[difficulty] = score
                logger.info(f"Task {difficulty} Complete. Final Score: {score:.3f}")
            else:
                logger.error(f"Failed to get final grade for {difficulty}")

        except Exception as e:
            logger.error(f"Unexpected error during task {difficulty}: {e}")

    # Final Summary
    print("\n" + "="*40)
    print(f"{'DIFFICULTY':<15} | {'FINAL SCORE':<15}")
    print("-" * 40)
    for diff, score in final_results.items():
        print(f"{diff.capitalize():<15} | {score:<15.3f}")
    print("="*40 + "\n")

if __name__ == "__main__":
    if not API_KEY:
        logger.error("OPENAI_API_KEY environment variable is not set.")
    else:
        try:
            run_baseline()
        except KeyboardInterrupt:
            logger.info("Baseline run interrupted by user.")
        except Exception as e:
            logger.critical(f"Fatal script error: {e}")
