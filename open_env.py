import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Tuple, Optional
from pydantic import BaseModel, Field
from openai import OpenAI

class Observation(BaseModel):
    """
    Represents the state of the environment at a specific point in time.
    """
    dataset_preview: List[Dict[str, Any]]
    schema: Dict[str, str]
    issues_detected: List[str]
    constraints: List[str]
    progress_score: float
    step_count: int

class Action(BaseModel):
    """
    Defines an operation to be performed on the environment.
    """
    action_type: Literal[
        "fill_missing",
        "drop_column",
        "normalize_format",
        "deduplicate",
        "enforce_constraint",
        "schedule_task",
        "merge_dataset"
    ]
    parameters: Dict[str, Any] = Field(default_factory=dict)

class Reward(BaseModel):
    """
    Encapsulates the feedback received after performing an action.
    """
    score: float
    delta: float
    done: bool
    breakdown: Dict[str, float]

class OpenEnv(ABC):
    """
    Abstract base class for an OpenEnv-style environment.
    """

    @abstractmethod
    def reset(self) -> Observation:
        """
        Resets the environment to its initial state.

        Returns:
            Observation: The initial observation of the environment.
        """
        pass

    @abstractmethod
    def step(self, action: Action) -> Tuple[Observation, Reward]:
        """
        Applies an action to the environment.

        Args:
            action (Action): The action to perform.

        Returns:
            Tuple[Observation, Reward]: A tuple containing the next observation and the reward.
        """
        pass

    @abstractmethod
    def state(self) -> Observation:
        """
        Retrieves the current state of the environment.

        Returns:
            Observation: The current observation of the environment.
        """
        pass

class SimpleAgent:
    """
    An agent that uses an LLM to propose actions for an OpenEnv environment.
    """
    def __init__(self, env: OpenEnv, api_key: Optional[str] = None, model: str = "gpt-4o"):
        self.env = env
        self.model = model
        
        # Determine API configuration
        api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
        
        # If using Gemini key, use Google's OpenAI-compatible endpoint
        if os.getenv("GEMINI_API_KEY") and not os.getenv("OPENAI_API_KEY"):
            default_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        else:
            default_url = "https://api.openai.com/v1"
            
        base_url = os.getenv("OPENAI_BASE_URL", default_url)
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def propose_actions(self, observation: Observation) -> List[Action]:
        """
        Queries the LLM to propose a list of valid candidate actions.
        """
        prompt = f"""
        You are a data engineering agent. Analyze the following environment observation and propose 3 to 5 candidate actions.
        
        Observation:
        {observation.model_dump_json(indent=2)}
        
        Constraints:
        1. Action types MUST be one of: "fill_missing", "drop_column", "normalize_format", "deduplicate", "enforce_constraint", "schedule_task", "merge_dataset".
        2. Parameters must be minimal, relevant, and valid for the specific action type.
        3. Output MUST be a valid JSON object with an "actions" key containing a list of action objects.
        
        Example Output:
        {{
            "actions": [
                {{"action_type": "normalize_format", "parameters": {{"column": "timestamp", "format": "ISO8601"}}}},
                {{"action_type": "fill_missing", "parameters": {{"column": "category", "value": "unknown"}}}}
            ]
        }}
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,  # Ensure determinism
                seed=42,
                response_format={"type": "json_object"}
            )
            
            raw_content = response.choices[0].message.content or ""
            
            # Safe JSON parsing
            try:
                data = json.loads(raw_content)
                # Handle potential wrapper objects (e.g. {"actions": [...]})
                actions_list = data if isinstance(data, list) else data.get("actions", [])
            except json.JSONDecodeError:
                print("Failed to parse LLM output as JSON.")
                return []

            # Validate and discard invalid actions
            valid_actions = []
            for item in actions_list:
                try:
                    valid_actions.append(Action(**item))
                except Exception:
                    # Discard actions that don't match the Pydantic schema
                    continue
            
            return valid_actions

        except Exception as e:
            print(f"LLM proposal error: {e}")
            return []

    def heuristic_score(self, observation: Observation, action: Action) -> float:
        """
        Calculates a heuristic score for a proposed action based on the current observation.
        """
        score = 0.0
        # Combine type and parameters for broader keyword matching
        full_action_ctx = f"{action.action_type} {json.dumps(action.parameters)}".lower()
        
        # Helper to check if any issue/constraint keywords appear in the action
        def matches_context(items: List[str]) -> bool:
            return any(any(word in full_action_ctx for word in item.lower().split() if len(word) > 2) for item in items)

        addressed_issue = matches_context(observation.issues_detected)
        addressed_constraint = matches_context(observation.constraints)

        if addressed_issue: score += 2.0
        if addressed_constraint: score += 2.0

        # Penalize risky actions
        if action.action_type == "drop_column":
            score -= 0.5 if (addressed_issue or addressed_constraint) else 2.0
            
        # Penalize irrelevant actions
        if not (addressed_issue or addressed_constraint):
            score -= 1.0

        return score

    def run(self, max_steps: int = 20, k: int = 3, stagnation_patience: int = 3) -> None:
        """
        Executes a robust beam search to find the most effective sequence of actions.
        """
        # Track cumulative environment reward for stagnation detection
        beam = [{"path": [], "h_score": 0.0, "env_score": 0.0, "done": False, "last_delta": 0.0}]
        best_env_score = -float('inf')
        stagnation_counter = 0
        
        print(f"Starting beam search (k={k}, max_steps={max_steps})...")

        for step_idx in range(1, max_steps + 1):
            new_candidates = []
            active_candidates = [c for c in beam if not c["done"]]
            if not active_candidates: break

            for candidate in beam:
                if candidate["done"]:
                    new_candidates.append(candidate)
                    continue

                try:
                    # Re-play path once to get current state
                    obs = self.env.reset()
                    for prev_action in candidate["path"]:
                        obs, _ = self.env.step(prev_action)
                    
                    proposals = self.propose_actions(obs)
                    if not proposals:
                        candidate["done"] = True
                        new_candidates.append(candidate)
                        continue

                    for action in proposals:
                        if any(a.action_type == action.action_type and a.parameters == action.parameters for a in candidate["path"]):
                            continue
                        
                        h_val = self.heuristic_score(obs, action)
                        
                        # Expansion step
                        self.env.reset()
                        for prev_action in candidate["path"]:
                            self.env.step(prev_action)
                        _, reward = self.env.step(action)
                        
                        new_candidates.append({
                            "path": candidate["path"] + [action],
                            "h_score": candidate["h_score"] + h_val,
                            "env_score": reward.score,
                            "done": reward.done,
                            "last_delta": reward.delta
                        })
                except Exception as e:
                    print(f"Step Error: {e}")
                    continue

            if not new_candidates: break

            # Prune based on heuristic score (agent's internal preference)
            new_candidates.sort(key=lambda x: x["h_score"], reverse=True)
            beam = new_candidates[:k]
            
            # Stagnation based on actual environment reward
            current_best_env = beam[0]["env_score"]
            if current_best_env > best_env_score:
                best_env_score = current_best_env
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            best = beam[0]
            action_name = best["path"][-1].action_type if best["path"] else "None"
            print(f"Step {step_idx:02d} | Action: {action_name:<18} | Delta: {best['last_delta']:>5.2f} | Env Score: {best['env_score']:>6.1f}")

            if stagnation_counter >= stagnation_patience:
                print(f"Stagnation detected at Env Score {best_env_score}. Terminating.")
                break

        # Finalize
        best_candidate = beam[0]
        print(f"\nSearch Complete. Final Env Score: {best_candidate['env_score']:.1f}")
        self.env.reset()
        for action in best_candidate["path"]:
            self.env.step(action)
