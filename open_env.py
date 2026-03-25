import json
import os
import random
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

class DataOpsEnv(OpenEnv):
    """
    Concrete implementation of the OpenEnv for data operations.
    """
    def __init__(self, difficulty: Literal["easy", "medium", "hard"] = "easy"):
        self.difficulty = difficulty
        self.full_dataset: List[Dict[str, Any]] = []
        self.current_dataset: List[Dict[str, Any]] = []
        self.schema: Dict[str, str] = {}
        self.constraints: List[str] = []
        self.step_count: int = 0
        self.history: List[Action] = []
        self.seed = 42
        self.last_action_valid = True
        self.last_action_changed = False
        self.target_schema: Dict[str, str] = {}
        self.max_steps = 20

    def reset(self) -> Observation:
        """
        Resets the environment and generates a dataset based on difficulty.
        """
        random.seed(self.seed)
        self.step_count = 0
        self.history = []
        self._generate_data()
        return self.state()

    def _generate_data(self):
        """
        Generates full_dataset and current_dataset based on difficulty.
        """
        if self.difficulty == "easy":
            self._generate_easy()
        elif self.difficulty == "medium":
            self._generate_medium()
        else:
            self._generate_hard()

    def _generate_easy(self):
        # Clean ground truth
        self.full_dataset = [
            {"id": 1, "name": "Alice", "age": 25, "city": "New York"},
            {"id": 2, "name": "Bob", "age": 30, "city": "London"},
            {"id": 3, "name": "Charlie", "age": 35, "city": "Paris"},
            {"id": 4, "name": "David", "age": 40, "city": "Berlin"},
            {"id": 5, "name": "Eve", "age": 45, "city": "Tokyo"},
        ]
        self.schema = {"id": "int", "name": "string", "age": "int", "city": "string"}
        self.target_schema = self.schema.copy()
        self.constraints = ["id must be unique", "age must be positive"]

        # Corrupted version
        corrupted = [
            {"id": 1, "name": "Alice", "age": 25, "city": "New York"},
            {"id": 1, "name": "Alice", "age": 25, "city": "New York"}, # Duplicate
            {"id": 2, "name": "Bob", "age": None, "city": "London"}, # Missing value
            {"id": 3, "name": "Charlie", "age": 35, "city": "paris"}, # Inconsistent format (lowercase)
            {"id": 4, "name": "David", "age": 40, "city": "Berlin"},
            {"id": 5, "name": "Eve", "age": 45, "city": "TOKYO"}, # Inconsistent format (uppercase)
        ]
        self.current_dataset = corrupted

    def _generate_medium(self):
        # Medium includes all of easy + schema mismatches + simple constraints
        self._generate_easy()
        
        # Add schema mismatch
        self.current_dataset[0]["age"] = "25" # String instead of int
        
        # Add simple constraint: end_date >= start_date
        self.full_dataset = [
            {**row, "start_date": "2023-01-01", "end_date": "2023-01-10"} for row in self.full_dataset
        ]
        self.current_dataset = [
            {**row, "start_date": "2023-01-01", "end_date": "2023-01-10"} for row in self.current_dataset
        ]
        
        # Violate constraint in one row
        self.current_dataset[4]["start_date"] = "2023-01-15"
        self.current_dataset[4]["end_date"] = "2023-01-10"
        
        self.schema.update({"start_date": "date", "end_date": "date"})
        self.target_schema = self.schema.copy()
        self.constraints.append("end_date >= start_date")

    def _generate_hard(self):
        # Hard includes multiple datasets, dependencies, and scheduling metadata
        self._generate_medium()
        
        # Add a second dataset: Orders
        orders_full = [
            {"order_id": 101, "user_id": 1, "amount": 50.0, "status": "shipped"},
            {"order_id": 102, "user_id": 2, "amount": 150.0, "status": "pending"},
        ]
        
        # Corrupted Orders
        orders_corrupted = [
            {"order_id": 101, "user_id": 1, "amount": 50.0, "status": "shipped"},
            {"order_id": 102, "user_id": 99, "amount": 150.0, "status": "pending"}, # Dependency violation (user 99 doesn't exist)
        ]
        
        # Merge for current_dataset representation
        for row in self.full_dataset: row["dataset"] = "users"
        for row in orders_full: row["dataset"] = "orders"
        self.full_dataset.extend(orders_full)
        
        for row in self.current_dataset: row["dataset"] = "users"
        for row in orders_corrupted: row["dataset"] = "orders"
        self.current_dataset.extend(orders_corrupted)
        
        # Add scheduling metadata
        self.current_dataset.append({
            "dataset": "tasks",
            "task_id": "T1",
            "deadline": "2023-12-31",
            "priority": "high"
        })
        
        self.constraints.append("order.user_id must exist in users.id")
        self.constraints.append("tasks must be completed before deadline")
        self.target_schema = self.schema.copy()
        self.target_schema.update({"order_id": "int", "user_id": "int", "amount": "float", "status": "string", "dataset": "string"})

    def check_constraints(self, dataset: List[Dict[str, Any]], constraints: List[str]) -> Dict[str, Any]:
        """
        Validates the dataset against the given constraints.
        Returns a dict with pass/fail status and details.
        """
        results = {}
        for constraint in constraints:
            passed = True
            violations = []
            
            if "unique" in constraint.lower():
                col = constraint.split()[0]
                seen = {}
                for i, row in enumerate(dataset):
                    if col in row and row.get("dataset", "users") == "users":
                        val = row[col]
                        if val in seen:
                            passed = False
                            violations.append(f"row_{i}:duplicate_value={val}")
                        seen[val] = i
            
            elif "positive" in constraint.lower():
                col = constraint.split()[0]
                for i, row in enumerate(dataset):
                    if col in row and isinstance(row[col], (int, float)):
                        if row[col] <= 0:
                            passed = False
                            violations.append(f"row_{i}:non_positive_value={row[col]}")
            
            elif ">=" in constraint:
                parts = constraint.split(">=")
                left = parts[0].strip()
                right = parts[1].strip()
                for i, row in enumerate(dataset):
                    if left in row and right in row:
                        l_val, r_val = row[left], row[right]
                        if l_val is not None and r_val is not None:
                            if l_val < r_val:
                                passed = False
                                violations.append(f"row_{i}:relational_violation:{left}<{right}")
            
            elif "exist in" in constraint.lower():
                user_ids = {row["id"] for row in dataset if row.get("dataset") == "users" and "id" in row}
                for i, row in enumerate(dataset):
                    if row.get("dataset") == "orders" and "user_id" in row:
                        if row["user_id"] not in user_ids:
                            passed = False
                            violations.append(f"row_{i}:missing_reference:user_id={row['user_id']}")
            
            elif "before deadline" in constraint.lower():
                for i, row in enumerate(dataset):
                    if row.get("dataset") == "tasks":
                        if "deadline" not in row or row["deadline"] is None:
                            passed = False
                            violations.append(f"row_{i}:missing_deadline")

            results[constraint] = {
                "passed": passed,
                "violations": violations
            }
        return results

    def detect_issues(self, dataset: List[Dict[str, Any]], schema: Dict[str, str], constraints: List[str]) -> List[str]:
        """
        Detects missing values, duplicates, schema mismatches, and constraint violations.
        """
        issues = []
        
        # 1. Missing values
        missing_cols = set()
        for row in dataset:
            for col, val in row.items():
                if val is None:
                    missing_cols.add(col)
        for col in sorted(list(missing_cols)):
            issues.append(f"missing_values:column={col}")
            
        # 2. Duplicate rows
        seen_rows = []
        has_duplicates = False
        for row in dataset:
            if row in seen_rows:
                has_duplicates = True
                break
            seen_rows.append(row)
        if has_duplicates:
            issues.append("duplicate_rows")
            
        # 3. Schema mismatches
        schema_mismatch_cols = set()
        for row in dataset:
            for col, expected_type in schema.items():
                if col in row and row[col] is not None:
                    val = row[col]
                    if expected_type == "int" and not isinstance(val, int):
                        schema_mismatch_cols.add(col)
                    elif expected_type == "string" and not isinstance(val, str):
                        schema_mismatch_cols.add(col)
        for col in sorted(list(schema_mismatch_cols)):
            issues.append(f"schema_mismatch:column={col}")
            
        # 4. Constraint violations
        constraint_results = self.check_constraints(dataset, constraints)
        for constraint, result in constraint_results.items():
            if not result["passed"]:
                if "unique" in constraint.lower():
                    col = constraint.split()[0]
                    issues.append(f"constraint_violation:{col}_uniqueness")
                elif "positive" in constraint.lower():
                    col = constraint.split()[0]
                    issues.append(f"constraint_violation:{col}_positivity")
                elif ">=" in constraint:
                    parts = constraint.split(">=")
                    left = parts[0].strip()
                    right = parts[1].strip()
                    issues.append(f"constraint_violation:{left}_ge_{right}")
                elif "exist in" in constraint.lower():
                    issues.append("constraint_violation:referential_integrity")
                elif "before deadline" in constraint.lower():
                    issues.append("constraint_violation:task_deadline")
                    
        return sorted(list(set(issues)))

    def apply_action(self, action: Action):
        """
        Applies the given action to the current_dataset.
        """
        self.last_action_valid = True
        self.last_action_changed = False
        
        # Deep copy for change detection
        old_dataset = json.loads(json.dumps(self.current_dataset))
        
        try:
            if action.action_type == "fill_missing":
                col = action.parameters.get("column")
                val = action.parameters.get("value")
                if col in self.schema:
                    for row in self.current_dataset:
                        if col in row and row[col] is None:
                            row[col] = val
                else:
                    self.last_action_valid = False

            elif action.action_type == "drop_column":
                col = action.parameters.get("column")
                if col in self.schema:
                    for row in self.current_dataset:
                        if col in row:
                            del row[col]
                    del self.schema[col]
                else:
                    self.last_action_valid = False

            elif action.action_type == "normalize_format":
                col = action.parameters.get("column")
                case = action.parameters.get("case", "lower")
                if col in self.schema and self.schema[col] == "string":
                    for row in self.current_dataset:
                        if col in row and isinstance(row[col], str):
                            if case == "lower": row[col] = row[col].lower()
                            elif case == "upper": row[col] = row[col].upper()
                            elif case == "title": row[col] = row[col].title()
                else:
                    self.last_action_valid = False

            elif action.action_type == "deduplicate":
                new_dataset = []
                for row in self.current_dataset:
                    if row not in new_dataset:
                        new_dataset.append(row)
                self.current_dataset = new_dataset

            elif action.action_type == "enforce_constraint":
                constraint = action.parameters.get("constraint", "")
                if "unique" in constraint.lower():
                    col = constraint.split()[0]
                    seen = set()
                    new_dataset = []
                    for row in self.current_dataset:
                        val = row.get(col)
                        if val not in seen:
                            seen.add(val)
                            new_dataset.append(row)
                    self.current_dataset = new_dataset
                elif "positive" in constraint.lower():
                    col = constraint.split()[0]
                    for row in self.current_dataset:
                        if col in row and isinstance(row[col], (int, float)) and row[col] <= 0:
                            row[col] = abs(row[col]) if row[col] != 0 else 1
                elif ">=" in constraint:
                    parts = constraint.split(">=")
                    left, right = parts[0].strip(), parts[1].strip()
                    for row in self.current_dataset:
                        if left in row and right in row:
                            if row[left] < row[right]:
                                row[left] = row[right] # Simple fix

            elif action.action_type == "merge_dataset":
                # For hard: fix referential integrity
                user_ids = {row["id"] for row in self.current_dataset if row.get("dataset") == "users"}
                for row in self.current_dataset:
                    if row.get("dataset") == "orders":
                        if row["user_id"] not in user_ids:
                            if user_ids:
                                row["user_id"] = list(user_ids)[0]
                            else:
                                self.last_action_valid = False

            elif action.action_type == "schedule_task":
                task_id = action.parameters.get("task_id")
                deadline = action.parameters.get("deadline")
                for row in self.current_dataset:
                    if row.get("dataset") == "tasks" and row.get("task_id") == task_id:
                        row["deadline"] = deadline

            else:
                self.last_action_valid = False

        except Exception:
            self.last_action_valid = False

        # Check if changed
        if old_dataset != self.current_dataset:
            self.last_action_changed = True
        
        self.step_count += 1
        self.history.append(action)

    def compute_score(self) -> Dict[str, float]:
        """
        Computes a deterministic score based on accuracy, schema, and constraints.
        """
        # 1. Data Accuracy (0-1)
        accuracy = 0.0
        if self.full_dataset:
            total_fields = 0
            matches = 0
            
            current_lookup = {}
            for row in self.current_dataset:
                key = f"{row.get('dataset', 'default')}_{row.get('id') or row.get('order_id') or row.get('task_id')}"
                current_lookup[key] = row
            
            for target_row in self.full_dataset:
                key = f"{target_row.get('dataset', 'default')}_{target_row.get('id') or target_row.get('order_id') or target_row.get('task_id')}"
                total_fields += len(target_row)
                if key in current_lookup:
                    curr_row = current_lookup[key]
                    for k, v in target_row.items():
                        if curr_row.get(k) == v:
                            matches += 1
            
            accuracy = matches / total_fields if total_fields > 0 else 0.0
            
            # Symmetric Row Penalty: Penalize for both missing and extra rows
            row_ratio = len(self.current_dataset) / len(self.full_dataset) if len(self.full_dataset) > 0 else 1.0
            if row_ratio != 1.0:
                # Use a penalty factor that drops as the ratio deviates from 1.0
                penalty_factor = min(row_ratio, 1.0 / row_ratio) if row_ratio > 0 else 0.0
                accuracy *= penalty_factor

        # 2. Schema Match (0-1)
        schema_score = 0.0
        if self.target_schema:
            total_schema_fields = len(self.target_schema)
            schema_matches = 0
            for k, v in self.target_schema.items():
                if self.schema.get(k) == v:
                    schema_matches += 1
            schema_score = schema_matches / total_schema_fields if total_schema_fields > 0 else 0.0

        # 3. Constraint Satisfaction (0-1)
        constraint_results = self.check_constraints(self.current_dataset, self.constraints)
        total_constraints = len(self.constraints)
        satisfied_constraints = 0
        
        for constraint, result in constraint_results.items():
            if result["passed"]:
                # Loophole Fix: Ensure there is actually data being validated
                # If the dataset for this constraint is empty, it shouldn't get full credit
                # (Except for referential integrity where an empty orders table is technically valid)
                if "exist in" in constraint.lower():
                    satisfied_constraints += 1
                else:
                    # Check if there's at least one row for the relevant dataset
                    relevant_dataset = "users" if "user" in constraint.lower() or "id" in constraint.lower() else "orders"
                    if any(row.get("dataset") == relevant_dataset for row in self.current_dataset):
                        satisfied_constraints += 1
                    else:
                        satisfied_constraints += 0.5 # Partial credit for "empty but valid"

        constraint_score = satisfied_constraints / total_constraints if total_constraints > 0 else 1.0

        # 4. Efficiency Penalty
        penalty = min(0.3, self.step_count * 0.015) # Slightly higher penalty

        # Final Score (Weighted)
        total = (accuracy * 0.4 + schema_score * 0.3 + constraint_score * 0.3) - penalty
        total = max(0.0, min(1.0, total))

        return {
            "total": round(total, 3),
            "accuracy": round(accuracy, 3),
            "schema": round(schema_score, 3),
            "constraints": round(constraint_score, 3),
            "penalty": round(penalty, 3)
        }

    def state(self) -> Observation:
        """
        Returns the current observation of the environment.
        """
        # Return a preview of the first 5 rows
        preview = self.current_dataset[:5]
        score_breakdown = self.compute_score()
        
        return Observation(
            dataset_preview=preview,
            schema=self.schema,
            issues_detected=self.detect_issues(self.current_dataset, self.schema, self.constraints),
            constraints=self.constraints,
            progress_score=score_breakdown["total"],
            step_count=self.step_count
        )

    def step(self, action: Action) -> Tuple[Observation, Reward]:
        """
        Applies an action to the environment.
        """
        # 1. Store previous score
        prev_score_breakdown = self.compute_score()
        prev_score = prev_score_breakdown["total"]
        
        # 2. Apply action
        self.apply_action(action)
        
        # 3. Get current state and score
        current_observation = self.state()
        curr_score_breakdown = self.compute_score()
        curr_score = curr_score_breakdown["total"]
        
        # 4. Calculate delta
        delta = curr_score - prev_score
        
        # 5. Apply penalties
        penalty = 0.0
        if not self.last_action_valid:
            penalty -= 0.1 # Increased from 0.05
        if not self.last_action_changed:
            penalty -= 0.05 # Increased from 0.02
            
        # Check for repeated actions (excluding current one)
        if len(self.history) > 1:
            last_actions = self.history[:-1]
            # Simple check for repeated action type and parameters
            for prev_action in last_actions:
                if prev_action.action_type == action.action_type and prev_action.parameters == action.parameters:
                    penalty -= 0.05 # Increased from 0.03
                    break
        
        # 6. Check if done
        done = False
        if curr_score >= 0.95:
            done = True
        elif self.step_count >= self.max_steps:
            done = True
            
        # 7. Create Reward object
        reward = Reward(
            score=round(curr_score, 3),
            delta=round(delta + penalty, 3),
            done=done,
            breakdown=curr_score_breakdown
        )
        
        return current_observation, reward

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
