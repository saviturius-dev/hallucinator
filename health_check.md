# Health Verification Checklist

Follow these steps to verify the DataOps Environment API is running correctly.

## 1. Service Availability
Verify the API is up and responding to basic requests.

**Check:** `GET /health` returns `{"status": "ok"}`
```bash
curl -X GET http://localhost:7860/health
```

## 2. Task Configuration & Schema
Ensure the environment can provide task details and the expected action format.

**Check:** `GET /tasks` returns a list of tasks and a valid JSON schema for actions.
```bash
curl -X GET http://localhost:7860/tasks
```

## 3. Environment Lifecycle
Verify that the environment can be reset and advanced.

**Check:** `POST /reset` initializes the environment.
```bash
# Reset to easy difficulty (default)
curl -X POST "http://localhost:7860/reset?difficulty=easy"

# Reset to hard difficulty
curl -X POST "http://localhost:7860/reset?difficulty=hard"
```

**Check:** `POST /step` applies an action and returns an observation.
```bash
curl -X POST http://localhost:7860/step \
     -H "Content-Type: application/json" \
     -d '{
       "action_type": "clean_data",
       "parameters": {"method": "drop_duplicates"}
     }'
```

## 4. Grading & Baseline
Verify the scoring and agent execution logic.

**Check:** `POST /grader` returns a score between 0 and 1.
```bash
curl -X POST http://localhost:7860/grader
```

**Check:** `POST /baseline` executes the internal agent for all tasks.
*Note: This may take a few seconds as it calls the LLM.*
```bash
curl -X POST http://localhost:7860/baseline
```

---

## 5. OpenEnv Validation
If you have the `openenv` CLI installed, you can validate the environment configuration file.

**Command:**
```bash
openenv validate openenv.yaml
```

**What it checks:**
- Validates YAML syntax.
- Ensures all required fields (`name`, `entrypoint`, `api`) are present.
- Verifies that the defined API endpoints match the expected structure.
