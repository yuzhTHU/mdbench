# Benchmark workflow

Use the repository's `proposal.md` for task design and `README.md` for the full
interface contract. New tasks use the Task schema. Legacy schemas are unsupported.

1. Save a YAML task anywhere under `tasks/`, named exactly after `task_name`.
2. Run `mdbench validate --tasks tasks` for recursive algebra, unit and probe checks.
3. Run `mdbench export --tasks tasks --output-dir data/tasks`.
4. Supply only exported `problem/problem.json` and `problem/train.npy` to the agent.
5. Run `mdbench run` with the public problem and private `answer/answer.json`.

Exports separate public observations from private mechanisms/probes and hold
train/ID/OOD arrays with shape `(variables, samples)`. Mechanism submissions are
plain text equalities, one per line. The feedback service accepts three multipart
files and returns train metrics, without knowing the true mechanism.

Offline evaluation separately scores observable predictions and independent
probe conversations restored from the saved end-of-run checkpoint. Each probe
receives objective symbolic and numerical evaluation. No LLM grader is used.

Scientific meaning, nontriviality and identifiability require human task review.
The algebra solver rejects nonunique explicit solutions and does not solve ODEs.
Legacy implementation and documentation are retained under `legacy/`.
