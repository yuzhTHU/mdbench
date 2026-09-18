# Benchmark workflow

Use the repository's `proposal.md` for task design and `README.md` for the full
interface contract. New tasks use the Task schema. Legacy schemas are unsupported.

1. Save an English YAML directly in `problems/`, named exactly after `task_name`.
2. Run `mdbench validate --problems problems` for algebra, units and probe checks.
3. Run `mdbench export --problems problems --output-dir data/tasks`.
4. Supply only exported `agent/problem.json` and `agent/train.npy` to the agent.
5. Run `python run.py` with the two public files and private `answer/answer.json`.

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
