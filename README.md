# Mechanism Discovery Benchmark

This benchmark follows [proposal.md](proposal.md). It distinguishes recovery of
observable input–output behavior from recovery of unobserved mechanistic states.
The two evaluations use symbolic equivalence and numerical accuracy. There is
no fundamentality assessment, DAG recovery score or LLM grading.

Requires Python 3.12+, NumPy, SymPy and PyYAML. The Codex baseline additionally
requires a working, authenticated `codex` CLI. Install with `pip install -e '.[dev]'`.

## Task instances

New tasks live directly in `problems/`, in English. The filename stem must equal
`task_name`, including spaces and capitalization. A family uses ` - Original`,
` - Variant 1`, or ` - Variant 1-2` suffixes and identical descriptions.
`demo_problem.yaml` is a template; its library copy is
`problems/Electrical Dissipation - Variant 1-2.yaml`. Validate the library copy because the
template filename deliberately does not match its task name.

The schema consists of `task_name`, `task_description`, `mutation`,
`mechanism_model`, `phenomenal_model`, `variables`, and `mechanism_probes`.
Variables have `name`, `description`, `unit`, `role`, and optional `sampling`.
Roles are `target`, `input`, `internal`, `auxiliary`. Only input/auxiliary
variables declare sampling; exactly one variable is the target. Auxiliary
variables generate a warning and probes must still expand into inputs only.

Mechanism equations may put expressions on either side of `=` and appear in any
order. The phenomenal equation puts the target on the left. Named numerical
constants are equations, not a separate constants collection. Constants may be
internal variables, or unlisted names defined by purely numerical equations;
in the latter case their defining equations are excluded from the internal
variable count rule. A phenomenal formula may reference such constants, which
are expanded before verification.

Supported formulas use numeric literals, ASCII identifiers, `+ - * /`, `^` or
`**`, `sqrt`, `exp`, `log`/`ln`, trig/inverse trig, hyperbolic functions and `abs`.
Define named constants such as `pi` numerically if you use them. No calculus or
inequalities are supported. Models must have a unique explicit algebraic
solution; unresolved/multiple branches are rejected rather than guessed.
Task authors can specify a branch directly with `sqrt`, for example.

Units use SI bases (`kg m s A K mol cd`), products, division and numeric powers.
Common derived units (`N J W Pa Hz C V ohm rad`) are also supported. Observed
variables require units; internal units can be inferred when uniquely constrained.
Numeric literals can carry any physical unit and never generate dimensional
literal warnings. Unknown unit constraints are not scientific evidence.

```sh
mdbench validate --problems problems
mdbench synthetic --problems problems --output-dir data/synthetic_data --seed 0
mdbench export --problems problems --output-dir data/tasks --seed 0
```

Validation checks schema, names, sampling, equation counts, algebraic solutions,
phenomenal equivalence, units, and probe solvability/consistency/structural
participation. Scientific meaning, probe nontriviality, resistance to memorization
and mechanism identifiability require human review. Family-description checks
apply when validating/exporting a collection.

## Export and feedback

Export creates one directory per task:

```text
<task>/agent/problem.json       observable metadata only
<task>/agent/train.npy          observed training samples
<task>/answer/answer.json       private task, mechanisms and probes
<task>/answer/train.npy
<task>/answer/id_test.npy
<task>/answer/ood_test.npy
```

NPY arrays have shape **(variables, samples)** in `data_columns` order, target
first. Training and ID tests sample `[min, ood_boundary)`; OOD tests sample
`[ood_boundary, max)` for every source variable. Rejection sampling filters
nonreal/nonfinite mechanism states. An explicit seed gives reproducible samples.
Give agents only the `agent/` files. Mutation, internal variables, mechanisms,
probe questions and probe answers are never included in public exports.

```sh
mdbench feedback --port 8000 --workers 4 --cache-size 128
curl --noproxy '*' http://127.0.0.1:8000/evaluate \
  -F 'problem=@problem.json' -F 'train_data=@train.npy' \
  -F 'submission=@submission.txt'
```

A submission contains one equation per line; semicolons and `#` comments are
accepted. It may introduce its own internal names. `POST /evaluate` accepts the
three multipart file fields shown above and returns JSON with its solved
expressions and train MAE/MSE/RMSE/MAPE/R²/NRMSE/max error/finite fraction and
numerical equivalence. No private answer is needed by the feedback server.
Invalid models return `ok: false` with an error. Cache entries depend on all
three file contents; simultaneous identical requests share one computation.
`GET /health` identifies the service protocol. Local clients should bypass
HTTP proxies (e.g. `requests.Session().trust_env = False`).

## Run and evaluate

```sh
python run.py --algorithm codex \
  --problem-file 'data/tasks/Electrical Dissipation - Variant 1-2/agent/problem.json' \
  --answer 'data/tasks/Electrical Dissipation - Variant 1-2/answer/answer.json' \
  --timeout 600 --probe-timeout 120 --save-path logs/demo
```

The runner starts a local feedback service on a free port, or reuses a verified
server with `--feedback-server-url http://127.0.0.1:8000/evaluate`. Train, ID and
OOD arrays are resolved from the private answer manifest; only the public problem
and train array are copied to the temporary agent workspace. The runner launches Codex and saves
`submission.txt`, `prompt.txt`, `performance.json`, audit logs under `audit/`, and
the resumable session plus model metadata under `saved_checkpoint/`. The timeout interrupts
the process and allows a short checkpoint-flush grace period. The agent is
instructed to save a submission early and keep it updated; a valid final equation
response also serves as a fallback submission. Missing submission/checkpoint
artifacts fail the run and are recorded.

Phenomenal evaluation scores train, ID and OOD predictions and symbolic
agreement. Each mechanism probe restores the same saved end-of-run session into
a fresh isolated Codex home, so probe conversations cannot influence each other.
Probe prompts include the variable meaning and frozen submitted equations,
never the true mechanism or answer. Shell, web, apps, MCP configuration, plugins,
subagents and other available tool features are disabled as far as the CLI
supports; read-only sandbox and prompt constraints supplement this. This is a
soft restriction on revising the mechanism, as specified in the proposal.
Only submitted internal/target expressions may expand a probe reply; true
internal states are never used to repair an agent answer.

Probe artifacts are saved separately under `probe/`. Each probe receives train,
ID and OOD symbolic/numerical scores. A failed probe remains in the denominator
of the mechanism recovery rates. Nonfinite predictions fail equivalence and
return null errors rather than being silently filtered. Numerical equivalence
uses `rtol=1e-6, atol=0`; symbolic comparison uses SymPy simplification.

Standalone evaluation:

```sh
mdbench evaluate --answer '<task>/answer/answer.json' \
  --submission logs/demo/submission.txt \
  --model logs/demo/saved_checkpoint/model.json --save-path logs/re-evaluate
```

Use the same Codex model/provider configuration when resuming. The baseline
copies authentication and provider settings into temporary homes but never
copies credentials to saved artifacts. Operational success (`ok`) means models
and probes were evaluable, not that their predictions were accurate.

## Code boundary and tests

The new pipeline is in `src/core`, `src/validate_problem.py`,
`src/synthetic_data.py`, `src/export_problems.py`, `src/scoring.py`,
`src/feedback_server.py`, `src/evaluate.py`, `src/algorithms`, `src/cli`, and `run.py`.
Old implementations, tests, baselines and scripts are archived in `legacy/`;
old task definitions remain in `problems/legacy/`. They are excluded from new
entry points, test discovery and packaging. There is no schema/API compatibility.
See [migration status](playground/rewrite_plan.md).

```sh
python -m pytest
```

Tests cover validation, coupled equation solving, dimensional constraints,
export privacy, sampling, concurrent HTTP feedback, runner cleanup and failure
reporting. If Codex is installed, its real CLI is tested against a local mock
Responses provider to verify independent checkpoint resumes without paid model
calls. Tests that start local servers require permission to bind sockets.
