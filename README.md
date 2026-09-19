# Mechanism Discovery Benchmark

[简体中文](README.zh-CN.md)

Mechanism Discovery Benchmark (`mdbench`) evaluates whether an agent can recover
both an observable input–output law and the hidden mechanism that produces it.
Observable behavior is evaluated on training, in-distribution, and
out-of-distribution data, while hidden states are assessed through independent
mechanism probes. All scores use objective symbolic equivalence and numerical
accuracy; no LLM judge is required.

See [`proposal.md`](proposal.md) for the motivation and methodological design.

## Quick start

Requires Python 3.12 or later.

```sh
pip install -e '.[dev]'

# Validate a task instance.
# --tasks accepts files or directories and scans directories recursively.
# Use --tasks tasks/ to select the complete bundled library; *.yaml.archived is ignored.
mdbench validate \
  --tasks 'tasks/astronomy/binary_mass_function/Astronomy Binary Mass Function - Original.yaml'

# Convert the task instance into problem, answer, and data files.
mdbench export \
  --tasks 'tasks/astronomy/binary_mass_function/Astronomy Binary Mass Function - Original.yaml' \
  --output-dir data/tasks \
  --seed 0

# Give the problem files to an algorithm and evaluate its submission automatically.
# Replace dummy with codex to run the Codex baseline; this requires a configured codex CLI.
mdbench run \
  --algorithm dummy \
  --problem-file 'data/tasks/Astronomy Binary Mass Function - Original/problem/problem.json' \
  --answer-file 'data/tasks/Astronomy Binary Mass Function - Original/answer/answer.json' \
  --save-path logs/demo
```

Run `mdbench <command> --help` for all options.

## Benchmark workflow

1. **Define a task.** A YAML file describes observed and hidden variables,
   algebraic mechanisms, the observable law, sampling ranges, units, and probes.
   Start from [`demo_task.yaml`](demo_task.yaml).
2. **Validate the task.** `mdbench validate` checks its schema, solvability,
   observable equivalence, dimensions, sampling rules, and probe consistency.
3. **Export the data.** `mdbench export` creates the public agent input and the
   private evaluation input with reproducible train/ID/OOD data.
4. **Run the evaluation.** `mdbench run` executes an algorithm, freezes its final
   state, scores the observable model, and evaluates each probe from an independent
   copy of that state.

Exports strictly separate public and private information:

```text
<task>/problem/problem.json                  public variable metadata
<task>/problem/train.npy                     public training observations
<task>/answer/answer.json                    private task and probe definitions
<task>/answer/{train,id_test,ood_test}.npy   private evaluation data
```

Arrays use `(variables, samples)` layout in `data_columns` order, with the target
first. Only files under `problem/` should be exposed to the evaluated agent.

## Feedback service

```sh
mdbench feedback --port 8000 --workers 4
```

`POST /evaluate` accepts `problem`, `train_data`, and `submission` multipart
files and returns explicit solutions and training metrics.

The service provides infrastructure such as equation solving and numerical
evaluation during discovery without exposing private answers.

`mdbench run` starts an isolated local service automatically unless an existing
service is supplied through `--feedback-server-url`.

## Repository guide

- `src/`: validation, generation, export, feedback, scoring, and algorithm adapters
- `tasks/`: active task library
- `tests/`: unit, behavior, pipeline, and runner tests
- `legacy/`: archived, unsupported implementation

Run the test suite with:

```sh
python -m pytest
```

Licensed under the [MIT License](LICENSE).
