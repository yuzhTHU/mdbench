# MDBench

[![PyPI](https://img.shields.io/pypi/v/mdbench.svg?logo=pypi&logoColor=white&label=PyPI&color=3775A9&cacheSeconds=300)](https://pypi.org/project/mdbench/)
[![Python](https://img.shields.io/pypi/pyversions/mdbench.svg?logo=python&logoColor=white&label=Python&color=3776AB&cacheSeconds=300)](https://pypi.org/project/mdbench/)
[![Documentation](https://img.shields.io/badge/Documentation-online-0A7B83?logo=readthedocs&logoColor=white)](https://yuzhthu.github.io/mdbench/)
[![License](https://img.shields.io/pypi/l/mdbench.svg?label=License&color=2E8B57&cacheSeconds=300)](LICENSE)

[简体中文](README.zh-CN.md)

MDBench evaluates whether an AI system can recover scientific laws and the
mechanisms that produce them from equations or observations.

## What mechanism discovery means

MDBench treats a phenomenological equation as the observable consequence of
several simple, mutually consistent relationships. The phenomenological law
describes *what* variables do; a mechanism explains *why* through physical
relationships, assumptions, and intermediate variables.

For example, Kepler's third law for a circular orbit follows from gravitation,
Newton's second law, and uniform circular motion. See
[`problems/demo_problem.yaml`](problems/demo_problem.yaml).

Each mechanism relationship uses `variable = formula`, where the formula must
be parseable by [nd2py](https://pypi.org/project/nd2py/). Explicit relationships
form a DAG:

```text
a = f1(x)
b = f2(x, a)
y = f3(x, a, b)
```

Implicit systems are also supported. Relationships are collected until the
unknown variables form a closed system, then solved symbolically or with a
numerical root finder:

```text
a = f1(x, a, b)
b = f2(x, a, b)
y = f3(x, a, b)
```

All variables are declared under `variable_description` as `target`, `inputs`,
`intermediates`, or `auxiliary_inputs`. The latter are external variables used
only by the mechanism and eliminated from the final law. The original
relationships remain in `Problem.mechanism`; executable solution steps are
stored in `Problem.solution`.

## Tasks and evaluation

MDBench provides three tasks:

1. **Symbolic regression:** `(X, y) → phenomenological equation`.
2. **Mechanism explanation:** phenomenological equation → mechanism equations.
3. **Mechanism discovery:** `(X, y) → mechanism equations`.

Mechanism evaluation reports independent metrics and deliberately has no
overall score:

- **Prediction accuracy:** for symbolic regression and mechanism discovery,
  Pearson correlation, R², MAE, RMSE, sMAPE, and tolerance accuracy on public
  training data (feedback) or train/ID/OOD data (final).
- **Derived-equation equivalence:** final-only SymPy, numeric, and LLM
  cross-check against the private phenomenological equation.
- **Mechanism fundamentality:** LLM assessment dominated by the least
  fundamental submitted relationship; no reference answer is required.
- **Ground-truth structure recovery:** soft formula-AST and dependency-graph
  matching against the reference mechanism. Variable names and numeric literal
  values are ignored.
- **Mechanism description complexity:** reference-free mean, maximum, and total
  nd2py AST nodes; lower values describe simpler submitted relationships.

Install MDBench with Python 3.12 or newer:

```bash
pip install -e ".[dev]"
mdbench --help
```

The Sphinx documentation lives in [`docs/`](docs/). Build it with:

```bash
cd docs
make html
```

## Commands

### Export bundled problems

MDBench includes its reference problem library. Export it to a local directory
without downloading repository files:

```bash
mdbench export --output-dir problems/
```

The same YAML collection is also available as a prebuilt archive from
[GitHub Releases](https://github.com/yuzhthu/mdbench/releases). Releases also
provide a complete public mechanism-discovery dataset containing `problem.json`,
`answer.json`, and the train, in-domain-test, and out-of-domain-test arrays for
every problem.

Existing files require confirmation; use `--force` for non-interactive
overwrites. Other files in the destination directory are never removed.

The lifecycle commands below accept one or more YAML files or directories
through `--problems`; the default is `./problems`.

### Validate problems

Checks schemas, variable usage, units, sampling specifications, explicit and
implicit equation solving, and derivation of the target law:

```bash
mdbench validate
mdbench validate --problems problems/demo_problem.yaml
```

An optional LLM check evaluates whether every relationship is sufficiently
fundamental. API or response failures are reported directly and do not fall
back to heuristics.

```bash
mdbench validate --check-fundamentality \
  --llm-provider deepseek --llm-model deepseek-v4-flash
```

### Generate synthetic data

Creates reproducible train, ID-test, and OOD-test splits:

```bash
mdbench synthetic --problems problems/ --output-dir data/synthetic_data/
```

Each NPZ stores the three arrays, their row order in `variables`, and a JSON
`generation_config` containing the seed and sample counts. Auxiliary inputs are
generated here and may be hidden later during task preparation.

### Prepare tasks

Synthetic data must already exist. Answers are private by default:

```bash
mdbench prepare \
  --problems problems/ \
  --synthetic-data-dir data/synthetic_data/ \
  --task mechanism_discovery \
  --format directory
```

Use `--save-answer` to include answers and test splits, `--reveal-auxiliary` to
expose auxiliary inputs in mechanism tasks, and `--force` to approve planned
overwrites. Existing directories are never cleared; redundant files are
reported. `--format directory` writes flat files, while `--format file` packs
the same logical artifacts into one NPZ.

### Evaluate submissions

A submission may be an inline formula, semicolon-separated mechanism equations,
or a plain-text file with one equation per non-empty line. JSON and YAML
submissions are intentionally unsupported.

```bash
mdbench evaluate \
  --evaluation-mode feedback \
  --problem data/problem/PREPARED_TASK \
  --submission submission.txt \
  --verbose
```

Feedback mode uses only the public task and training data. Benchmark operators
run final evaluation with `--evaluation-mode final --answer answer.json`, which
also enables hidden ID/OOD tests and reference-mechanism recovery. For Agent
runs, copy only the prepared public task into an isolated temporary working
directory and require the Agent to remain there. Without source problem YAML or
private answer artifacts, the other lifecycle commands and final evaluation
cannot access the material they require. `--verbose` prints concise equation
chains for explicit or implicit solution steps.

Fundamentality scoring automatically uses the configured external model and
prints its provider and model:

```bash
mdbench evaluate \
  --evaluation-mode feedback \
  --problem data/problem/PREPARED_TASK \
  --submission submission.txt \
  --llm-provider deepseek \
  --llm-model deepseek-v4-flash
```

Standalone entry points with equivalent behavior are available in `scripts/`:

```text
validate_problem_main.py   validate problem definitions
synthetic_data_main.py     generate synthetic datasets
prepare_problem_main.py    prepare public/private task artifacts
evaluate_result_main.py    evaluate a submission
```

`scripts/visualize_mechanism_main.py` renders a solved mechanism as DOT, SVG,
PNG, or PDF. Non-DOT formats require Graphviz.

## Directory conventions

```text
problems/                  source problem YAML files
data/
  synthetic_data/         generated train/ID/OOD datasets
  problem/                prepared benchmark tasks
src/
  core/                   dependency-light data models
  features/               project-specific I/O, solving, validation, sampling
  metrics/                formula and mechanism metrics
  utils/                  reusable utilities and LLM clients
scripts/                   standalone command entry points
tests/                     unit tests and validation fixtures
```

A prepared directory contains:

```text
problem.json               public task description
data_train.npy             public training data
answer.json                optional private answer
data_id_test.npy           optional private ID test data
data_ood_test.npy          optional private OOD test data
```

Without `--save-answer`, only `problem.json` and `data_train.npy` are written.
Public interchange types remain simple: units are `Dict[str, int | float]`,
formulas are nd2py-compatible strings, and arrays use NumPy formats.
