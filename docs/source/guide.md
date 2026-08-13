# User guide

Symbolic regression discovers interpretable mathematical equations from data.
The resulting equations are phenomenological laws: they describe relationships
between observed variables, but generally do not explain the physical processes
that produce those relationships.

Physical laws are often derived by combining more fundamental relationships,
such as conservation laws, force balances, constitutive relations, and
geometric constraints. Mechanism discovery therefore asks for more than a
fitted equation: it seeks basic, physically meaningful relationships whose
composition explains the observations.

MDBench evaluates this capability using problems that pair phenomenological
laws with their underlying mechanism descriptions. It separates three tasks:

1. **Symbolic regression:** `(X, y) → phenomenological equation`.
2. **Mechanism explanation:** phenomenological equation → mechanism equations.
3. **Mechanism discovery:** `(X, y) → mechanism equations`.

## Quick start

MDBench requires Python 3.12 or newer.

```bash
pip install mdbench
mdbench --help
```

### Choose what to evaluate

A prepared task states what the solver receives and what it must return:

| Task | Solver receives | Solver submits |
| --- | --- | --- |
| `symbolic_regression` | `(X, y)` training data | one phenomenological formula |
| `mechanism_explanation` | a phenomenological formula | mechanism equations |
| `mechanism_discovery` | `(X, y)` training data | mechanism equations |

For symbolic regression, submit one equation:

```text
T = sqrt(4 * π**2 * a**3 / (G * M))
```

For a mechanism task, submit one relationship per line:

```text
r = a
F = G * M * m / r**2
acc = F / m
v = sqrt(r * acc)
T = 2 * π * r / v
```

The relationships must form a solvable system that produces the declared target
variable.

### Prepare a local task

First export the problem definitions bundled with the installed package. The
remaining commands validate those definitions, generate data, and prepare
mechanism-discovery tasks. `--save-answer` creates the private files needed
for a local evaluation.

```bash
mdbench export  --output-dir problems # The same problems are available from GitHub Releases.
mdbench validate --problems problems
mdbench synthetic --problems problems --output-dir data/synthetic_data # The same synthetic data is available from GitHub Releases.
mdbench prepare \
  --task mechanism_discovery \
  --problems problems \
  --synthetic-data-dir data/synthetic_data \
  --output-dir data/problem \
  --format directory \
  --reveal-auxiliary \
  --save-answer
```

The generated directory contains public task files and, because
`--save-answer` was supplied, private evaluation files:

```text
problem.json        public problem statement and task contract
data_train.npy      public training observations
answer.json         private reference answer, included by --save-answer
data_id_test.npy    private in-domain test observations
data_ood_test.npy   private out-of-domain test observations
```

Read `problem.json` before solving. Its most important fields are:

| Field | Information provided |
| --- | --- |
| `problem_description` | Physical setting and assumptions. |
| `input` | Whether to use `data` or a `formula`. |
| `expected_output` | Whether to return a `formula` or `mechanisms`. |
| `target_variable` | Variable the answer must produce. |
| `variables` and `constants` | Names, meanings, units, and known constant values. |
| `phenomenological_formula` | Input equation for mechanism explanation. |

For a data-input task, load `data_train.npy`; its target row is first, followed
by the observable inputs listed in `variables`.

### Submit and evaluate

Save the mechanism equations in `submission.txt`. During exploration, request
feedback using only the prepared public task and its training data:

```bash
# (feedback) Public feedback uses only problem.json and public training data.
mdbench evaluate \
  --evaluation-mode feedback \
  --problem PATH/TO/PREPARED_TASK \
  --submission submission.txt \
  --verbose

# (final) Final evaluation uses answer.json and the private ID/OOD test data.
mdbench evaluate \
  --evaluation-mode final \
  --answer PATH/TO/PREPARED_TASK/answer.json \
  --submission submission.txt \
  --verbose
```

The benchmark operator runs final evaluation separately with
`--evaluation-mode final --answer PATH/TO/answer.json`. In a private
benchmark, participants receive only `problem.json`, `data_train.npy`, and a
temporary working directory; the operator keeps the answer and test arrays.
Run `mdbench <command> --help` for command-specific options.

## Problem YAML format

Problems contain six top-level fields:

```yaml
problem_name: Example law
problem_description: A short physical description.
phenomenological_formula: y = sqrt(k * x)

variable_description:
  target:
    name: y
    description: Observable output
    unit: m
  inputs:
    - name: x
      description: Observable input
      unit: m
      sampling:
        min: 1.0
        max: 10.0
        ood_boundary: 8.0
        distribution: uniform
  intermediates:
    - name: a
      description: Derived mechanism state
      unit: m
  auxiliary_inputs: []

constants:
  - name: k
    description: Model coefficient
    value: 1.0
    unit: 1 (dimensionless)

mechanism:
  - formula: a = k * x
    formula_description: Constitutive relationship
  - formula: y = sqrt(a)
    formula_description: Observation relationship
```

### Variable roles

- `target` is the phenomenological dependent variable.
- `inputs` occur in the phenomenological formula and require sampling rules.
- `intermediates` are produced by mechanism relationships. Sampling is not
  allowed; units may be inferred when omitted.
- `auxiliary_inputs` are externally sampled mechanism inputs eliminated from
  the phenomenological law.

Every subsection must be present. Use `[]` when a category is empty.

### Units, formulas, and sampling

Units are exchanged internally as base-SI exponent dictionaries and may be
written as strings such as `kg m^-1 s^-2` in YAML. Dimensionless quantities use
`1 (dimensionless)`.

Formulas must be parseable by nd2py. A mechanism left-hand side is currently a
single variable; `0 = formula` is not supported. Declare every referenced
variable and constant, and do not declare unused entries.

For sampling, `min` to `ood_boundary` defines the train and ID domain, while
`ood_boundary` to `max` defines the OOD domain. Supported distributions are
`uniform` and `log_uniform`.

## Describing mechanisms in MDBench

A mechanism description records the underlying relationships in their natural,
physically meaningful form. Authors do not need to eliminate intermediate
variables or rewrite every relationship into the final phenomenological
equation. MDBench turns the relationships into an executable model that maps
known inputs to intermediate states and ultimately to the target variable.

### A concrete example

Consider a planet of mass `m` moving on a circular orbit of radius `a` around a
star of mass `M`. Its observable period follows Kepler's third law:

```text
T = sqrt(4 * π**2 * a**3 / (G * M))
```

Here `G` is the gravitational constant and `π` is the circle constant. A
mechanism description introduces orbital radius `r`, gravitational force `F`,
centripetal acceleration `acc`, and orbital speed `v`:

```text
r = a                         circular-orbit assumption
F = G * M * m / r**2          universal gravitation
acc = F / m                   Newton's second law
v = sqrt(r * acc)             uniform circular motion
T = 2 * π * r / v             period of one revolution
```

Substitution eliminates `r`, `F`, `acc`, `v`, and the auxiliary input `m`,
leaving the phenomenological law. The mechanism retains the lower-level
physical relationships instead of only the final input-output equation.

### Symbolic derivation and numerical execution

The same mechanism description supports two distinct routes:

| Route | Result | Most useful when |
| --- | --- | --- |
| **Symbolic derivation** | Explicit formulas are substituted until the target is expressed in terms of the inputs. | A known, trustworthy phenomenological equation should be explained. |
| **Numerical execution** | The mechanism is evaluated at input values, solving implicit relationships numerically when necessary. | The target law is unknown, lacks a closed form, or is only an approximate empirical fit. |

Symbolic derivation makes a mechanism explanation easy to inspect: it can show
that gravitation and circular motion algebraically reproduce Kepler's law. This
is particularly valuable for the **mechanism explanation** task, where the
phenomenological equation is provided and symbolic equivalence is meaningful.

In a real **mechanism discovery** setting, the phenomenological equation is
usually unavailable. Even when an empirical equation exists, it may only
approximate the observations and should not be treated as an exact identity.
The numerical route is therefore more general: execute the proposed mechanism
on observed inputs, compare its target predictions with data, and inspect its
behavior in ID and OOD domains. It remains usable when intermediate equations
have no closed-form solution.

### From relationships to executable steps

At the start of execution, MDBench knows the values of three kinds of symbols:

- `inputs`, supplied by the phenomenological problem or dataset;
- `auxiliary_inputs`, supplied only to the mechanism;
- `constants`, fixed by the problem definition.

Every other symbol appearing in a relationship is initially unresolved.
MDBench repeatedly selects the smallest closed set of remaining relationships:
it must contain as many independent equations as unresolved variables.
Relationships in the same closed system need not be adjacent in the YAML file.

### Directly computable relationships

If an equation introduces one unresolved left-hand variable and its right-hand
side contains only known values, it can be evaluated immediately:

```text
a = x1 + x2
b = a**2 / x1
y = sqrt(a + b)
```

The solution order is `a`, then `b`, then `y`. These relationships form an
ordinary directed acyclic dependency graph.

```{figure} _static/mechanism_explicit.svg
:alt: Explicit mechanism dependency graph from x through a and b to y
:width: 90%

An explicit mechanism is rendered as a left-to-right DAG. Blue ellipses are
external inputs and the double circle is the phenomenological target.
```

### One-variable implicit relationships

A variable may occur on both sides when that form expresses the underlying
relationship more naturally:

```text
a = cos(a) + x
y = 2 * a
```

The first equation does not provide a direct assignment because evaluating its
right-hand side already requires `a`. MDBench instead treats it as
`a - cos(a) - x = 0` and solves for `a` as a function of the known input `x`.

```{figure} _static/mechanism_implicit_single.svg
:alt: Single-variable implicit mechanism with a self-loop
:width: 90%

The purple self-loop marks a one-variable implicit solution. Dashed edges mean
that no explicit closed form was found and evaluation uses numerical root
finding.
```

### Coupled implicit systems

Several unresolved variables may need to be solved together:

```text
a = (x + b) / 2
b = (x + a) / 3
y = a + b
```

Neither `a` nor `b` is directly computable. The first two equations form a
closed two-equation system for `{a, b}`. Once both are available, the final
relationship computes `y` normally. The same rule extends to three or more
variables.

```{figure} _static/mechanism_implicit_coupled.svg
:alt: Coupled implicit mechanism with a and b solved together
:width: 90%

Variables `a` and `b` share an implicit-solution block and are connected as a
cycle to show that neither can be evaluated before the other is solved.
```

### Closed systems and numerical fallback

For each closed system, MDBench asks SymPy for explicit formulas such as
`a = f(x)` and `b = g(x)`. Suitable closed forms support both symbolic
substitution and direct numerical evaluation. If symbolic solving fails,
MDBench creates a batched numerical root-finding function from the residual
equations, so the mechanism remains executable without a closed form.

An underdetermined system has fewer equations than unresolved variables and is
rejected. A pending equation that contains no unresolved variable is also
rejected because it duplicates or overdetermines a relationship already used.

The problem definition keeps the original, physically readable equations.
The derived explicit formulas or numerical functions are stored separately as
ordered solution steps and are used by validation, data generation, evaluation,
and visualization.

## Problem library

The installed MDBench distribution includes all reference problem YAML files.
Export them before authoring or running local benchmark tasks:

```bash
mdbench export --output-dir problems
```

The command creates the destination directory when necessary. It refuses to
overwrite existing problem files unless the user confirms the operation or
passes `--force`; it never clears the directory or removes unrelated files.

## Validation

`mdbench validate` checks schema completeness, variable usage, physical units,
sampling specifications, symbolic or numerical solvability, and equivalence
between the solved mechanism and target law.

An optional LLM check evaluates whether each relationship is more fundamental
than the phenomenological law:

```bash
mdbench validate --problems problems \
  --check-fundamentality --llm-provider deepseek --llm-model deepseek-v4-flash
```

API or response failures are reported directly and do not fall back to a
heuristic.

## Synthetic data

`mdbench synthetic` creates reproducible train, ID-test, and OOD-test arrays.
Each NPZ stores `train`, `id_test`, `ood_test`, the row order in `variables`,
and a JSON `generation_config` containing the seed and sample counts.

The pilot stage samples the full configured range before dataset generation and
rejects problems that do not yield enough finite, real-valued samples. Auxiliary
inputs are generated here and may be hidden during task preparation.

## Task preparation

Task preparation converts an author-facing problem definition into a
leakage-controlled package for one benchmark task. Data-input tasks require a
matching NPZ previously produced by `mdbench synthetic`; mechanism explanation
does not require or include synthetic arrays.

```bash
mdbench prepare \
  --problems PATH/TO/problems \
  --synthetic-data-dir data/synthetic_data \
  --output-dir data/problem \
  --task mechanism_discovery \
  --format directory
```

### Public task contract

Every `problem.json` contains these fields:

| Field | Meaning |
| --- | --- |
| `problem_name` | Human-readable problem identifier. |
| `problem_description` | Scientific setting, assumptions, and relevant context. |
| `task` | One of the three MDBench task names. |
| `input` | Authoritative input kind: `data` or `formula`. |
| `expected_output` | Required answer kind: `formula` or `mechanisms`. |
| `target_variable` | Variable the submitted law or mechanism must produce. |
| `variables` | Target first, then observable inputs; each includes name, description, and unit. |
| `constants` | Constants visible to the solver, including value, description, and unit. |

Additional fields depend on the selected task:

| Task | Additional public information | Hidden information |
| --- | --- | --- |
| `symbolic_regression` | training data | phenomenological formula and mechanisms |
| `mechanism_explanation` | `phenomenological_formula` | reference mechanisms |
| `mechanism_discovery` | training data | phenomenological formula and reference mechanisms |

`phenomenological_formula`, when public, is a complete equation such as
`T = sqrt(4 * π**2 * a**3 / (G * M))`.

### Training array

For symbolic regression and mechanism discovery, `data_train.npy` is a
two-dimensional `float64` array with shape
`(number_of_visible_variables, number_of_samples)`. Its rows are ordered as:

```text
target variable
observable input 1
observable input 2
...
revealed auxiliary input 1, if requested
...
```

The target and observable-input portion exactly matches the order of
`problem.json["variables"]`. If auxiliary inputs are revealed, their metadata is
stored separately in `problem.json["auxiliary_input_variables"]` and their rows
are appended after the observable inputs.

Mechanism-explanation packages contain no data arrays because their public
input is the phenomenological formula.

### Auxiliary-input visibility

By default, a mechanism task exposes only variables and constants appearing in
the phenomenological equation. This tests whether the solver can propose useful
latent physical quantities rather than receiving them from the task author.

`--reveal-auxiliary` changes three things:

1. adds `auxiliary_input_variables` metadata to `problem.json`;
2. appends their rows to `data_train.npy` for mechanism discovery;
3. reveals constants used only by the mechanism.

The option is available only for mechanism explanation and mechanism discovery.
It is rejected for symbolic regression because those variables are absent from
the target phenomenological relationship.

### Public and private artifacts

Without `--save-answer`, a data-input package contains:

```text
problem.json
data_train.npy
```

For mechanism explanation it contains only `problem.json`. With
`--save-answer`, preparation adds `answer.json`; data-input tasks additionally
receive the private test arrays:

```text
answer.json
data_id_test.npy    data-input tasks only
data_ood_test.npy   data-input tasks only
```

`answer.json` always contains:

| Field | Meaning |
| --- | --- |
| `task` | Task needed to interpret the submission. |
| `target_variable` | Required final left-hand variable. |
| `data_variables` | Row order used by private arrays; present only for data-input tasks. |
| `source_variables` | Inputs, auxiliary inputs, and constants available during evaluation. |
| `phenomenological_formula` | Private target equation. |
| `constants` | Complete constant metadata, including mechanism-only constants. |

For mechanism tasks it also contains:

- `mechanisms`: reference equations and their physical descriptions;
- `intermediate_variables`: reference intermediate names, descriptions, and
  units.

Symbolic-regression answers intentionally omit both mechanism fields. The ID and
OOD arrays retain all generated target, observable-input, and auxiliary-input
rows needed for private evaluation, regardless of what was revealed publicly.

### Directory and file formats

`--format directory` writes the artifacts above as separate flat files under
one task directory. `--format file` stores the same logical artifacts in one
compressed NPZ with these keys:

```text
problem_json
data_train        data-input tasks only
answer_json       only with --save-answer
data_id_test      data-input tasks only, with --save-answer
data_ood_test     data-input tasks only, with --save-answer
```

The JSON entries are scalar JSON strings; the data entries are NumPy arrays.
Choosing a storage format does not change field contents or visibility.

### Safe overwrite behavior

MDBench never clears an existing output directory. If planned output files
already exist, it requests confirmation; `--force` supplies that confirmation
for non-interactive batch runs. Existing files not produced by the current task
remain untouched and are reported as redundant, which helps detect stale
private artifacts such as an old `answer.json` in a newly public package.

## Evaluation

MDBench deliberately does not report an overall mechanism score. Each metric
answers a different scientific question.

### Feedback and final evaluation

`--evaluation-mode feedback` accepts a prepared public task through
`--problem`. Symbolic regression and mechanism discovery are evaluated only on
`data_train`; mechanism explanation is checked against its public
phenomenological equation. Fundamentality feedback is available for mechanism
tasks. This mode never loads an answer or reports reference-mechanism recovery.

`--evaluation-mode final` accepts `--answer`. It additionally evaluates hidden
ID and OOD arrays stored beside the answer (or inside the same packed NPZ) and,
for mechanism tasks, reports recovery of the reference mechanism structure.
Prediction accuracy is reported for symbolic regression and mechanism discovery;
mechanism explanation instead compares the derived equation with the supplied
phenomenological equation in both modes.

For an Agent run, copy only the prepared public task into an isolated temporary
working directory and require the Agent to remain in that directory. The Agent
may use the normal `mdbench evaluate --evaluation-mode feedback` command. Other
lifecycle commands cannot find their default `./problems` source directory, and
final evaluation cannot find a private answer. This boundary depends on the
runtime enforcing the working-directory and filesystem restrictions; CLI
arguments alone are not an access-control mechanism.

### Prediction accuracy

Feedback for data-driven tasks executes the submission on public training
inputs. Final evaluation applies the identical calculation to train, hidden ID,
and hidden OOD data. For every split MDBench reports Pearson correlation, R²,
MAE, RMSE, symmetric MAPE, and the percentage within `rtol=1e-6` and
`atol=1e-9`. Metrics use only paired finite prediction and target values.

### Derived-equation equivalence

This metric resolves the submitted mechanism into its target equation and
compares the two relationships through SymPy simplification, numerical
evaluation, and an LLM equivalence judgment. Mechanism-explanation feedback can
use the phenomenological equation supplied by its public task. Mechanism
discovery reports this metric only during final evaluation, against the private
phenomenological equation. The report includes the derived equation and each
judge's result; the LLM provider and model are recorded explicitly.

### Mechanism fundamentality

This LLM-assisted metric asks whether every submitted relationship expresses an
independently meaningful lower-level principle. It requires no reference
mechanism and is reported in both feedback and final evaluation. The aggregate is
dominated by the weakest item:

```text
0.7 × minimum item score + 0.3 × mean item score
```

The report prints the evaluator as `model @ provider`, the bottleneck and mean,
and the aggregation rule. Each relationship is shown once with its percentage,
judgment, and short reason. Model failures are reported as unavailable and do
not fall back to a heuristic.

### Ground-truth structure recovery

nd2py expressions are converted to anonymous AST signatures. Variable spelling
and numeric literal values are ignored, while repeated-variable patterns are
retained. Relationship dependencies, including implicit self-dependencies,
form a directed graph. Normal benchmark-sized mechanisms use globally optimal
attributed-graph assignment; larger systems use iterative assignment.

```text
score = 0.6 × formula similarity + 0.4 × dependency graph similarity
```

The two components are diagnostics, not additional benchmark metrics.

### Mechanism description complexity

This reference-free metric parses the right-hand side of each submitted
relationship with nd2py and uses `len(expression)` directly. It reports the
mean expression length, largest expression, and total length. The left-hand
variable is not counted. Lower values are simpler. The calculation is identical
in feedback and final evaluation and does not establish physical validity or
fundamentality.

### Submission format and diagnostics

Pass an inline formula, semicolon-separated mechanism equations, or a text file
with one equation per non-empty line. JSON and YAML submissions are not
accepted. `--verbose` prints one equation chain per solved variable. A second
right-hand side is appended only when the submitted formula depends on an
intermediate variable and can be expanded into a formula containing only input
variables, auxiliary inputs, and constants.

## Mechanism visualization

The visualization API returns a complete Graphviz DOT document:

```python
from pathlib import Path

from src.features.io import load_problem
from src.features.visualization import MechanismGraphBuilder

problem = load_problem("problem.yaml", solve=True)
dot = MechanismGraphBuilder().build(problem)
Path("mechanism.dot").write_text(dot, encoding="utf-8")
```

DOT output has no external system dependency. SVG, PNG, and PDF require
Graphviz. Explicit dependencies are rendered as a DAG; implicit solution blocks
use cyclic styling, and numerically solved dependencies use dashed edges.

The circular-orbit example from the beginning of this guide produces the
following complete physical mechanism graph:

```{figure} _static/mechanism_kepler.svg
:alt: Mechanism graph for Kepler's third law under the circular-orbit assumption
:width: 100%

Kepler's third law derived from the circular-orbit assumption, universal
gravitation, Newton's second law, and uniform circular motion. Constants are
diamonds; the planet mass `m` is an auxiliary input eliminated before `T`.
```

The four embedded SVGs are generated by `docs/generate_assets.py` through the
same `MechanismGraphBuilder` used by MDBench, so the documentation reflects the
actual visualization behavior.

## Build the documentation

```bash
cd docs
make html
```

The generated site is written to `docs/build/html`.
