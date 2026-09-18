# Example

This example creates a circular-orbit problem, gives Codex only the public
mechanism-discovery task, and evaluates its final submission privately. The
benchmark author and the Agent work in separate directories so the Agent cannot
read the source mechanism, answer, or hidden test data.

## 1. Define the scientific problem

Create `example/problems/circular_orbit.yaml` in the benchmark-authoring
environment:

```yaml
problem_name: Circular orbit

problem_description: |
  A planet of mass m follows a circular orbit of radius a around a star of
  mass M. Discover physically meaningful relationships that explain its
  orbital period T.

phenomenological_formula: T = sqrt(4 * π^2 * a^3 / (G * M))

variable_description:
  target:
    name: T
    description: Orbital period
    unit: s

  inputs:
    - name: a
      description: Orbital radius
      unit: m
      sampling:
        min: 1.0e9
        max: 1.0e12
        ood_boundary: 5.0e11
        distribution: log_uniform
    - name: M
      description: Stellar mass
      unit: kg
      sampling:
        min: 1.0e28
        max: 1.0e31
        ood_boundary: 5.0e30
        distribution: log_uniform

  intermediates:
    - {name: r, description: Circular-orbit radius, unit: m}
    - {name: F, description: Gravitational force, unit: kg m s^-2}
    - {name: acc, description: Centripetal acceleration, unit: m s^-2}
    - {name: v, description: Orbital speed, unit: m s^-1}

  auxiliary_inputs:
    - name: m
      description: Planetary mass
      unit: kg
      sampling:
        min: 1.0e20
        max: 1.0e28
        ood_boundary: 1.0e24
        distribution: log_uniform

constants:
  - name: π
    description: Circle constant
    value: 3.141592653589793
    unit: 1 (dimensionless)
  - name: G
    description: Gravitational constant
    value: 6.67430e-11
    unit: m^3 kg^-1 s^-2

mechanism:
  - formula: r = a
    formula_description: A circular orbit has constant radius.
  - formula: F = G * M * m / r^2
    formula_description: Newtonian universal gravitation.
  - formula: acc = F / m
    formula_description: Newton's second law.
  - formula: v = sqrt(r * acc)
    formula_description: Speed in uniform circular motion.
  - formula: T = 2 * π * r / v
    formula_description: Time to traverse one circumference.
```

The phenomenological formula and mechanism are both present here because this
is the private authoring definition. They will not both be exposed to Codex.

## 2. Validate, generate, and prepare

Run the authoring pipeline:

```bash
mdbench validate --problems example/problems/circular_orbit.yaml

mdbench synthetic \
  --problems example/problems/circular_orbit.yaml \
  --output-dir example/synthetic_data

mdbench prepare \
  --problems example/problems/circular_orbit.yaml \
  --synthetic-data-dir example/synthetic_data \
  --output-dir example/tasks \
  --task mechanism_discovery \
  --reveal-auxiliary \
  --save-answer
```

The prepared task contains public and private artifacts:

```text
example/tasks/Circular_orbit/
├── problem.json        public
├── data_train.npy      public
├── answer.json         private
├── data_id_test.npy    private
└── data_ood_test.npy   private
```

`--reveal-auxiliary` exposes the planetary mass `m` and its training row. This
lets the Agent express the force and acceleration laws separately even though
`m` cancels from the final period law.

## 3. Create the Agent workspace

Create a fresh temporary directory containing only the public artifacts:

```bash
MDBENCH_AGENT_DIR="$(mktemp -d /tmp/mdbench-circular-orbit-XXXXXX)"
cp example/tasks/Circular_orbit/problem.json "$MDBENCH_AGENT_DIR/"
cp example/tasks/Circular_orbit/data_train.npy "$MDBENCH_AGENT_DIR/"
```

Do not copy the source YAML, answer, or test arrays. Configure the Agent so this
directory is its working root and it cannot access the authoring environment.
Consequently, lifecycle commands cannot find a source `./problems` directory,
and final evaluation has no answer path to consume.

## 4. Let Codex explore

Start Codex in the temporary directory:

```bash
codex \
  -C "$MDBENCH_AGENT_DIR" \
  --sandbox workspace-write \
  --ask-for-approval never \
  'Solve the MDBench mechanism-discovery task in the current directory. Do not
  access files outside this directory. Read problem.json and data_train.npy.
  Submit an explanatory chain of reusable lower-level physical relationships,
  not the phenomenological equation or an algebraic rewrite of it; a
  one-equation fit is invalid even if training-data agreement is 100%. Use a simple
  variable name on every left-hand side and ** for powers. Save one equation
  per line in submission.txt. Iterate with the feedback evaluator:
  mdbench evaluate --evaluation-mode feedback --problem .
  --submission submission.txt --verbose'
```

Codex can inspect the public metadata, analyze the training array, formulate a
candidate mechanism, and use feedback repeatedly. It cannot obtain
reference-mechanism recovery or hidden ID/OOD results from feedback mode.

A Codex run following these steps produced the following valid alternative to
the private reference mechanism:

```text
Fg = G * M * m / a**2
ag = Fg / m
ac = ag
v = (ac * a)**0.5
C = 2 * π * a
T = C / v
```

Codex can check this candidate inside its workspace:

```bash
mdbench evaluate \
  --evaluation-mode feedback \
  --problem . \
  --submission submission.txt \
  --verbose
```

The feedback reports prediction accuracy on public training targets, evaluates
mechanism fundamentality and description complexity, and shows a concise
equation chain. In this run, training prediction accuracy was 100%, the
fundamentality score was 74.50%, and the trace ended with
`T = C/v = 2*a*π/sqrt(G*M/a)`. A relationship already expressed only through
inputs and constants was printed once. Feedback does not compare the expanded
target with the private phenomenological equation or reveal the private
reference mechanism. A direct restatement such as
`T = 2*π*sqrt(a**3/(G*M))` would also fit the data, which is why the Agent must
be instructed to seek explanatory lower-level relationships rather than merely
maximize training-data agreement.

## 5. Evaluate the final submission

After Codex finishes, the benchmark operator evaluates the same submission in
the authoring environment:

```bash
mdbench evaluate \
  --evaluation-mode final \
  --answer example/tasks/Circular_orbit/answer.json \
  --submission "$MDBENCH_AGENT_DIR/submission.txt"
```

Final evaluation additionally reports private derived-equation equivalence,
train/ID/OOD agreement, and recovery of the private reference structure.
Fundamentality is printed as `model @ provider`; its aggregate is shown as
`0.70 * bottleneck + 0.30 * mean`. MDBench keeps these components separate
rather than reducing them to one overall score.

For the alternative mechanism above, train, ID, and OOD tolerance accuracy were
all 100%, and derived-equation equivalence was 100%. Structure recovery was
66.10% because the submission introduces a different set of intermediate
variables and dependencies; the soft structural metric still recognizes much
of the shared physical organization.
