# Example

The template `demo_task.yaml` defines a modified electrical-response mechanism with two hidden-state probes. The commands below use one task from the nested library.

```sh
mdbench validate \
  --tasks 'tasks/astronomy/binary_mass_function/Astronomy Binary Mass Function - Original.yaml'

mdbench export \
  --tasks 'tasks/astronomy/binary_mass_function/Astronomy Binary Mass Function - Original.yaml' \
  --output-dir data/tasks \
  --seed 0

mdbench run \
  --algorithm codex \
  --problem-file 'data/tasks/Astronomy Binary Mass Function - Original/problem/problem.json' \
  --answer-file 'data/tasks/Astronomy Binary Mass Function - Original/answer/answer.json' \
  --timeout 600 \
  --save-path logs/demo
```

`performance.json` contains elapsed time and objective scores. Probe prompts, responses and per-probe audit artifacts are under `probe/`. The original saved session is immutable and each question restores an independent copy.
