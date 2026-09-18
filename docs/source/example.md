# Example

The template `demo_problem.yaml` defines a modified electrical-response mechanism
with two hidden-state probes. The library copy has a matching task filename.

```sh
mdbench validate --problems problems
mdbench export --problems problems --output-dir data/tasks --seed 0
python run.py --algorithm codex \
  --problem-file 'data/tasks/Electrical Dissipation - Variant 1-2/agent/problem.json' \
  --train-data-npy-file 'data/tasks/Electrical Dissipation - Variant 1-2/agent/train.npy' \
  --answer 'data/tasks/Electrical Dissipation - Variant 1-2/answer/answer.json' \
  --timeout 600 --save-path logs/demo
```

`performance.json` contains elapsed time and objective scores. Probe prompts,
responses, events and status are under `probes/`. The original saved session is
immutable and each question restores an independent copy.
