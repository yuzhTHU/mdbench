# Example

The template `demo_problem.yaml` defines a modified electrical-response mechanism
with two hidden-state probes. The library copy has a matching task filename.

```sh
mdbench validate --problems problems
mdbench export --problems problems --output-dir data/tasks --seed 0
python run.py --algorithm codex \
  --problem-file 'data/tasks/Electrical Dissipation - Variant 1-2/agent/problem.json' \
  --answer 'data/tasks/Electrical Dissipation - Variant 1-2/answer/answer.json' \
  --timeout 600 --save-path logs/demo
```

`performance.json` contains elapsed time and objective scores. Probe prompts,
responses and per-probe audit artifacts are under `probe/`. The original saved session is
immutable and each question restores an independent copy.
