"""Export public observation tasks and private evaluation answers.

No mechanism, mutation, internal variable, sampling range or probe is exported
into the public problem JSON. NPY rows follow public data_columns.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
from importlib import resources
import json
from pathlib import Path
from .core import Task
from .validate_problem import load_task, discover_tasks, validate_task
from .synthetic_data import generate_synthetic_data
import numpy as np


def task_to_dict(task: Task) -> dict:
    return {'task_name': task.task_name, 'task_description': task.task_description,
            'mutation': task.mutation, 'phenomenal_model': task.phenomenal_model,
            'mechanism_model': [{'formula': m.formula_str, 'role': m.role, 'description': m.description}
                                for m in task.mechanism_model],
            'variables': [{k: v for k, v in vars(variable).items() if v is not None} for variable in task.variables],
            'mechanism_probes': [vars(p) for p in task.mechanism_probes]}


def public_problem(task: Task) -> dict:
    # Even the original/variant suffix can hint at a mutation; expose no task name.
    return {'task_description': task.task_description,
            'variables': [{'name': v.name, 'description': v.description, 'unit': v.unit,
                           'role': 'input' if v.role == 'auxiliary' else v.role}
                          for v in task.observed],
            'data_columns': [v.name for v in task.observed], 'data_layout': 'variables_by_samples'}


def export_task(task: Task, output_dir: str | Path, *, force=False, **sampling) -> dict[str, Path]:
    validate_task(task)
    root = Path(output_dir)
    public, private = root / 'agent', root / 'answer'
    paths = {'problem': public / 'problem.json', 'train': public / 'train.npy',
             'answer': private / 'answer.json'}
    expected = [*paths.values(), *(private / f'{s}.npy' for s in ('train', 'id_test', 'ood_test'))]
    if not force and any(p.exists() for p in expected):
        raise FileExistsError(f'Export exists: {root}; pass --force to overwrite.')
    data = generate_synthetic_data(task, **sampling)
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    paths['problem'].write_text(json.dumps(public_problem(task), indent=2) + '\n')
    np.save(paths['train'], data['train'], allow_pickle=False)
    for split in ('train', 'id_test', 'ood_test'):
        np.save(private / f'{split}.npy', data[split], allow_pickle=False)
    answer = {'task': task_to_dict(task), 'data_columns': [v.name for v in task.observed],
              'data_layout': 'variables_by_samples',
              'data': {split: f'{split}.npy' for split in ('train', 'id_test', 'ood_test')},
              'generation': {'seed': sampling.get('seed', 0)}}
    paths['answer'].write_text(json.dumps(answer, indent=2) + '\n')
    return paths


def get_parser(parser=None):
    parser = parser or argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--problems', nargs='+', default=None, help='Task paths; defaults to the bundled task library.')
    parser.add_argument('--output-dir', default='data/tasks')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    for split in ('train', 'id-test', 'ood-test'):
        parser.add_argument(f'--{split}-samples', type=int, default=1000)
    return parser


def main(args):
    with ExitStack() as stack:
        paths = args.problems or [stack.enter_context(resources.as_file(resources.files('problems')))]
        return _export_collection(args, paths)


def _export_collection(args, paths):
    seen, families = set(), {}
    for path in discover_tasks(paths):
        task = load_task(path)
        validate_task(task, path=path, seen=seen)
        family = task.task_name.rsplit(' - ', 1)[0]
        if family in families and families[family] != task.task_description:
            raise ValueError('Task family descriptions must be identical.')
        families[family] = task.task_description
        result = export_task(task, Path(args.output_dir) / task.task_name, force=args.force,
                             seed=args.seed, train_samples=args.train_samples,
                             id_test_samples=args.id_test_samples, ood_test_samples=args.ood_test_samples)
        print(json.dumps({name: str(p) for name, p in result.items()}))
    return 0

if __name__ == '__main__':
    raise SystemExit(main(get_parser().parse_args()))
