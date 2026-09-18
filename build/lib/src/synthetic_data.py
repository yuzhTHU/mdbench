"""Generate reproducible feature-by-sample ID/OOD arrays for proposal tasks."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import sympy as sp
from .core import Task
from .validate_problem import load_task, validate_task, discover_tasks, ValidationError


def evaluate_expression(expression: sp.Expr, values: dict[str, np.ndarray], count: int) -> np.ndarray:
    symbols = sorted(expression.free_symbols, key=str)
    if missing := {str(s) for s in symbols} - values.keys():
        raise ValidationError(f'Missing numerical inputs: {sorted(missing)}')
    with np.errstate(all='ignore'):
        result = sp.lambdify(symbols, expression, modules='numpy')(*[values[str(s)] for s in symbols])
        array = np.asarray(result)
        if np.iscomplexobj(array):
            array = np.where(array.imag == 0, array.real, np.nan)
        try:
            return np.array(np.broadcast_to(array, (count,)), dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValidationError(f'Expression returned invalid shape/type: {array.shape}') from exc


def generate_synthetic_data(task: Task, *, seed=0, train_samples=1000,
                            id_test_samples=1000, ood_test_samples=1000) -> dict[str, np.ndarray]:
    if any(type(n) is not int or n <= 0 for n in (train_samples, id_test_samples, ood_test_samples)):
        raise ValidationError('Sample counts must be positive integers.')
    validate_task(task)
    rng = np.random.default_rng(seed)
    sources = task.by_role('input', 'auxiliary')

    def split(count, ood):
        chunks = []
        accepted = 0
        for _ in range(50):
            if accepted >= count: break
            batch = max(1024, 2 * (count - accepted))
            inputs = {}
            for v in sources:
                spec = v.sampling
                low, high = (spec['ood_boundary'], spec['max']) if ood else (spec['min'], spec['ood_boundary'])
                inputs[v.name] = (np.exp(rng.uniform(np.log(low), np.log(high), batch))
                                 if spec['distribution'] == 'log_uniform' else rng.uniform(low, high, batch))
            derived = {name: evaluate_expression(e, inputs, batch) for name, e in task.solution.items()}
            valid = np.ones(batch, dtype=bool)
            for array in derived.values(): valid &= np.isfinite(array)
            indices = np.flatnonzero(valid)[:count - accepted]
            values = inputs | derived
            chunks.append(np.vstack([values[v.name][indices] for v in task.observed]))
            accepted += len(indices)
        if accepted < count:
            raise ValidationError(f'Generated only {accepted}/{count} finite real samples; review sampling ranges.')
        return np.concatenate(chunks, axis=1)

    return {'train': split(train_samples, False), 'id_test': split(id_test_samples, False),
            'ood_test': split(ood_test_samples, True),
            'variables': np.asarray([v.name for v in task.observed])}


def get_parser(parser=None):
    parser = parser or argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--problems', nargs='+', default=['problems'])
    parser.add_argument('--output-dir', default='data/synthetic_data')
    parser.add_argument('--seed', type=int, default=0)
    for split in ('train', 'id-test', 'ood-test'):
        parser.add_argument(f'--{split}-samples', type=int, default=1000)
    return parser


def main(args):
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    seen = set()
    for path in discover_tasks(args.problems):
        task = load_task(path)
        validate_task(task, path=path, seen=seen)
        data = generate_synthetic_data(task, seed=args.seed, train_samples=args.train_samples,
                                       id_test_samples=args.id_test_samples, ood_test_samples=args.ood_test_samples)
        np.savez_compressed(root / f'{task.task_name}.npz', **data)
    return 0

if __name__ == '__main__':
    raise SystemExit(main(get_parser().parse_args()))
