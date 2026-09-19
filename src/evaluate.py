"""Objective phenomenal and independent mechanism-probe evaluation."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from typing import Callable
import numpy as np
import sympy as sp
from .core import Task
from .scoring import (accuracy_metrics, symbolic_equivalent, score_expression,
                      validate_data)
from .synthetic_data import evaluate_expression
from .validate_problem import (ValidationError, task_from_dict, validate_task, solve_model,
                               expand_expression, parse_equation, parse_expression,
                               split_equation, symbols_for)


def load_answer(path: str | Path):
    path = Path(path)
    answer = json.loads(path.read_text())
    task = task_from_dict(answer['task'])
    validate_task(task)
    columns = [v.name for v in task.observed]
    if answer.get('data_columns') != columns or answer.get('data_layout') != 'variables_by_samples':
        raise ValidationError('Private answer data layout/columns do not match task.')
    arrays = {}
    for split in ('train', 'id_test', 'ood_test'):
        arrays[split] = np.load(path.parent / answer['data'][split], allow_pickle=False)
        validate_data(arrays[split], columns)
    return task, arrays, columns


def format_probe(task: Task, probe, submission: list[str]) -> str:
    source_names = [v.name for v in task.by_role('input', 'auxiliary')]
    submitted_names = set().union(*(map(str, parse_equation(formula).free_symbols)
                                    for formula in submission))
    internal_names = sorted(submitted_names - set(source_names) - {task.target.name})
    variables = ', '.join([*source_names, *internal_names])
    submitted_model = '\n'.join(submission)
    return f'''Frozen submitted mechanism model:
{submitted_model}

Using only the mechanism model you submitted above,
derive {probe.probe} ({probe.description}) as a function of the variables: {variables}.

Return exactly one equation in the form:
{probe.probe} = <expression>

The right-hand side may contain only the given variables (including internal
variables already defined in the frozen submitted mechanism model), numeric literals,
and mathematical operators/functions supported by the benchmark.
Do not introduce new variables, additional equations, or prose.
Do not modify, replace, or extend your previously submitted mechanism model.
Do not use any tools. Answer directly from the frozen submitted model.
'''


def expand_probe_reply(formula: str, probe_name: str, sources, solution) -> sp.Expr:
    left, right = split_equation(formula)
    if left != probe_name: raise ValidationError(f'Probe response must put {probe_name} on the left.')
    symbols = symbols_for(sources)
    expression = parse_expression(right, symbols)
    # Never use true internal states to repair a submitted probe expression.
    expression = sp.simplify(expression.subs({sp.Symbol(n, real=True): e for n, e in solution.items()}))
    allowed = {symbols[v.name] for v in sources}
    if expression.free_symbols - allowed:
        raise ValidationError('Probe expression cannot be expanded into input and auxiliary variables using the submitted model.')
    return expression


def evaluate(args, answer_file: str | Path, submission: list[str],
             make_ask: Callable[[], Callable]) -> dict:
    task, arrays, columns = load_answer(answer_file)
    sources = task.by_role('input', 'auxiliary')
    solution = {}
    phenomenal = {}
    try:
        solution = solve_model(submission, sources, required=[task.target.name])
        predicted = solution[task.target.name]
        for split, data in arrays.items():
            values = {name: data[i] for i, name in enumerate(columns)}
            phenomenal[split] = accuracy_metrics(evaluate_expression(predicted, values, data.shape[1]),
                                                 data[columns.index(task.target.name)])
            phenomenal[split]['symbolically_equivalent'] = symbolic_equivalent(predicted, task.solution[task.target.name])
        phenomenal['ok'] = True
    except Exception as exc:
        phenomenal = {'ok': False, 'error': str(exc), 'error_type': type(exc).__name__}
    output = Path(args.save_path) / 'probe'
    def ask_probe(item):
        index, probe = item
        question = format_probe(task, probe, submission)
        try:
            # The caller-owned factory restores an independent copy of the same
            # frozen state; evaluate never needs to know its checkpoint format.
            ask = make_ask()
            if not callable(ask): raise TypeError('Probe ask factory must return a callable.')
            # Algorithms may use the explicit probe metadata to construct a custom
            # request, but using the supplied prompt directly is recommended so
            # probe evaluation remains comparable across algorithms.
            formula = ask(question, probe.probe, probe.description,
                          output_dir=output / f'{index:03d}-{probe.probe}')
            predicted = expand_probe_reply(formula, probe.probe, sources, solution)
            reference = expand_expression(probe.answer, task, lhs=probe.probe)
            return {'probe': probe.probe, 'ok': True, 'reply': formula, 'expression': str(predicted),
                    'scores': {split: score_expression(predicted, reference, data, columns)
                               for split, data in arrays.items()}}
        except Exception as exc:
            return {'probe': probe.probe, 'ok': False, 'error': str(exc), 'error_type': type(exc).__name__}
    workers = getattr(args, 'probe_workers', 1)
    if workers < 1: raise ValueError('probe_workers must be positive.')
    with ThreadPoolExecutor(max_workers=workers) as executor:
        probes = list(executor.map(ask_probe, enumerate(task.mechanism_probes)))
    rates = {}
    for split in arrays:
        rates[split] = {metric: (sum(bool(p.get('scores', {}).get(split, {}).get(metric, False)) for p in probes) / len(probes)
                                if probes else None)
                        for metric in ('symbolically_equivalent', 'numerically_equivalent')}
    return {'task_name': task.task_name, 'phenomenal': phenomenal,
            'submitted_solution': {name: str(e) for name, e in solution.items()},
            'mechanism_probes': probes, 'mechanism_recovery': rates,
            'probe_count': len(probes)}
