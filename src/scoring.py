"""Shared objective scoring; no LLM judgments or graph similarity metrics."""
from __future__ import annotations
from functools import lru_cache
import json
import pickle
from pathlib import Path
import subprocess
import sys
import numpy as np
import sympy as sp
from .core import VariableSpec
from .validate_problem import ValidationError, NAME, parse_equation, solve_model
from .synthetic_data import evaluate_expression


SYMBOLIC_EQUIVALENCE_TIMEOUT_SECONDS = 30.0
SYMBOLIC_DECIMAL_RELATIVE_TOLERANCE = sp.Rational(1, 10**14)
SYMBOLIC_DECIMAL_MAX_MANTISSA_DENOMINATOR = 10**9


def _decimal_order(value: sp.Rational) -> int:
    """Return floor(log10(abs(value))) without converting to binary float."""
    numerator, denominator = abs(int(value.p)), int(value.q)
    order = len(str(numerator)) - len(str(denominator))
    scale = sp.Integer(10) ** order
    if abs(value) < scale:
        order -= 1
    return order


def _normalize_numeric_atom(value: sp.Expr) -> sp.Expr:
    """Remove insignificant decimal tails while retaining meaningful digits.

    Agent answers commonly contain artifacts such as
    ``0.10000000000000002``.  SymPy correctly treats those literals as
    different exact rationals, but SA is intended to compare mathematical
    formulas rather than binary floating-point serialization artifacts.  We
    therefore seek a simpler representation of the mantissa and accept it
    only within a tight relative tolerance.
    """
    if isinstance(value, sp.Float):
        value = sp.Rational(str(value))
    if not isinstance(value, sp.Rational) or value == 0:
        return value

    def limited_mantissa(rational: sp.Rational) -> sp.Rational:
        order = _decimal_order(rational)
        scale = sp.Integer(10) ** order
        mantissa = rational / scale
        return mantissa.limit_denominator(
            SYMBOLIC_DECIMAL_MAX_MANTISSA_DENOMINATOR) * scale

    # A literal used as a denominator has already become its reciprocal by the
    # time we see the expression, so consider simplifications in both forms.
    candidates = (limited_mantissa(value), 1 / limited_mantissa(1 / value))
    valid = [candidate for candidate in candidates
             if abs(candidate - value)
             <= SYMBOLIC_DECIMAL_RELATIVE_TOLERANCE * abs(value)]
    if not valid:
        return value

    def complexity(rational: sp.Rational) -> tuple[int, int]:
        numerator = str(abs(int(rational.p)))
        denominator = str(int(rational.q))
        significant_digits = (len(numerator.rstrip('0'))
                              + len(denominator.rstrip('0')))
        return significant_digits, len(numerator) + len(denominator)

    candidate = min(valid, key=complexity)
    return candidate if complexity(candidate) < complexity(value) else value


def _normalize_numeric_constants(expression: sp.Expr) -> sp.Expr:
    replacements = {
        atom: normalized
        for atom in expression.atoms(sp.Rational, sp.Float)
        if (normalized := _normalize_numeric_atom(atom)) != atom
    }
    return expression.xreplace(replacements)


@lru_cache(maxsize=128)
def _simplify_is_zero_with_timeout(
        expression: sp.Expr, timeout: float = SYMBOLIC_EQUIVALENCE_TIMEOUT_SECONDS) -> bool:
    """Return SymPy's exact zero proof, terminating it after ``timeout`` seconds."""
    if timeout <= 0:
        return False
    try:
        completed = subprocess.run(
            [sys.executable, '-m', 'src._symbolic_equivalence_worker'],
            input=pickle.dumps(expression, protocol=pickle.HIGHEST_PROTOCOL),
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=timeout, check=False)
        return completed.returncode == 0 and completed.stdout == b'1'
    except (OSError, pickle.PickleError, subprocess.TimeoutExpired):
        return False


def submission_formulas(text: str) -> list[str]:
    # TXT format: one equality per line (or semicolon), optional # comments.
    formulas = []
    for line in text.splitlines():
        line = line.split('#', 1)[0].strip()
        if not line: continue
        for formula in line.split(';'):
            if formula.strip():
                parse_equation(formula.strip())
                formulas.append(formula.strip())
    if not formulas: raise ValidationError('Empty submission.')
    return formulas


def public_variables(problem: dict) -> tuple[list[VariableSpec], str, list[str]]:
    if not isinstance(problem, dict) or problem.get('data_layout') != 'variables_by_samples':
        raise ValidationError('Expected public problem JSON with variables_by_samples layout.')
    items = problem.get('variables')
    if not isinstance(items, list): raise ValidationError('variables must be a list.')
    variables = []
    for item in items:
        if not isinstance(item, dict) or set(item) != {'name', 'description', 'unit', 'role'}:
            raise ValidationError('Public variable fields must be name, description, unit, role.')
        if not isinstance(item['name'], str) or not NAME.fullmatch(item['name']):
            raise ValidationError('Invalid public variable name.')
        if item['role'] not in ('input', 'target', 'auxiliary'):
            raise ValidationError('Public metadata must contain observed variables only.')
        variables.append(VariableSpec(**item))
    columns = problem.get('data_columns')
    if columns != [v.name for v in variables] or len(set(columns)) != len(columns):
        raise ValidationError('data_columns must match unique public variable names in order.')
    targets = [v.name for v in variables if v.role == 'target']
    if len(targets) != 1: raise ValidationError('Exactly one public target is required.')
    return variables, targets[0], columns


def validate_data(data, columns):
    if (not isinstance(data, np.ndarray) or data.ndim != 2 or data.shape[0] != len(columns)
            or data.shape[1] == 0 or data.dtype.kind not in 'fiu' or not np.isfinite(data).all()):
        raise ValidationError('Data must be a finite real (variables, samples) array in data_columns order.')


def accuracy_metrics(predicted, truth) -> dict:
    predicted, truth = np.asarray(predicted, dtype=float), np.asarray(truth, dtype=float)
    if predicted.shape != truth.shape or truth.size == 0 or not np.isfinite(truth).all():
        raise ValidationError('Metric inputs must have matching nonempty shapes and finite truth.')
    valid = np.isfinite(predicted)
    report = {'n_samples': int(truth.size), 'finite_fraction': float(valid.mean()), 'numerically_equivalent': False}
    if not valid.all():
        # Never reward an expression by silently discarding invalid predictions.
        return report | {key: None for key in ('mae', 'mse', 'rmse', 'mape', 'r2', 'nrmse', 'max_absolute_error')}
    with np.errstate(all='ignore'):
        error = predicted - truth
        mse = float(np.mean(error ** 2))
        variance = float(np.var(truth))
        rmse = float(np.sqrt(mse))
        scale = float(np.sqrt(np.mean(truth ** 2)))
        epsilon = np.finfo(float).eps
        scores = {'mae': float(np.mean(np.abs(error))), 'mse': mse, 'rmse': rmse,
                  'mape': float(np.mean(np.abs(error) / np.maximum(np.abs(truth), epsilon))),
                  'r2': (1 - mse / variance) if variance > 0 else (1.0 if mse == 0 else 0.0),
                  'nrmse': rmse / max(scale, epsilon), 'max_absolute_error': float(np.max(np.abs(error)))}
    scores = {key: value if np.isfinite(value) else None for key, value in scores.items()}
    return report | scores | {'numerically_equivalent': bool(np.allclose(predicted, truth, rtol=1e-6, atol=0))}


def _difference_is_zero(predicted: sp.Expr, truth: sp.Expr) -> bool:
    difference = predicted - truth
    # A high-precision counterexample is enough to prove non-equivalence and
    # avoids pathological simplify() calls for decimal approximations of exact
    # transcendental constants.  Expressions that survive still receive the
    # exact symbolic check below; numerical agreement alone is never rewarded.
    symbols = sorted(difference.free_symbols, key=str)
    for offset in range(3):
        substitutions = {symbol: sp.Rational(index + offset + 2)
                         for index, symbol in enumerate(symbols)}
        try:
            value = difference.evalf(50, subs=substitutions)
            if value.is_number and value.is_finite:
                numeric = complex(value)
                if abs(numeric) > 1e-30:
                    return False
        except (TypeError, ValueError, OverflowError):
            continue
    return _simplify_is_zero_with_timeout(difference)


def symbolic_equivalent(predicted: sp.Expr, truth: sp.Expr) -> bool:
    # Normalize independently parsed symbol objects while retaining task assumptions.
    canonical = {str(s): s for s in truth.free_symbols}
    predicted = predicted.xreplace({s: canonical.get(str(s), s) for s in predicted.free_symbols})

    # Never let the approximate decimal cleanup invalidate an identity that is
    # already exact.  Only formulas that fail the original exact comparison get
    # a second chance after insignificant machine tails have been removed.
    if _difference_is_zero(predicted, truth):
        return True
    normalized_predicted = _normalize_numeric_constants(predicted)
    normalized_truth = _normalize_numeric_constants(truth)
    if normalized_predicted == predicted and normalized_truth == truth:
        return False
    return _difference_is_zero(normalized_predicted, normalized_truth)


def score_expression(expression, reference, data, columns):
    values = {name: data[i] for i, name in enumerate(columns)}
    prediction = evaluate_expression(expression, values, data.shape[1])
    truth = evaluate_expression(reference, values, data.shape[1])
    return accuracy_metrics(prediction, truth) | {'symbolically_equivalent': symbolic_equivalent(expression, reference)}


def feedback(problem: dict, train: np.ndarray, submission: str) -> dict:
    variables, target, columns = public_variables(problem)
    validate_data(train, columns)
    sources = [v for v in variables if v.role in ('input', 'auxiliary')]
    solution = solve_model(submission_formulas(submission), sources, required=[target])
    prediction = evaluate_expression(solution[target], {n: train[i] for i, n in enumerate(columns)}, train.shape[1])
    return {'ok': True, 'solution': {name: str(expr) for name, expr in solution.items()},
            'train': accuracy_metrics(prediction, train[columns.index(target)])}
