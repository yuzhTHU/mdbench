"""Turn ordered mechanism equations into executable solution steps.

Mechanism equations may contain an implicit system such as
``a = f(x, a, b); b = g(x, a, b)``.  The resolver first attempts a closed
symbolic solution and otherwise evaluates the block with a numerical nonlinear
least-squares solver.  The numerical implementation is isolated here so it can
also serve as the reference used to validate user-provided closed forms later.
"""
from __future__ import annotations
import nd2py as nd
import sympy as sp
import numpy as np
from typing import Any
from itertools import combinations
from scipy.optimize import least_squares
from ...core import Problem, SolutionFunction, SolutionItem


def _variables(formula: nd.Symbol) -> set[str]:
    return {node.name for node in formula.iter_preorder() if isinstance(node, nd.Variable)}


def _sample_count(values: dict[str, Any]) -> int:
    for value in values.values():
        if (array := np.asarray(value)).ndim > 0:
            return len(array)
    else:
        return 1


def _select_positive_solution(
    solutions: list[dict[sp.Symbol, sp.Expr]],
    target_symbols: list[sp.Symbol],
) -> dict[sp.Symbol, sp.Expr] | None:
    """Select the unique all-positive branch under positive free symbols.

    This is the benchmark's default physical-quantity heuristic. If positivity
    cannot identify exactly one complete branch, numerical solving remains the
    fallback instead of relying on SymPy's solution ordering.
    """
    positive = []
    for solution in solutions:
        if any(symbol not in solution for symbol in target_symbols):
            continue
        signs = [sp.posify(solution[symbol])[0].is_positive for symbol in target_symbols]
        if all(sign is True for sign in signs):
            positive.append(solution)
    return positive[0] if len(positive) == 1 else None


def _symbolic_solution(
    target_names: list[str],
    left_expressions: list[nd.Symbol],
    right_expressions: list[nd.Symbol],
) -> list[nd.Symbol] | None:
    """Return one complete nd2py closed form, or ``None`` for numerical solving."""
    if not (len(target_names) == len(left_expressions) == len(right_expressions)):
        raise ValueError("Number of targets, left expressions, and right expressions must match.")
    if (
        len(target_names) == 1
        and isinstance(left_expressions[0], nd.Variable)
        and target_names[0] == left_expressions[0].name
        and target_names[0] not in _variables(right_expressions[0])
    ):
        return right_expressions

    all_names = set()
    for expression in [*left_expressions, *right_expressions]:
        all_names |= _variables(expression)
    if missing := set(target_names) - all_names:
        raise ValueError(f"Missing variables in the system: {missing}")
    
    symbols = {name: sp.Symbol(name) for name in all_names}
    equations = []
    for left, right in zip(left_expressions, right_expressions):
        equations.append(sp.Eq(
            sp.sympify(str(left), locals=symbols), 
            sp.sympify(str(right), locals=symbols)
        ))
    try:
        target_symbols = [symbols[name] for name in target_names]
        solutions = sp.solve(equations, target_symbols, dict=True)
    except Exception:
        return None
    solution = (
        solutions[0]
        if len(solutions) == 1
        else _select_positive_solution(solutions, target_symbols)
    )
    if solution is None:
        return None
    if any(symbols[name] not in solution for name in target_names):
        return None
    try:
        solution_strings = [str(solution[symbols[name]]) for name in target_names]
        solution_formulas = [nd.parse(s) for s in solution_strings]
    except Exception:
        return None
    if any(_variables(f) & set(target_names) for f in solution_formulas):
        return None
    return solution_formulas


def _evaluate_numerically(
    target_names: list[str],
    left_expressions: list[nd.Symbol],
    right_expressions: list[nd.Symbol],
    mechanism_indices: list[int],
    values: dict[str, Any],
    *,
    residual_tolerance: float = 1e-8,
    max_nfev: int = 1000,
) -> list[np.ndarray]:
    """Numerically solve an implicit group independently for every sample.

    The initial value for a target already present in ``values`` is reused. This
    normally seeds the phenomenological target with its expected value; other
    internal variables start at 1. Numerical results are accepted only when the
    scaled equation residual is below ``residual_tolerance``.
    """
    count = _sample_count(values)
    outputs = {name: np.empty(count, dtype=float) for name in target_names}

    for sample_index in range(count):
        context = {
            name: (np.asarray(value)[sample_index] if np.asarray(value).ndim else value)
            for name, value in values.items()
        }
        if any(not np.isfinite(value) for value in context.values()):
            for name in target_names:
                outputs[name][sample_index] = np.nan
            continue
        initial = np.asarray([
            float(context[name]) if name in context and np.isfinite(context[name]) else 1.0
            for name in target_names
        ])
        initially_known = set(context) & set(target_names)
        initial_context = {**context, **dict(zip(target_names, initial))}
        def raw_residual(candidate):
            local = {**context, **dict(zip(target_names, candidate))}
            return np.asarray([
                float(left.eval(local)) - float(right.eval(local))
                for left, right in zip(left_expressions, right_expressions)
            ], dtype=float)

        initial_residual = raw_residual(initial)
        scales = np.maximum(np.abs(initial_residual), 1.0)
        result = least_squares(
            lambda candidate: raw_residual(candidate) / scales,
            initial,
            max_nfev=max_nfev,
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
        )
        residual = raw_residual(result.x) / scales
        if (
            not result.success
            or not np.isfinite(result.x).all()
            or not np.isfinite(residual).all()
            or np.linalg.norm(residual, ord=np.inf) > residual_tolerance
        ):
            indices = ", ".join(map(str, mechanism_indices))
            raise ValueError(
                f"Numerical solution failed for mechanism group [{indices}] at "
                f"sample {sample_index}: {result.message}; maximum scaled residual "
                f"is {np.linalg.norm(residual, ord=np.inf):.3g}."
            )
        for name, value in zip(target_names, result.x):
            outputs[name][sample_index] = value
    return [outputs[name] for name in target_names]


def _as_sample_array(value: Any, count: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 0:
        array = np.full(count, array.item())
    return np.asarray(np.broadcast_to(array, (count,)))


def _explicit_function(formulas: tuple[nd.Symbol, ...]) -> SolutionFunction:
    def evaluate(values: dict[str, Any]) -> list[np.ndarray]:
        count = _sample_count(values)
        return [_as_sample_array(formula.eval(values), count) for formula in formulas]
    return evaluate


def _numerical_function(
    targets: tuple[str, ...],
    left_expressions: tuple[nd.Symbol, ...],
    right_expressions: tuple[nd.Symbol, ...],
    indices: tuple[int, ...],
) -> SolutionFunction:
    def evaluate(values: dict[str, Any]) -> list[np.ndarray]:
        return _evaluate_numerically(
            targets, left_expressions, right_expressions, indices, values
        )
    return evaluate


def solve_mechanism_equations(problem: Problem) -> list[SolutionItem]:
    """Build and attach the ordered mechanism solution to ``problem``.

    At each step, the smallest square subset of the remaining equations is
    selected: it must contain as many equations as variables not yet computable
    from prior steps. The equations in an implicit system need not be adjacent.
    """
    auxiliary_names = {var.name for var in problem.auxiliary_input_variables}
    constant_names = {cons.name for cons in problem.constants}
    input_names = {var.name for var in problem.input_variables}
    known = input_names | constant_names | auxiliary_names

    def unresolved_names(indices: tuple[int, ...], known: set[str]) -> list[str]:
        unresolved = []
        for index in indices:
            item = problem.mechanism[index]
            if item.variable not in known and item.variable not in unresolved:
                unresolved.append(item.variable)
            for node in nd.parse(item.formula).iter_preorder():
                if (
                    isinstance(node, nd.Variable)
                    and node.name not in known
                    and node.name not in unresolved
                ):
                    unresolved.append(node.name)
        return unresolved

    pending = list(range(len(problem.mechanism)))
    solutions: list[SolutionItem] = []
    while pending:
        for index in pending:
            if not unresolved_names((index,), known):
                raise ValueError(
                    f"Mechanism {index + 1} is marked as pending "
                    f"but has no unresolved variables. "
                )

        def loader(pending):
            for size in range(1, len(pending) + 1):
                for selected in combinations(pending, size):
                    yield (size, selected)
            
        for size, selected in loader(pending):
            if size == len(unresolved := unresolved_names(selected, known)):
                break
        else:
            unresolved = unresolved_names(tuple(pending), known)
            relation = "underdetermined" if len(pending) < len(unresolved) else "overdetermined"
            selected = ", ".join(str(idx + 1) for idx in pending)
            raise ValueError(
                f"Remaining mechanisms [{selected}] are {relation}: "
                f"{len(pending)} equations for {len(unresolved)} unresolved "
                f"variables ({', '.join(unresolved)})."
            )

        left_expressions = [nd.Variable(problem.mechanism[idx].variable) for idx in selected]
        right_expressions = [nd.parse(problem.mechanism[idx].formula) for idx in selected]
        solved = _symbolic_solution(unresolved, left_expressions, right_expressions)
        formulas = [str(f) for f in solved] if solved is not None else []
        if solved is not None:
            func = _explicit_function(solved)
        else:
            mechanism_indices = tuple(index + 1 for index in selected)
            func = _numerical_function(unresolved, left_expressions, right_expressions, mechanism_indices)
        solutions.append(SolutionItem(variables=unresolved, formulas=formulas, function=func))
        known.update(unresolved)
        pending = [index for index in pending if index not in selected]

    return solutions  # Leave assignment to the caller.


def evaluate_solution(problem: Problem, values: dict[str, Any]) -> dict[str, Any]:
    """Evaluate ``problem.solution`` in dependency order."""
    if not problem.solution:
        raise ValueError(
            "Problem has no solution. Call solve_mechanism_equations(problem) first."
        )
    result = dict(values)
    for item in problem.solution:
        result.update(zip(item.variables, item.function(result)))
    return result
