"""Mechanism-to-phenomenological-equation derivation checks."""
from __future__ import annotations

from typing import Any, Protocol
import nd2py as nd
import numpy as np

from ...core import Problem
from ..io import evaluate_solution, solve_mechanism_equations


class MechanismDerivationChecker(Protocol):
    """Decide whether ordered mechanism equations derive their target formula.

    Input is a parsed ``Problem``. Output is a JSON-compatible report with
    ``equivalent`` and diagnostic detail. Implementations may use numeric
    testing, symbolic elimination, or an LLM/CAS hybrid.
    """

    def check(
        self, problem: Problem
    ) -> dict[str, Any]: ...


class NumericDerivationChecker:
    """Check resolved mechanism steps at deterministic positive random points."""

    def __init__(self, samples: int = 20, seed: int = 20260713):
        self.samples, self.seed = samples, seed

    def check(
        self, problem: Problem
    ) -> dict[str, Any]:
        names = {
            node.name
            for node in nd.parse(problem.phenomenological_formula).iter_preorder()
            if isinstance(node, nd.Variable)
        }
        for item in problem.mechanism:
            names.add(item.variable)
            names |= {
                node.name for node in nd.parse(item.formula).iter_preorder()
                if isinstance(node, nd.Variable)
            }
        names -= {
            problem.target_variable.name,
            *(variable.name for variable in problem.intermediate_variables),
        }
        rng = np.random.default_rng(self.seed)
        variables = [*problem.input_variables, *problem.auxiliary_input_variables]
        input_specs = {variable.name: variable for variable in variables}
        values = {}
        for name in sorted(names):
            if (specification := input_specs.get(name)) is None:
                values[name] = 10 ** rng.uniform(-1, 1, self.samples)
            else:
                lower = float(specification.sampling["min"])
                upper = float(specification.sampling["max"])
                if specification.sampling.get("distribution") == "log_uniform":
                    values[name] = 10 ** rng.uniform(np.log10(lower), np.log10(upper), self.samples)
                else:
                    values[name] = rng.uniform(lower, upper, self.samples)
        values.update({constant.name: constant.value for constant in problem.constants})
        expected = np.asarray(nd.parse(problem.phenomenological_formula).eval(values))
        if expected.ndim == 0:
            expected = np.full(self.samples, expected.item())
        values[problem.target_variable.name] = expected
        if not problem.solution:
            problem.solution = solve_mechanism_equations(problem)
        produced_variables = {
            variable
            for item in problem.solution
            for variable in item.variables
        }
        if problem.target_variable.name not in produced_variables:
            return {
                "equivalent": False,
                "tested": 0,
                "detail": (
                    "Mechanism solution does not produce target variable "
                    f"{problem.target_variable.name!r}."
                ),
            }
        values = evaluate_solution(problem, values)
        actual = values[problem.target_variable.name]
        valid = np.isfinite(expected) & np.isfinite(actual)
        tested = int(np.count_nonzero(valid))
        close = np.isclose(actual[valid], expected[valid], rtol=1e-8, atol=1e-10)
        equivalent = bool(tested > 0 and close.all())
        return {
            "equivalent": equivalent,
            "score": float(close.mean()) if tested else 0.0,
            "tested": tested,
            "detail": (
                f"Mechanism derivation tested at {tested}/{self.samples} "
                "finite numeric points."
            ),
        }
