"""Regenerate mechanism-graph SVGs embedded in the documentation."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core import MechanismItem, Problem, UNIT, VariableSpec
from src.features.io import load_problem, solve_mechanism_equations
from src.features.visualization import MechanismGraphBuilder

OUTPUT_DIR = ROOT / "docs" / "source" / "_static"


def _variable(name: str) -> VariableSpec:
    return VariableSpec(name=name, description=name, unit=UNIT({}))


def _formal_problem(
    name: str,
    target: str,
    inputs: list[str],
    intermediates: list[str],
    formulas: list[tuple[str, str]],
) -> Problem:
    variables = {
        variable: _variable(variable)
        for variable in [target, *inputs, *intermediates]
    }
    problem = Problem(
        problem_name=name,
        problem_description=name,
        phenomenological_formula=target,
        target_variable=variables[target],
        input_variables=[variables[variable] for variable in inputs],
        intermediate_variables=[variables[variable] for variable in intermediates],
        mechanism=[
            MechanismItem(variable, formula, "") for variable, formula in formulas
        ],
    )
    problem.solution = solve_mechanism_equations(problem)
    return problem


def _render(problem: Problem, filename: str, *, constants: bool = True) -> None:
    dot = MechanismGraphBuilder().build(problem, include_constants=constants)
    output = OUTPUT_DIR / filename
    subprocess.run(
        ["dot", "-Tsvg", "-o", str(output)],
        input=dot,
        text=True,
        check=True,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _render(
        _formal_problem(
            "Explicit DAG",
            target="y",
            inputs=["x"],
            intermediates=["a", "b"],
            formulas=[("a", "x**2"), ("b", "a + x"), ("y", "sqrt(b)")],
        ),
        "mechanism_explicit.svg",
    )
    _render(
        _formal_problem(
            "One-variable implicit solution",
            target="y",
            inputs=["x"],
            intermediates=["a"],
            formulas=[("a", "cos(a) + x"), ("y", "2 * a")],
        ),
        "mechanism_implicit_single.svg",
    )
    _render(
        _formal_problem(
            "Coupled implicit solution",
            target="y",
            inputs=["x"],
            intermediates=["a", "b"],
            formulas=[
                ("a", "(x + b) / 2"),
                ("b", "(x + a) / 3"),
                ("y", "a + b"),
            ],
        ),
        "mechanism_implicit_coupled.svg",
    )
    _render(
        load_problem(ROOT / "demo_problem.yaml", solve=True),
        "mechanism_kepler.svg",
    )


if __name__ == "__main__":
    main()
