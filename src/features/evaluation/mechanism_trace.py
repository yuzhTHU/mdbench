"""Explain how a submitted mechanism list is solved into its target formula."""
from __future__ import annotations

from typing import Any

import nd2py as nd
import sympy as sp

from ...core import ConstantSpec, MechanismItem, Problem, UNIT, VariableSpec
from ..io import solve_mechanism_equations


def _unit(value: Any) -> UNIT:
    return UNIT(dict(value or {}))


def build_submission_problem(
    mechanisms: list[dict[str, Any]],
    answer: dict[str, Any],
) -> Problem:
    """Construct the minimal Problem needed by the shared mechanism solver.

    The private answer format intentionally omits full sampling metadata. This
    adapter uses only basic names, formulas, constant values, descriptions, and
    units; they are sufficient for symbolic/implicit dependency resolution.
    """
    target_name = answer["target_variable"]
    constants = [
        ConstantSpec(
            name=item["name"],
            description=item.get("description", ""),
            unit=_unit(item.get("unit")),
            value=item["value"],
        )
        for item in answer.get("constants", [])
    ]
    constant_names = {item.name for item in constants}
    source_names = [
        name for name in answer.get("source_variables", [])
        if name != target_name and name not in constant_names
    ]
    equations = []
    mentioned_names = set(source_names) | constant_names | {target_name}
    for item in mechanisms:
        equation = item["formula"]
        left, right = (part.strip() for part in equation.split("=", 1))
        equations.append(MechanismItem(
            variable=left,
            formula=right,
            formula_description=item.get("formula_description", ""),
        ))
        mentioned_names.add(left)
        mentioned_names.update(
            node.name
            for node in nd.parse(right.replace("^", "**")).iter_preorder()
            if isinstance(node, nd.Variable)
        )

    reference_descriptions = {
        item["name"]: item
        for item in answer.get("intermediate_variables", [])
    }
    intermediate_names = sorted(
        mentioned_names - set(source_names) - constant_names - {target_name}
    )
    phenomenological = answer.get("phenomenological_formula", target_name)
    if "=" in phenomenological:
        _, phenomenological = phenomenological.split("=", 1)
    return Problem(
        problem_name=answer.get("problem_name", "Submission diagnostic"),
        problem_description=answer.get("problem_description", ""),
        phenomenological_formula=phenomenological.strip(),
        target_variable=VariableSpec(target_name, "target variable", None),
        input_variables=[
            VariableSpec(name, "source variable", None) for name in source_names
        ],
        intermediate_variables=[
            VariableSpec(
                name,
                reference_descriptions.get(name, {}).get("description", ""),
                _unit(reference_descriptions[name].get("unit"))
                if name in reference_descriptions else None,
            )
            for name in intermediate_names
        ],
        mechanism=equations,
        constants=constants,
    )


def _expanded_formulas(solution) -> dict[str, str]:
    """Substitute earlier closed forms into later closed-form solution steps."""
    expanded: dict[sp.Symbol, sp.Expr] = {}
    rendered: dict[str, str] = {}
    for item in solution:
        for variable, formula in zip(item.variables, item.formulas):
            names = {
                node.name
                for node in nd.parse(formula).iter_preorder()
                if isinstance(node, nd.Variable)
            } | {variable}
            symbols = {name: sp.Symbol(name) for name in names}
            expression = sp.sympify(formula, locals=symbols).subs(expanded)
            expression = sp.simplify(expression)
            expanded[sp.Symbol(variable)] = expression
            rendered[variable] = str(expression)
    return rendered


def trace_mechanism_submission(
    mechanisms: list[dict[str, Any]],
    answer: dict[str, Any],
) -> dict[str, Any]:
    """Return a JSON-compatible mechanism-solving trace for verbose reports.

    Output contains the normalized submitted equations, ordered solver steps,
    closed forms after recursively substituting prior steps, and the final
    target formula when symbolic resolution succeeds. Numerical implicit steps
    are identified explicitly because they have no portable formula string.
    """
    problem = build_submission_problem(mechanisms, answer)
    solution = solve_mechanism_equations(problem)
    expanded = _expanded_formulas(solution)
    submitted = {item.variable: item.formula for item in problem.mechanism}
    base_names = {
        variable.name for variable in problem.input_variables
    } | {constant.name for constant in problem.constants}
    steps = []
    for item in solution:
        original_formulas = [submitted[variable] for variable in item.variables]
        steps.append({
            "variables": list(item.variables),
            "formulas": list(item.formulas),
            "original_formulas": original_formulas,
            "expanded_formulas": [
                expanded.get(variable) for variable in item.variables
            ] if item.formulas else [],
            "show_expanded_formulas": [
                any(
                    isinstance(node, nd.Variable) and node.name not in base_names
                    for node in nd.parse(original).iter_preorder()
                )
                and expanded_formula is not None
                and all(
                    not isinstance(node, nd.Variable) or node.name in base_names
                    for node in nd.parse(expanded_formula).iter_preorder()
                )
                for original, expanded_formula in zip(
                    original_formulas,
                    [expanded.get(variable) for variable in item.variables],
                )
            ],
            "numerical": not bool(item.formulas),
        })
    target = answer["target_variable"]
    return {
        "submitted_equations": [item.equation for item in problem.mechanism],
        "solution_steps": steps,
        "target_variable": target,
        "derived_formula": expanded.get(target),
        "uses_numerical_solver": any(step["numerical"] for step in steps),
    }
