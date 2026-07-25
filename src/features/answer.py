"""Construct private benchmark answers from problem definitions."""
from __future__ import annotations

from ..core import Problem


def build_answer(problem: Problem, task: str) -> dict:
    """Return a JSON-compatible answer preserving all constant metadata."""
    produced = {variable.name for variable in problem.intermediate_variables}
    source_variables = [variable.name for variable in problem.input_variables]
    source_variables.extend(
        variable.name for variable in problem.auxiliary_input_variables
        if variable.name not in produced
    )
    source_variables.extend(constant.name for constant in problem.constants)
    answer = {
        "task": task,
        "target_variable": problem.target_variable.name,
        "data_variables": [
            problem.target_variable.name,
            *(variable.name for variable in problem.input_variables),
            *(variable.name for variable in problem.auxiliary_input_variables),
        ],
        "source_variables": source_variables,
        "phenomenological_formula": (
            f"{problem.target_variable.name} = {problem.phenomenological_formula}"
        ),
        "constants": [{
            "name": constant.name,
            "value": constant.value,
            "description": constant.description,
            "unit": constant.unit.to_dict(),
        } for constant in problem.constants],
    }
    if task != "symbolic_regression":
        answer["mechanisms"] = [{
            "formula": item.equation,
            "formula_description": item.formula_description,
        } for item in problem.mechanism]
        answer["intermediate_variables"] = [{
            "name": variable.name,
            "description": variable.description,
            "unit": variable.unit.to_dict() if variable.unit is not None else {},
        } for variable in problem.intermediate_variables]
    return answer
