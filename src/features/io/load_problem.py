# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""Load a benchmark problem from YAML."""
import yaml
import nd2py as nd
from typing import Any
from pathlib import Path
from ...utils.unit_parser import parse_unit
from ...core import ConstantSpec, MechanismItem, Problem, VariableSpec, UNIT
from .solve_mechanism_equations import solve_mechanism_equations


def _load_variable_spec(row: dict[str, Any]) -> VariableSpec:
    """Normalize YAML values and let ``VariableSpec`` enforce its field schema."""
    try:
        values = dict(row)
        if "name" in values:
            values["name"] = str(values["name"])
        if "description" in values:
            values["description"] = str(values["description"])
        values["unit"] = (
            UNIT(parse_unit(str(values["unit"])))
            if values.get("unit") is not None else None
        )
        if "sampling" in values and values["sampling"] is not None:
            values["sampling"] = dict(values["sampling"])
        return VariableSpec(**values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid variable_description item: {exc}") from exc


def _load_constant_spec(row: dict[str, Any]) -> ConstantSpec:
    """Normalize YAML values and let ``ConstantSpec`` enforce its field schema."""
    try:
        values = dict(row)
        if "name" in values:
            values["name"] = str(values["name"])
        if "description" in values:
            values["description"] = str(values["description"])
        if "unit" in values:
            values["unit"] = UNIT(parse_unit(str(values["unit"])))
        if "value" in values:
            values["value"] = float(values["value"])
        return ConstantSpec(**values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid constants item: {exc}") from exc


def _check_keys(data: dict[str, Any], required_keys: set[str], allowed_keys: set[str]) -> None:
    missing = sorted(required_keys - set(data))
    unexpected = sorted(set(data) - allowed_keys)
    if missing or unexpected:
        if missing and unexpected:
            info = f"Missing required keys: {', '.join(missing)}. Unexpected keys: {', '.join(unexpected)}."
        elif missing:
            info = f"Missing required keys: {', '.join(missing)}."
        else:
            info = f"Unexpected keys: {', '.join(unexpected)}."
        raise ValueError(info)


def _check_formula(formula_str: str) -> None:
    """Check if the formula string is valid."""
    try:
        formula_str = formula_str.strip().replace("^", "**")
        formula = nd.parse(formula_str)
    except Exception as exc:
        raise ValueError(f"Invalid formula: {formula_str}. Error: {exc}") from exc


def load_problem(problem_path: str, solve=True) -> Problem:
    path = Path(problem_path)
    if not path.is_file():
        raise FileNotFoundError(f"Cannot find the problem file: {problem_path}")
    with path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file)

    required_keys = {
        "problem_name",
        "problem_description",
        "phenomenological_formula",
        "variable_description",
        "mechanism",
    }
    allowed_keys = {*required_keys, "constants"}
    _check_keys(data, required_keys, allowed_keys)

    variable_description = data["variable_description"]
    required_keys = {"target", "inputs", "intermediates", "auxiliary_inputs"}
    if not isinstance(variable_description, dict):
        raise ValueError("variable_description must be a mapping.")
    _check_keys(variable_description, required_keys, required_keys)

    target_variable = _load_variable_spec(variable_description["target"])
    input_variables = [_load_variable_spec(row) for row in variable_description["inputs"]]
    intermediate_variables = [_load_variable_spec(row) for row in variable_description["intermediates"]]
    auxiliary_input_variables = [_load_variable_spec(row) for row in variable_description["auxiliary_inputs"]]
    variables = [
        target_variable,
        *input_variables,
        *intermediate_variables,
        *auxiliary_input_variables,
    ]
    constants = [_load_constant_spec(row) for row in data.get("constants", [])]
    names = [var.name for var in variables + constants]
    if len(names) != len(set(names)):
        duplicated = [name for name in names if names.count(name) > 1]
        raise ValueError(
            f"Duplicate variable or constant names found in the problem file: "
            f"{', '.join(set(duplicated))}"
        )

    mechanisms = []
    for row in data["mechanism"]:
        required_keys = {"formula", "formula_description"}
        _check_keys(row, required_keys, required_keys)
        formula_text = str(row["formula"])
        _variable, _formula = formula_text.split("=", 1)
        _variable = _variable.strip()
        _formula = _formula.strip().replace("^", "**")
        if not _variable.isidentifier():
            raise ValueError(
                f"Invalid mechanism variable name: {_variable!r}. "
                f"Expected a valid Python identifier."
            )
        if not _variable or not _formula:
            raise ValueError(
                f"Invalid mechanism formula: {formula_text}. "
                f"Expected format: 'Variable = Formula'."
            )
        _check_formula(_formula)
        mechanism = MechanismItem(
            variable=_variable,
            formula=_formula,
            formula_description=str(row["formula_description"]),
        )
        mechanisms.append(mechanism)

    variable, formula = str(data["phenomenological_formula"]).split("=", 1)
    variable = variable.strip()
    formula = formula.strip().replace("^", "**")
    _check_formula(formula)
    if variable != target_variable.name:
        raise ValueError(
            "Phenomenological formula target does not match variable_description.target: "
            f"{variable!r} != {target_variable.name!r}."
        )
    problem = Problem(
        problem_name=str(data["problem_name"]),
        problem_description=str(data["problem_description"]),
        phenomenological_formula=str(formula),
        target_variable=target_variable,
        input_variables=input_variables,
        intermediate_variables=intermediate_variables,
        mechanism=mechanisms,
        auxiliary_input_variables=auxiliary_input_variables,
        constants=constants,
    )
    if solve:
        problem.solution = solve_mechanism_equations(problem)
    return problem
