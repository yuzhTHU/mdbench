"""Load and validate task-aware benchmark submissions."""
from __future__ import annotations
import nd2py as nd
from pathlib import Path
from .mechanism_equation import parse_mechanism_equation
from .solve_mechanism_equations import select_solvable_equation_groups

SYMBOLIC_REGRESSION = "symbolic_regression"
MECHANISM_TASKS = {"mechanism_explanation", "mechanism_discovery"}
SUPPORTED_TASKS = {SYMBOLIC_REGRESSION, *MECHANISM_TASKS}


def _unsupported_structured_submission(path: Path | None = None) -> ValueError:
    subject = (
        f"Structured submission file {str(path)!r} is"
        if path is not None
        else "JSON and YAML submissions are"
    )
    return ValueError(
        f"{subject} not supported. "
        "For a mechanism task, pass equations separated by semicolons, for "
        "example --submission 'r = a; F = G * M * m / r**2', or pass a "
        "plain-text file containing one equation per "
        "non-empty line. Run 'mdbench evaluate --help' for the complete syntax."
    )


def _read_submission_input(value: str, task: str) -> list[str]:
    if not value.strip():
        raise ValueError("Submission must not be empty.")
    path = Path(value)
    if not path.is_file():
        if value.lstrip().startswith(("{", "[")):
            raise _unsupported_structured_submission()
        if task == SYMBOLIC_REGRESSION:
            return [value.strip()]
        return [formula.strip() for formula in value.split(";") if formula.strip()]
    if path.suffix.lower() in {".json", ".yaml", ".yml"}:
        raise _unsupported_structured_submission(path)
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _parse_equation(formula: str, *, require_target: bool, expected_target: str | None = None):
    if not formula:
        raise ValueError("Submission contains an empty formula.")
    if "=" in formula:
        target, rhs = formula.split("=", 1)
        target, rhs = target.strip(), rhs.strip()
        if not target or not target.isidentifier() or not rhs:
            raise ValueError(f"Invalid equation {formula!r}; expected 'target_variable = formula'.")
        if expected_target is not None and target != expected_target:
            raise ValueError(f"Equation target must be {expected_target!r}, got {target!r}.")
    else:
        if require_target:
            raise ValueError(f"Mechanism equation must use 'target_variable = formula': {formula!r}")
        target, rhs = None, formula.strip()
    try:
        expression = nd.parse(rhs.replace("^", "**"))
    except Exception as exc:
        raise ValueError(f"nd2py cannot parse formula {formula!r}: {exc}") from exc
    if not isinstance(expression, nd.Symbol):
        raise ValueError(f"nd2py formula did not produce a symbolic expression: {formula!r}")
    return target, expression


def _validate_mechanisms(formulas: list[str], answer: dict) -> list[dict[str, str]]:
    if "source_variables" not in answer:
        raise ValueError("Answer is missing source_variables required for mechanism validation.")
    normalized = []
    equations = []
    for index, formula in enumerate(formulas, 1):
        formula_str = str(formula).strip()
        try:
            residual = parse_mechanism_equation(formula_str)
        except ValueError as exc:
            raise ValueError(f"Mechanism {index}: {exc}") from exc
        equations.append(residual)
        normalized.append({
            "formula": formula_str,
            "formula_description": "",
        })
    known = set(answer["source_variables"]) - {answer.get("target_variable")}
    select_solvable_equation_groups(equations, known)
    return normalized


def normalize_submission(formulas: list[str], task: str, answer: dict) -> dict:
    """Validate formula strings and return the canonical evaluator schema."""
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task: {task!r}.")
    if not formulas:
        raise ValueError("Submission does not contain any non-empty formulas.")
    if task == SYMBOLIC_REGRESSION:
        if len(formulas) != 1:
            raise ValueError("Symbolic regression requires exactly one phenomenological formula.")
        _parse_equation(
            formulas[0], require_target=False, expected_target=answer.get("target_variable")
        )
        return {"phenomenological_formula": formulas[0]}
    return {"mechanisms": _validate_mechanisms(formulas, answer)}


def load_submission(submission: str, *, task: str, answer: dict) -> dict:
    """Load one formula, semicolon-separated mechanisms, or a plain-text file."""
    if not isinstance(submission, str):
        raise TypeError("Submission must be a string.")
    formulas = _read_submission_input(submission, task)
    return normalize_submission(formulas, task, answer)
