"""Load and validate task-aware benchmark submissions."""
from __future__ import annotations

from pathlib import Path

import nd2py as nd

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
        "plain-text file containing one 'variable = formula' equation per "
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
    known = set(answer["source_variables"]) - {answer.get("target_variable")}
    normalized = []
    block_equations = 0
    unresolved: list[str] = []
    for index, formula in enumerate(formulas, 1):
        if formula.count("=") != 1:
            raise ValueError(
                f"Mechanism {index} must contain exactly one '=': {formula!r}."
            )
        left, right = (part.strip() for part in formula.split("=", 1))
        if not left or not right:
            raise ValueError(f"Invalid mechanism equation: {formula!r}.")
        if not left.isidentifier():
            raise ValueError(
                f"Mechanism {index} left side must be a variable name: {left!r}."
            )
        parsed_expressions = []
        for expression in (left, right):
            try:
                parsed = nd.parse(expression.replace("^", "**"))
            except Exception as exc:
                raise ValueError(
                    f"nd2py cannot parse mechanism {index} expression "
                    f"{expression!r}: {exc}"
                ) from exc
            if not isinstance(parsed, nd.Symbol):
                raise ValueError(
                    f"Mechanism {index} did not produce a symbolic expression: {expression!r}."
                )
            parsed_expressions.append(parsed)
        block_equations += 1
        for parsed in parsed_expressions:
            for node in parsed.iter_preorder():
                if (
                    isinstance(node, nd.Variable)
                    and node.name not in known
                    and node.name not in unresolved
                ):
                    unresolved.append(node.name)
        if block_equations > len(unresolved):
            raise ValueError(
                f"Mechanism {index} introduces no unresolved variable or "
                "overdetermines the current equation block."
            )
        if block_equations == len(unresolved):
            known.update(unresolved)
            block_equations = 0
            unresolved = []
        normalized.append({
            "formula": f"{left} = {right.replace('^', '**')}",
            "formula_description": "",
        })
    if unresolved:
        raise ValueError(
            f"Final mechanism block is underdetermined: {block_equations} equations "
            f"for {len(unresolved)} unresolved variables ({', '.join(unresolved)})."
        )
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
