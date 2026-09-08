"""Parsing helpers for fundamental mechanism relationships."""
from __future__ import annotations
import nd2py as nd


def split_mechanism_equation(formula: str) -> tuple[str, str]:
    """Return the two non-empty sides of a relationship with one equals sign."""
    formula = str(formula).strip()
    if formula.count("=") != 1:
        raise ValueError(f"Mechanism equation must contain exactly one '=': {formula!r}.")
    left, right = formula.split("=", 1)
    left = left.strip()
    right = right.strip()
    if not left or not right:
        raise ValueError(f"Invalid mechanism equation: {formula!r}.")
    return left, right


def parse_mechanism_equation(formula: str) -> nd.Symbol:
    """Parse a relationship as its ``lhs - rhs`` residual."""
    original = str(formula).strip()
    left, right = split_mechanism_equation(original)
    try:
        left_expression = nd.parse(left.replace("^", "**"))
        right_expression = nd.parse(right.replace("^", "**"))
        residual = left_expression - right_expression
    except Exception as exc:
        raise ValueError(f"nd2py cannot parse mechanism equation {original!r}: {exc}") from exc
    if not isinstance(residual, nd.Symbol):
        raise ValueError(f"Mechanism equation did not produce a symbolic residual: {original!r}.")
    return residual
