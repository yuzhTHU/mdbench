"""Phenomenological-formula similarity metrics."""
from __future__ import annotations

from typing import Any, Protocol
import nd2py as nd
import numpy as np


class FormulaEquivalenceChecker(Protocol):
    """Compare two equation strings under named numeric inputs.

    The result contains at least ``equivalent`` and a score in ``[0, 1]``.
    This interface may later be implemented by a CAS or another judge.
    """

    def compare(
        self, candidate: str, reference: str, values: dict[str, np.ndarray]
    ) -> dict[str, Any]: ...


class NumericEquivalenceChecker:
    """Deterministic finite-sample formula comparison using nd2py."""

    def __init__(self, rtol: float = 1e-6, atol: float = 1e-9):
        self.rtol, self.atol = rtol, atol

    @staticmethod
    def _rhs(formula: str) -> str:
        return formula.split("=", 1)[-1].strip().replace("^", "**")

    def compare(self, candidate: str, reference: str, values: dict[str, np.ndarray]) -> dict[str, Any]:
        try:
            actual = np.asarray(nd.parse(self._rhs(candidate)).eval(values), dtype=float)
            expected = np.asarray(nd.parse(self._rhs(reference)).eval(values), dtype=float)
            actual, expected = np.broadcast_arrays(actual, expected)
            finite = np.isfinite(actual) & np.isfinite(expected)
            if not finite.any():
                return {"equivalent": False, "score": 0.0, "detail": "No shared finite evaluation points."}
            close = np.isclose(actual[finite], expected[finite], rtol=self.rtol, atol=self.atol)
            return {
                "equivalent": bool(close.all()),
                "score": float(close.mean()),
                "tested": int(close.size),
            }
        except Exception as exc:
            return {"equivalent": False, "score": 0.0, "detail": f"Formula evaluation failed: {exc}"}
