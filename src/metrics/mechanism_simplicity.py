"""Reference-free description complexity for a mechanism list."""
from __future__ import annotations

from typing import Any

import nd2py as nd


def formula_complexity(equation: str) -> int:
    """Return ``len(nd2py_expression)`` for the equation's right-hand side."""
    expression = equation.split("=", 1)[-1].strip()
    return len(nd.parse(expression))


class MechanismSimplicityScorer:
    """Measure submitted mechanism size without a reference equation.

    The primary value is mean nd2py AST nodes per relationship; lower is
    simpler. Total and maximum sizes are diagnostics. No normalized score is
    fabricated because there is no natural reference-free upper bound.
    """

    def compare(self, candidate: list[dict[str, Any]]) -> dict[str, Any]:
        item_complexities = [formula_complexity(item["formula"]) for item in candidate]
        total = sum(item_complexities)
        return {
            "mean_ast_nodes_per_item": (
                float(total / len(item_complexities)) if item_complexities else 0.0
            ),
            "total_ast_nodes": total,
            "maximum_ast_nodes": max(item_complexities, default=0),
            "item_count": len(item_complexities),
            "item_complexities": item_complexities,
        }
