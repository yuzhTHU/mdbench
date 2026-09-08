"""Reference-free description complexity for a mechanism list."""
from __future__ import annotations
from typing import Any
from ..features.io.mechanism_equation import parse_mechanism_equation


def formula_complexity(formula: str) -> int:
    """Return the nd2py AST size of a formula's residual expression."""
    return len(parse_mechanism_equation(formula))


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
