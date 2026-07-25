"""Name-invariant structural similarity for mechanism systems."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Any, Protocol

import nd2py as nd
import numpy as np
from scipy.optimize import linear_sum_assignment


class MechanismMatcher(Protocol):
    """Compare submitted and reference mechanism lists."""

    def compare(self, candidate: list[dict], reference: list[dict]) -> dict[str, Any]: ...


def _variables(expression: str) -> set[str]:
    return {
        node.name
        for node in nd.parse(expression.replace("^", "**")).iter_preorder()
        if isinstance(node, nd.Variable)
    }


def _structural_tokens(expression: str) -> tuple[str, ...]:
    """Return a prefix AST signature with anonymous variables and constants.

    Variable identities are local to one formula: repeated occurrences retain
    their equality pattern (``x/x`` differs from ``x/y``), while spelling and
    numeric literal values never enter the signature.
    """
    aliases: dict[str, int] = {}
    tokens = []
    for node in nd.parse(expression.replace("^", "**")).iter_preorder():
        if isinstance(node, nd.Variable):
            aliases.setdefault(node.name, len(aliases))
            tokens.append(f"VAR{aliases[node.name]}")
        elif type(node).__name__ == "Number":
            tokens.append("CONST")
        else:
            tokens.append(type(node).__name__)
    return tuple(tokens)


def _sequence_similarity(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    """Normalized Levenshtein similarity between two formula AST traversals."""
    if not left and not right:
        return 1.0
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (a != b),
            ))
        previous = current
    return 1.0 - previous[-1] / max(len(left), len(right), 1)


@dataclass(frozen=True)
class _Graph:
    formulas: tuple[tuple[str, ...], ...]
    edges: frozenset[tuple[int, int]]


def _graph(items: list[dict]) -> _Graph:
    outputs: dict[str, list[int]] = {}
    parsed: list[tuple[str, str]] = []
    for index, item in enumerate(items):
        equation = str(item.get("formula", ""))
        if equation.count("=") != 1:
            raise ValueError(f"Mechanism equation must contain one '=': {equation!r}")
        left, right = (part.strip() for part in equation.split("=", 1))
        outputs.setdefault(left, []).append(index)
        parsed.append((left, right))
    edges = {
        (producer, consumer)
        for consumer, (_, right) in enumerate(parsed)
        for name in _variables(right)
        for producer in outputs.get(name, [])
    }
    return _Graph(
        tuple(_structural_tokens(right) for _, right in parsed),
        frozenset(edges),
    )


class StructuralMechanismMatcher:
    """Softly match formula structure and the directed dependency graph.

    Mechanisms are represented as directed factor graphs whose nodes are
    relationships and whose edges mean that one relationship produces a
    variable consumed by another. Node attributes are anonymized nd2py ASTs:
    variable names and numeric values are ignored. For normal benchmark-sized
    systems the matcher exhaustively finds the globally optimal partial node
    assignment. Larger systems use an iterative linear-assignment
    approximation. The final scalar
    is a weighted combination of matched-formula similarity and directed-edge
    F1. Unmatched relationships reduce formula coverage.

    This module deliberately isolates the heuristic optimizer so it can later
    be replaced by exact quadratic assignment or learned graph matching without
    changing the evaluator interface.
    """

    def __init__(self, formula_weight: float = 0.6, iterations: int = 8,
                 exact_limit: int = 9):
        if not 0 <= formula_weight <= 1:
            raise ValueError("formula_weight must be between 0 and 1.")
        self.formula_weight = formula_weight
        self.iterations = iterations
        self.exact_limit = exact_limit

    def _mapping_score(
        self, mapping: dict[int, int], local: np.ndarray,
        candidate_graph: _Graph, reference_graph: _Graph,
    ) -> tuple[float, float, float]:
        n, m = local.shape
        formula = sum(local[i, j] for i, j in mapping.items()) / max(n, m)
        mapped_edges = {
            (mapping[a], mapping[b])
            for a, b in candidate_graph.edges
            if a in mapping and b in mapping
        }
        overlap = len(mapped_edges & reference_graph.edges)
        denominator = len(candidate_graph.edges) + len(reference_graph.edges)
        dag = 2 * overlap / denominator if denominator else 1.0
        return self.formula_weight * formula + (1 - self.formula_weight) * dag, formula, dag

    def _exact_mapping(
        self, local: np.ndarray, candidate_graph: _Graph, reference_graph: _Graph,
    ) -> dict[int, int]:
        """Globally maximize the declared formula-plus-DAG objective."""
        n, m = local.shape
        best_score, best = -1.0, {}
        if n <= m:
            mappings = (
                dict(enumerate(reference_indices))
                for reference_indices in permutations(range(m), n)
            )
        else:
            mappings = (
                dict(zip(candidate_indices, range(m)))
                for candidate_indices in permutations(range(n), m)
            )
        for mapping in mappings:
            score, _, _ = self._mapping_score(
                mapping, local, candidate_graph, reference_graph
            )
            if score > best_score:
                best_score, best = score, mapping
        return best

    @staticmethod
    def _neighbors(graph: _Graph, index: int, incoming: bool) -> set[int]:
        return {
            a if incoming else b
            for a, b in graph.edges
            if (b == index if incoming else a == index)
        }

    def compare(self, candidate: list[dict], reference: list[dict]) -> dict[str, Any]:
        candidate_graph, reference_graph = _graph(candidate), _graph(reference)
        n, m = len(candidate), len(reference)
        if not n or not m:
            score = 1.0 if n == m else 0.0
            return {"score": score, "formula_similarity": score,
                    "dag_similarity": score, "matches": []}

        local = np.array([
            [_sequence_similarity(c, r) for r in reference_graph.formulas]
            for c in candidate_graph.formulas
        ])
        if max(n, m) <= self.exact_limit:
            mapping = self._exact_mapping(local, candidate_graph, reference_graph)
            optimizer = "exact"
        else:
            combined = local.copy()
            mapping = {}
            for _ in range(self.iterations):
                rows, columns = linear_sum_assignment(-combined)
                new_mapping = dict(zip(rows.tolist(), columns.tolist()))
                reverse = {value: key for key, value in new_mapping.items()}
                structural = np.zeros_like(local)
                for i in range(n):
                    c_in = self._neighbors(candidate_graph, i, True)
                    c_out = self._neighbors(candidate_graph, i, False)
                    for j in range(m):
                        r_in = self._neighbors(reference_graph, j, True)
                        r_out = self._neighbors(reference_graph, j, False)
                        agreements = [
                            sum(mapping.get(node) in r_in for node in c_in),
                            sum(mapping.get(node) in r_out for node in c_out),
                            sum(reverse.get(node) in c_in for node in r_in),
                            sum(reverse.get(node) in c_out for node in r_out),
                        ]
                        denominator = len(c_in) + len(c_out) + len(r_in) + len(r_out)
                        structural[i, j] = sum(agreements) / denominator if denominator else 1.0
                combined = self.formula_weight * local + (1 - self.formula_weight) * structural
                if new_mapping == mapping:
                    break
                mapping = new_mapping
            optimizer = "iterative"

        score, matched_formula, dag_score = self._mapping_score(
            mapping, local, candidate_graph, reference_graph
        )
        return {
            "score": float(score),
            "formula_similarity": float(matched_formula),
            "dag_similarity": float(dag_score),
            "optimizer": optimizer,
            "matches": [
                {
                    "candidate_index": i,
                    "reference_index": j,
                    "formula_similarity": float(local[i, j]),
                }
                for i, j in sorted(mapping.items())
            ],
        }
