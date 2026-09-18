"""LLM-assisted fundamentality score for submitted mechanism relationships."""
from __future__ import annotations

from typing import Any

import numpy as np

from ..core import Problem
from ..features.validation import LLMMechanismFundamentalityChecker


class LLMMechanismFundamentalityScorer:
    """Score submitted mechanisms independently of reference-list similarity.

    The aggregate score is a weighted combination of the weakest item and the
    item mean, so one non-mechanistic relationship
    cannot be hidden by many good relationships.

    A full ``Problem`` supplies the richest context. ``answer`` is accepted for
    evaluating private answer packages and is converted to the same basic
    context schema without inventing missing problem descriptions.
    """

    def __init__(
        self,
        *,
        llm_provider: str = "deepseek",
        llm_model: str = "deepseek-v4-flash",
        checker=None,
        bottleneck_weight: float = 0.7,
    ):
        if not 0 <= bottleneck_weight <= 1:
            raise ValueError("bottleneck_weight must be between 0 and 1.")
        self.checker = checker or LLMMechanismFundamentalityChecker(
            llm_provider=llm_provider,
            llm_model=llm_model,
        )
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.bottleneck_weight = bottleneck_weight

    @staticmethod
    def _answer_context(
        answer: dict[str, Any],
        mechanisms: list[dict[str, Any]],
    ) -> dict[str, Any]:
        descriptions = {
            item["name"]: item.get("description", "")
            for item in answer.get("intermediate_variables", [])
        }
        descriptions.update(answer.get("variable_descriptions", {}))
        descriptions.update({
            item["name"]: item.get("description", "")
            for item in answer.get("constants", [])
        })
        variable_names = list(answer.get("source_variables", []))
        target = answer.get("target_variable")
        if target:
            variable_names.insert(0, target)
        for item in answer.get("intermediate_variables", []):
            if item["name"] not in variable_names:
                variable_names.append(item["name"])
        return {
            "problem_name": answer.get("problem_name", ""),
            "problem_description": answer.get("problem_description", ""),
            "target_variable": {
                "name": target or "",
                "description": descriptions.get(target, ""),
            },
            "variables": [
                {"name": name, "description": descriptions.get(name, "")}
                for name in variable_names
            ],
            "mechanisms": [
                {
                    "index": index,
                    "equation": item["formula"],
                    "description": item.get("formula_description", ""),
                }
                for index, item in enumerate(mechanisms, 1)
            ],
        }

    def compare(
        self,
        candidate: list[dict[str, Any]],
        *,
        answer: dict[str, Any],
        problem: Problem | None = None,
    ) -> dict[str, Any]:
        context = (
            self.checker.build_context(problem, candidate)
            if problem is not None
            else self._answer_context(answer, candidate)
        )
        report = self.checker.check_context(context)
        items = []
        for judgment, mechanism in zip(report["items"], context["mechanisms"]):
            semantic = judgment["relative_fundamentality"]
            items.append({
                **judgment,
                "score": semantic,
            })
        item_scores = [item["score"] for item in items]
        minimum = float(min(item_scores)) if item_scores else 0.0
        mean = float(np.mean(item_scores)) if item_scores else 0.0
        return {
            "score": self.bottleneck_weight * minimum
            + (1 - self.bottleneck_weight) * mean,
            "minimum_item_score": minimum,
            "mean_item_score": mean,
            "bottleneck_weight": self.bottleneck_weight,
            "items": items,
            "provider": report["provider"],
            "model": report["model"],
        }
