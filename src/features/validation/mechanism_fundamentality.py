"""LLM-based judgment of whether mechanism relationships are fundamental."""
from __future__ import annotations

import json
from typing import Any, Protocol

from dotenv import load_dotenv

from ...core import Problem
from ...utils.llm import LLMAPI


JUDGMENTS = {"fundamental", "not_fundamental", "uncertain"}


class MechanismFundamentalityChecker(Protocol):
    """Judge whether candidate relationships provide mechanistic explanation.

    Input is a loaded ``Problem``. Output is a JSON-compatible mapping with an
    ``items`` list. Each item contains ``index``, ``judgment``,
    ``relative_fundamentality``, ``reason``, and
    ``preferred_relation``. Implementations may use an LLM, curated rules,
    retrieval, or a future physics knowledge base.
    """

    def check(self, problem: Problem) -> dict[str, Any]: ...


class LLMMechanismFundamentalityChecker:
    """Ask an LLM to review all mechanism relationships in one problem."""

    def __init__(
        self,
        *,
        llm_provider: str = "deepseek",
        llm_model: str = "deepseek-v4-flash",
        api=None,
        max_retries: int = 3,
    ):
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.api = api
        self.max_retries = max_retries

    def _get_api(self):
        if self.api is None:
            load_dotenv()
            self.api = LLMAPI.create(
                llm_provider=self.llm_provider,
                llm_model=self.llm_model,
            )
        return self.api

    @staticmethod
    def build_context(
        problem: Problem,
        mechanisms: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Build the shared validation/evaluation context from a full problem."""
        variables = [
            {
                "name": variable.name,
                "description": variable.description,
            }
            for variable in problem.all_variables
        ]
        if mechanisms is None:
            mechanisms = [
                {
                    "formula": item.equation,
                    "formula_description": item.formula_description,
                }
                for item in problem.mechanism
            ]
        return {
            "problem_name": problem.problem_name,
            "problem_description": problem.problem_description,
            "target_variable": {
                "name": problem.target_variable.name,
                "description": problem.target_variable.description,
            },
            "variables": variables,
            "mechanisms": [
                {
                    "index": index,
                    "equation": item["formula"],
                    "description": item.get("formula_description", ""),
                }
                for index, item in enumerate(mechanisms, 1)
            ],
        }

    @staticmethod
    def _prompt(context: dict[str, Any]) -> str:
        return (
            "For each equation, judge whether it is an acceptable fundamental "
            "building block of the proposed physical mechanism. Give a high score "
            "to a recognized physical or mathematical relationship, or to a "
            "scientifically meaningful modeling assumption stated by the problem. "
            "This includes laws, balance relations, constitutive laws, boundary "
            "conditions, geometry, and standard definitions. Equivalent algebraic "
            "forms count as the same relationship.\n\n"
            "Give a low score to a physically invalid equation, a mere renaming or "
            "identity, an arbitrary algebraic helper, or an unexplained numerical "
            "fit. A one-equation chain that directly maps the problem inputs to the "
            "target is an end-result rather than a mechanism if that equation "
            "combines multiple more basic relationships. An equation may use "
            "quantities defined earlier and still express one basic law, balance, or "
            "definition; do not call it bundled for that reason alone. Use the "
            "problem description, equation descriptions, and the whole chain to "
            "interpret each equation.\n\n"
            "Return only a JSON object with key 'items'. Return one item per equation, "
            "in order, with: index, judgment ('fundamental', 'not_fundamental', or "
            "'uncertain'), relative_fundamentality (0 to 1), reason (one short "
            "sentence), and preferred_relation (a more basic "
            "equation, or null).\n\nInput\n"
            + json.dumps(context, ensure_ascii=False, indent=2)
        )

    @staticmethod
    def _parse(content: str, expected_count: int) -> dict[str, Any]:
        text = content.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1]).strip()
        try:
            report = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM returned invalid JSON: {exc}") from exc
        items = report.get("items") if isinstance(report, dict) else None
        if not isinstance(items, list):
            raise ValueError("LLM response must contain an 'items' list.")
        indices = []
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("Every LLM judgment must be an object.")
            required = {
                "index",
                "judgment",
                "relative_fundamentality",
                "reason",
                "preferred_relation",
            }
            if missing := required - set(item):
                raise ValueError(
                    "LLM judgment is missing fields: " + ", ".join(sorted(missing))
                )
            if item["judgment"] not in JUDGMENTS:
                raise ValueError(f"Invalid LLM judgment: {item['judgment']!r}.")
            relative_fundamentality = float(item["relative_fundamentality"])
            if not 0 <= relative_fundamentality <= 1:
                raise ValueError(
                    "LLM relative_fundamentality must be between 0 and 1."
                )
            item["relative_fundamentality"] = relative_fundamentality
            item.pop("confidence", None)
            indices.append(item["index"])
        expected = list(range(1, expected_count + 1))
        if sorted(indices) != expected:
            raise ValueError(
                f"LLM judgments must cover mechanism indices {expected}, got "
                f"{sorted(indices)}."
            )
        return report

    def check_context(self, context: dict[str, Any]) -> dict[str, Any]:
        """Judge mechanisms in a basic, JSON-compatible context mapping."""
        errors = []
        for attempt in range(1, self.max_retries + 1):
            content = ""
            message = {}
            try:
                request_options = {
                    "n": 1,
                    "max_tokens": 8192,
                    "temperature": 0.0,
                }
                if self.llm_provider.lower() == "deepseek":
                    request_options["thinking"] = "disabled"
                result = self._get_api()(self._prompt(context), **request_options)
                for content, _, message in result:
                    pass
                if not content.strip():
                    reasoning = message.get("reasoning_content") or ""
                    finish_reason = message.get("finish_reason") or "unknown"
                    raise ValueError(
                        "LLM returned no final answer "
                        f"(finish_reason={finish_reason}, "
                        f"reasoning_chars={len(reasoning)})."
                    )
                report = self._parse(content, len(context["mechanisms"]))
                report["provider"] = self.llm_provider
                report["model"] = self.llm_model
                report["attempt"] = attempt
                return report
            except Exception as exc:
                errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
        raise ValueError(
            "Mechanism fundamentality judgment failed after "
            f"{self.max_retries} attempts: " + "; ".join(errors)
        )

    def check(
        self,
        problem: Problem,
        mechanisms: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return self.check_context(self.build_context(problem, mechanisms))
