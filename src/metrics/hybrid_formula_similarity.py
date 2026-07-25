"""Numeric and LLM-assisted phenomenological-formula equivalence."""
from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import sympy as sp
from dotenv import load_dotenv

from .formula_similarity import NumericEquivalenceChecker


class HybridFormulaEquivalenceChecker:
    """Cross-check formula equivalence with SymPy, numeric data, and an LLM.

    The checker first evaluates both formulas on the supplied points. It then
    asks an LLM to judge mathematical or domain-restricted equivalence using
    the formulas, variable ranges, and numeric result. A valid LLM judgment is
    the final decision, matching the policy used by SRAgent's symbolic
    accuracy evaluation. If every LLM attempt fails, the numeric decision is
    retained and the failure is recorded explicitly.

    ``api`` may be injected for offline tests or replaced by a future judge.
    Otherwise ``LLMAPI.create(provider, model)`` constructs the copied API.
    """

    def __init__(
        self,
        *,
        llm_provider: str = "openrouter",
        llm_model: str = "qwen/qwen3.5-flash-02-23",
        rtol: float = 1e-6,
        atol: float = 1e-9,
        max_retries: int = 3,
        api: Any = None,
    ):
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.numeric_checker = NumericEquivalenceChecker(rtol=rtol, atol=atol)
        self.max_retries = max_retries
        self.api = api

    @staticmethod
    def _ranges(values: dict[str, Any]) -> dict[str, tuple[float, float]]:
        ranges = {}
        for name, value in values.items():
            array = np.asarray(value, dtype=float)
            finite = array[np.isfinite(array)]
            if finite.size:
                ranges[name] = (float(finite.min()), float(finite.max()))
        return ranges

    @staticmethod
    def _parse_judgment(content: str) -> dict[str, Any]:
        candidates = [content.strip()]
        candidates.extend(
            match.group(1).strip()
            for match in re.finditer(
                r"```(?:json)?\s*(.*?)```", content, re.DOTALL | re.IGNORECASE
            )
        )
        first, last = content.find("{"), content.rfind("}")
        if 0 <= first < last:
            candidates.append(content[first:last + 1])
        for candidate in candidates:
            try:
                result = json.loads(candidate)
            except (json.JSONDecodeError, TypeError):
                continue
            if (
                isinstance(result, dict)
                and isinstance(result.get("equivalent"), bool)
                and isinstance(result.get("reason"), str)
            ):
                return {"equivalent": result["equivalent"], "reason": result["reason"]}
        raise ValueError("LLM response does not match {'equivalent': bool, 'reason': str}.")

    def _build_messages(
        self,
        candidate: str,
        reference: str,
        ranges: dict[str, tuple[float, float]],
        numeric: dict[str, Any],
    ) -> list[dict[str, str]]:
        range_text = "\n".join(
            f"- {name}: [{low:.16g}, {high:.16g}]"
            for name, (low, high) in sorted(ranges.items())
        ) or "- No variable ranges were supplied."
        return [
            {
                "role": "system",
                "content": (
                    "You judge symbolic-regression formulas. Decide whether the candidate "
                    "and reference are equivalent over the supplied variable ranges. Accept "
                    "algebraic rewrites, domain-restricted identities, and tiny floating-point "
                    "constant differences. Do not accept formulas that merely fit a few points. "
                    "Return only JSON with schema "
                    "{\"equivalent\": true_or_false, \"reason\": \"brief explanation\"}."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Reference formula:\n{reference}\n\n"
                    f"Candidate formula:\n{candidate}\n\n"
                    f"Variable ranges:\n{range_text}\n\n"
                    f"Numeric check:\n"
                    f"- equivalent: {numeric.get('equivalent')}\n"
                    f"- close-point fraction: {numeric.get('score', 0.0):.8f}\n"
                    f"- tested points: {numeric.get('tested', 0)}"
                ),
            },
        ]

    def _get_api(self):
        if self.api is not None:
            return self.api
        load_dotenv()
        from ..utils.llm import LLMAPI

        self.api = LLMAPI.create(
            llm_provider=self.llm_provider,
            llm_model=self.llm_model,
        )
        return self.api

    @staticmethod
    def _rhs(formula: str) -> str:
        return formula.split("=", 1)[-1].strip().replace("^", "**")

    @classmethod
    def _symbolic_judgment(cls, candidate: str, reference: str) -> dict[str, Any]:
        """Try to prove global algebraic equivalence with SymPy."""
        try:
            local_dict = {
                "sqrt": sp.sqrt,
                "sin": sp.sin,
                "cos": sp.cos,
                "tan": sp.tan,
                "exp": sp.exp,
                "log": sp.log,
                "abs": sp.Abs,
                "Abs": sp.Abs,
            }
            candidate_expr = sp.sympify(cls._rhs(candidate), locals=local_dict)
            reference_expr = sp.sympify(cls._rhs(reference), locals=local_dict)
            difference = sp.simplify(candidate_expr - reference_expr)
            equivalent = bool(difference == 0 or difference.equals(0) is True)
            return {
                "equivalent": equivalent,
                "reason": (
                    "SymPy simplified the formula difference to zero."
                    if equivalent
                    else f"SymPy did not prove equality; simplified difference: {difference}"
                ),
            }
        except Exception as exc:
            return {
                "equivalent": None,
                "reason": f"SymPy comparison failed: {type(exc).__name__}: {exc}",
            }

    def compare(
        self,
        candidate: str,
        reference: str,
        values: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        symbolic = self._symbolic_judgment(candidate, reference)
        numeric = self.numeric_checker.compare(candidate, reference, values)
        ranges = self._ranges(values)
        messages = self._build_messages(candidate, reference, ranges, numeric)
        errors = []
        for attempt in range(1, self.max_retries + 1):
            content = ""
            try:
                for content, _, _ in self._get_api()(
                    messages, n=1, max_tokens=1024, temperature=0.0
                ):
                    pass
                llm = self._parse_judgment(content)
                equivalent = (
                    True if symbolic.get("equivalent") is True else llm["equivalent"]
                )
                agreement = llm["equivalent"] == numeric.get("equivalent")
                return {
                    "equivalent": equivalent,
                    "score": 1.0 if equivalent else 0.0,
                    "reason": (
                        symbolic["reason"]
                        if symbolic.get("equivalent") is True
                        else llm["reason"]
                    ),
                    "symbolic": symbolic,
                    "numeric": numeric,
                    "llm": {
                        **llm,
                        "provider": self.llm_provider,
                        "model": self.llm_model,
                        "attempt": attempt,
                        "agrees_with_numeric": agreement,
                    },
                }
            except Exception as exc:
                errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")

        equivalent = (
            True
            if symbolic.get("equivalent") is True
            else bool(numeric.get("equivalent", False))
        )
        return {
            "equivalent": equivalent,
            "score": 1.0 if symbolic.get("equivalent") is True else float(
                numeric.get("score", 0.0)
            ),
            "reason": (
                symbolic["reason"]
                if symbolic.get("equivalent") is True
                else "LLM judgment failed; retained the numeric decision."
            ),
            "symbolic": symbolic,
            "numeric": numeric,
            "llm": {
                "equivalent": None,
                "provider": self.llm_provider,
                "model": self.llm_model,
                "errors": errors,
            },
        }
