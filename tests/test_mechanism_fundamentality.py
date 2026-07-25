import json

import pytest

from src.features.io import load_problem
from src.features.validation import LLMMechanismFundamentalityChecker
from src.metrics import LLMMechanismFundamentalityScorer
from src.metrics.mechanism_simplicity import formula_complexity
from src.validate_problem import ValidationError, _check_fundamentality


class FakeAPI:
    def __init__(self, response):
        self.response = response
        self.prompt = None
        self.kwargs = None

    def __call__(self, prompt, **kwargs):
        self.prompt = prompt
        self.kwargs = kwargs
        return iter([(self.response, [], {})])


class SequenceAPI:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def __call__(self, prompt, **kwargs):
        self.calls += 1
        content, message = next(self.responses)
        return iter([(content, [], message)])


def _response(problem, judgment="fundamental"):
    return json.dumps({
        "items": [
            {
                "index": index,
                "judgment": judgment,
                "relative_fundamentality": (
                    1.0 if judgment == "fundamental" else 0.0
                ),
                "reason": "A basic physical relationship.",
                "preferred_relation": None,
            }
            for index in range(1, len(problem.mechanism) + 1)
        ]
    })


def test_llm_checker_sends_complete_physical_context():
    problem = load_problem("problems/demo_problem.yaml", solve=False)
    api = FakeAPI(_response(problem))
    checker = LLMMechanismFundamentalityChecker(api=api)

    report = checker.check(problem)

    assert len(report["items"]) == len(problem.mechanism)
    assert problem.problem_description.strip() in api.prompt
    assert problem.phenomenological_formula not in api.prompt
    assert '"target_variable"' in api.prompt
    assert problem.target_variable.name in api.prompt
    assert problem.mechanism[0].equation in api.prompt
    assert problem.mechanism[0].formula_description in api.prompt
    assert api.kwargs["temperature"] == 0.0
    assert api.kwargs["thinking"] == "disabled"
    assert "acceptable fundamental building block" in api.prompt
    assert "mere renaming or identity" in api.prompt
    assert "quantities defined earlier" in api.prompt
    assert "reference phenomenological equation" not in api.prompt
    assert "hidden reference" not in api.prompt
    assert "Other metrics" not in api.prompt
    assert "Newtonian" not in api.prompt
    assert "Kepler" not in api.prompt


def test_llm_checker_rejects_incomplete_results():
    problem = load_problem("problems/demo_problem.yaml", solve=False)
    api = FakeAPI(json.dumps({"items": []}))
    checker = LLMMechanismFundamentalityChecker(api=api)

    with pytest.raises(ValueError, match="must cover mechanism indices"):
        checker.check(problem)


def test_llm_checker_retries_empty_final_answer():
    problem = load_problem("problems/demo_problem.yaml", solve=False)
    api = SequenceAPI([
        ("", {"finish_reason": "length", "reasoning_content": "thinking"}),
        (_response(problem), {"finish_reason": "stop"}),
    ])
    checker = LLMMechanismFundamentalityChecker(api=api)

    report = checker.check(problem)

    assert api.calls == 2
    assert report["attempt"] == 2


def test_fundamentality_check_rejects_nonfundamental_relationship():
    problem = load_problem("problems/demo_problem.yaml", solve=False)
    report = json.loads(_response(problem))
    report["items"][0].update({
        "judgment": "not_fundamental",
        "reason": "This is already a derived law.",
        "preferred_relation": "F = m * a",
    })
    report.update({"provider": "test", "model": "fake"})

    class Checker:
        def check(self, _problem):
            return report

    with pytest.raises(ValidationError, match="Preferred relationship: F = m \\* a"):
        _check_fundamentality(problem, Checker())


def test_submission_scorer_returns_one_score_per_mechanism():
    from src.features.answer import build_answer

    problem = load_problem("problems/demo_problem.yaml", solve=False)
    answer = build_answer(problem, "mechanism_discovery")
    response = json.loads(_response(problem))
    response["items"][0]["judgment"] = "not_fundamental"
    response["items"][0]["relative_fundamentality"] = 0.0
    response["items"][1]["judgment"] = "uncertain"
    response["items"][1]["relative_fundamentality"] = 0.5
    checker = LLMMechanismFundamentalityChecker(
        api=FakeAPI(json.dumps(response))
    )
    scorer = LLMMechanismFundamentalityScorer(checker=checker)

    report = scorer.compare(answer["mechanisms"], answer=answer, problem=problem)

    assert report["items"][0]["score"] == 0.0
    assert "simplicity_score" not in report["items"][0]
    assert report["items"][2]["score"] == 1.0
    expected = 0.7 * report["minimum_item_score"] + 0.3 * report["mean_item_score"]
    assert report["score"] == pytest.approx(expected)
    assert report["minimum_item_score"] <= report["score"] < report["mean_item_score"]


def test_submission_scorer_can_build_context_from_answer_only():
    from src.features.answer import build_answer

    problem = load_problem("problems/demo_problem.yaml", solve=False)
    answer = build_answer(problem, "mechanism_explanation")
    api = FakeAPI(_response(problem))
    checker = LLMMechanismFundamentalityChecker(api=api)
    scorer = LLMMechanismFundamentalityScorer(checker=checker)

    report = scorer.compare(answer["mechanisms"], answer=answer)

    assert report["score"] == 1.0
    assert answer["phenomenological_formula"] not in api.prompt
    assert answer["target_variable"] in api.prompt
    assert answer["mechanisms"][0]["formula"] in api.prompt


def test_formula_complexity_counts_equation_ast_nodes():
    assert formula_complexity("a = x") < formula_complexity(
        "a = x**2 + x + 1"
    )
