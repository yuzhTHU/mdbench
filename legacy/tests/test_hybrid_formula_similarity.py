import json

import numpy as np

from src.utils.llm import LLMAPI
from src.metrics import HybridFormulaEquivalenceChecker


class FakeAPI:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def __call__(self, messages, **kwargs):
        self.calls += 1
        return iter([(next(self.responses), None, {"role": "assistant"})])


def test_copied_api_package_is_importable():
    assert hasattr(LLMAPI, "create")


def test_valid_llm_judgment_adjudicates_numeric_result():
    api = FakeAPI([json.dumps({"equivalent": False, "reason": "Not globally equal."})])
    checker = HybridFormulaEquivalenceChecker(api=api)
    report = checker.compare("y = 2 * x", "y = x + x", {"x": np.arange(5.0)})
    assert report["numeric"]["equivalent"] is True
    assert report["llm"]["equivalent"] is False
    assert report["llm"]["agrees_with_numeric"] is False
    assert report["equivalent"] is True
    assert report["symbolic"]["equivalent"] is True
    assert report["score"] == 1.0


def test_llm_parser_accepts_fenced_json():
    api = FakeAPI([
        'Analysis omitted.\n```json\n{"equivalent": true, "reason": "Same identity."}\n```'
    ])
    checker = HybridFormulaEquivalenceChecker(api=api)
    report = checker.compare("y = x + x", "y = 2 * x", {"x": np.arange(5.0)})
    assert report["equivalent"] is True
    assert report["reason"] == "SymPy simplified the formula difference to zero."


def test_failed_llm_judgment_retains_numeric_result():
    api = FakeAPI(["invalid", "still invalid"])
    checker = HybridFormulaEquivalenceChecker(api=api, max_retries=2)
    report = checker.compare("y = 2 * x", "y = x + x", {"x": np.arange(5.0)})
    assert api.calls == 2
    assert report["equivalent"] is True
    assert report["llm"]["equivalent"] is None
    assert len(report["llm"]["errors"]) == 2
