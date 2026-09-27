from argparse import Namespace

import pytest

from src.algorithms import pysr_and_llm_direct as algorithm


def test_run_reuses_existing_submission_without_pysr(tmp_path, monkeypatch):
    source = tmp_path / "old"
    source.mkdir()
    (source / "submission.txt").write_text("y = x + 1\n")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    args = Namespace(pysr_source_dir=None, submission_source_dir=source,
                     pysr_llm_model="test/model", pysr_llm_reasoning_effort="low")

    formulas, checkpoint = algorithm.run(args, tmp_path / "missing-problem.json",
                                         tmp_path / "missing-train.npy", None)

    assert formulas == ["y = x + 1"]
    assert checkpoint["reused_submission_result"] == str(source.resolve())
    assert checkpoint["probe_model"] == "test/model"
    assert checkpoint["probe_reasoning_effort"] == "low"


def test_extract_formula_accepts_exact_probe_equation():
    assert algorithm._extract_formula("rate = x/(1 + x)", "rate") == "rate = x/(1 + x)"


def test_direct_probe_conversation_reuses_cached_formula(tmp_path, monkeypatch):
    output = tmp_path / "probe"
    output.mkdir()
    (output / "formula.txt").write_text("rate = x/(1 + x)\n")
    args = Namespace()
    conversation = algorithm.DirectProbeConversation(args, {})

    def unexpected_request(*args, **kwargs):
        raise AssertionError("a cached formula must not issue an HTTP request")

    monkeypatch.setattr(algorithm.requests.Session, "post", unexpected_request)
    assert conversation.ask("unused", "rate", "unused", output_dir=output) == "rate = x/(1 + x)"
    assert (output / "cache-reused.txt").read_text() == "formula.txt\n"


@pytest.mark.parametrize("reply", [
    "rate = x\nother = 1",
    "wrong = x",
])
def test_extract_formula_rejects_nonconforming_reply(reply):
    with pytest.raises(ValueError):
        algorithm._extract_formula(reply, "rate")
