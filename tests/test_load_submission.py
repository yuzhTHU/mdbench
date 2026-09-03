import json

import pytest

from src.features.answer import build_answer
from src.features.io import load_problem, load_submission
from src.evaluate_result import get_parser


@pytest.fixture
def answer():
    return build_answer(load_problem("demo_problem.yaml"), "mechanism_discovery")


def test_symbolic_regression_accepts_raw_formula():
    symbolic_answer = build_answer(
        load_problem("demo_problem.yaml"), "symbolic_regression"
    )
    result = load_submission(
        "T = sqrt(4 * π^2 * a^3 / G / M)",
        task="symbolic_regression", answer=symbolic_answer,
    )
    assert result["phenomenological_formula"].startswith("T =")


def test_submission_cli_argument_is_one_string(monkeypatch):
    argv = [
        "--evaluation-mode", "feedback",
        "--submission", "r = a; F = G * M * m / r^2",
        "--problem", "demo_problem.yaml",
        "--task", "mechanism_discovery",
    ]
    monkeypatch.setattr("sys.argv", ["evaluate_result"] + argv)
    args = get_parser().parse_args(argv)
    assert isinstance(args.submission, str)
    assert not hasattr(args, "data")
    assert args.llm_provider == "deepseek"
    assert args.llm_model == "deepseek-v4-flash"


def test_symbolic_regression_parser_adds_formula_evaluation_options(monkeypatch):
    argv = [
        "--evaluation-mode", "final",
        "--submission", "T = a",
        "--answer", "answer.json",
        "--task", "symbolic_regression",
        "--data", "data.npz",
        "--llm-provider", "openrouter",
        "--llm-model", "example/model",
    ]
    monkeypatch.setattr("sys.argv", ["evaluate_result"] + argv)
    args = get_parser().parse_args(argv)
    assert args.data == "data.npz"
    assert args.llm_model == "example/model"


def test_parser_infers_task_specific_options_from_answer(tmp_path, monkeypatch):
    answer_path = tmp_path / "answer.json"
    answer_path.write_text(
        json.dumps({"task": "symbolic_regression"}), encoding="utf-8"
    )
    argv = [
        "--evaluation-mode", "final",
        "--answer", str(answer_path),
        "--submission", "T = a",
        "--data", "data.npz",
    ]
    monkeypatch.setattr("sys.argv", ["evaluate_result"] + argv)
    args = get_parser().parse_args(argv)
    assert args.task is None
    assert args.data == "data.npz"


def test_mechanism_task_accepts_multiple_raw_formulas(answer):
    result = load_submission(
        "r = a; F = G * M * m / r^2",
        task="mechanism_discovery", answer=answer,
    )
    assert [item["formula"] for item in result["mechanisms"]] == [
        "r = a", "F = G * M * m / r**2",
    ]


def test_plain_text_file_uses_one_formula_per_line(tmp_path, answer):
    path = tmp_path / "submission.txt"
    path.write_text("r = a\nF = G * M * m / r^2\n", encoding="utf-8")
    result = load_submission(str(path), task="mechanism_discovery", answer=answer)
    assert len(result["mechanisms"]) == 2


@pytest.mark.parametrize("suffix", [".json", ".yaml", ".yml"])
def test_structured_submission_file_explains_supported_formats(
    tmp_path, answer, suffix
):
    path = tmp_path / f"submission{suffix}"
    path.write_text('{"mechanisms": []}', encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        load_submission(str(path), task="mechanism_discovery", answer=answer)

    message = str(exc_info.value)
    assert "Structured submission file" in message
    assert "equations separated by semicolons" in message
    assert "one 'variable = formula' equation per non-empty line" in message
    assert "mdbench evaluate --help" in message


def test_raw_structured_submission_explains_supported_formats(answer):
    with pytest.raises(ValueError, match="JSON and YAML submissions"):
        load_submission(
            '{"mechanisms": []}',
            task="mechanism_discovery",
            answer=answer,
        )


def test_rejects_unparseable_formula(answer):
    with pytest.raises(ValueError, match="nd2py cannot parse"):
        load_submission("r = sqrt(", task="mechanism_discovery", answer=answer)


def test_rejects_mechanism_without_target(answer):
    with pytest.raises(ValueError, match="exactly one"):
        load_submission("G * M", task="mechanism_discovery", answer=answer)


def test_rejects_unknown_or_duplicate_auxiliary_input_variables(answer):
    with pytest.raises(ValueError, match="underdetermined"):
        load_submission("z = unknown + a", task="mechanism_discovery", answer=answer)
    with pytest.raises(ValueError, match="no unresolved variable"):
        load_submission("z = a; z = M", task="mechanism_discovery", answer=answer)


def test_mechanism_submission_rejects_zero_left_side(answer):
    with pytest.raises(ValueError, match="left side must be a variable name"):
        load_submission(
            "F_d = k * v^2; 0 = F - F_d",
            task="mechanism_discovery",
            answer={**answer, "source_variables": [*answer["source_variables"], "k", "F"]},
        )
