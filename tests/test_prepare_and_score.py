import json
import numpy as np
import nd2py as nd
import pytest
from pathlib import Path

from src.core import ConstantSpec, UNIT
from src.prepare_problem import get_parser as get_prepare_parser, main as prepare_main, prepare_problem
from src.synthetic_data import generate_synthetic_data, get_parser as get_synthetic_parser, main as synthetic_main
from src.evaluate_result import validate_result
from src.features.io import load_problem


def test_prepare_all_task_types(tmp_path):
    problem = load_problem("problems/demo_problem.yaml")
    data = generate_synthetic_data(
        problem, train_samples=8, id_test_samples=4, ood_test_samples=4, pilot_samples=64
    )
    for task in ("symbolic_regression", "mechanism_explanation", "mechanism_discovery"):
        artifacts = prepare_problem(problem, data, task=task, save_answer=True)
        public = artifacts["problem_json"]
        assert "mechanisms" not in public
        assert set(artifacts) == {
            "problem_json", "answer_json", "data_train", "data_id_test", "data_ood_test"
        }
        if task == "symbolic_regression":
            assert "mechanisms" not in artifacts["answer_json"]


def test_prepare_controls_auxiliary_and_constant_visibility():
    problem = load_problem("problems/demo_problem.yaml")
    mechanism_only = ConstantSpec("k", "mechanism-only constant", UNIT({}), 1.0)
    problem.constants.append(mechanism_only)
    problem.mechanism[0].formula = "a + 0 * k"
    data = generate_synthetic_data(
        problem, train_samples=8, id_test_samples=4, ood_test_samples=4, pilot_samples=64
    )

    hidden = prepare_problem(problem, data, task="mechanism_discovery")
    assert {item["name"] for item in hidden["problem_json"]["constants"]} == {"π", "G"}
    assert "auxiliary_input_variables" not in hidden["problem_json"]
    assert hidden["data_train"].shape[0] == 1 + len(problem.input_variables)

    revealed = prepare_problem(
        problem, data, task="mechanism_discovery", reveal_auxiliary=True
    )
    assert {item["name"] for item in revealed["problem_json"]["constants"]} == {"π", "G", "k"}
    assert "auxiliary_input_variables" in revealed["problem_json"]
    assert revealed["data_train"].shape[0] == (
        1 + len(problem.input_variables) + len(problem.auxiliary_input_variables)
    )

    with pytest.raises(ValueError, match="not available for symbolic regression"):
        prepare_problem(
            problem, data, task="symbolic_regression", reveal_auxiliary=True
        )


def test_exact_answer_scores_one(tmp_path):
    problem = load_problem("problems/demo_problem.yaml")
    data = generate_synthetic_data(
        problem, train_samples=8, id_test_samples=8, ood_test_samples=8, pilot_samples=64
    )
    artifacts = prepare_problem(problem, data, task="symbolic_regression", save_answer=True)
    answer = artifacts["answer_json"]
    submission = {"phenomenological_formula": answer["phenomenological_formula"]}
    report = validate_result(submission, answer, problem=problem)
    assert report["formula"]["equivalent"]
    assert "score" not in report


def test_problem_can_supply_answer_without_data_file():
    from src.features.answer import build_answer
    problem = load_problem("problems/demo_problem.yaml")
    answer = build_answer(problem, "symbolic_regression")
    submission = {"phenomenological_formula": answer["phenomenological_formula"]}
    report = validate_result(submission, answer, problem=problem)
    assert report["formula"]["equivalent"]
    assert "score" not in report


def test_exact_mechanism_formula_sequence_scores_one():
    from src.features.answer import build_answer
    from src.features.io import load_submission

    problem = load_problem("problems/demo_problem.yaml")
    answer = build_answer(problem, "mechanism_discovery")
    formulas = [item["formula"] for item in answer["mechanisms"]]
    submission = load_submission(
        "; ".join(formulas), task="mechanism_discovery", answer=answer
    )
    report = validate_result(submission, answer, problem=problem)
    assert report["derivation"]["score"] == 1.0
    assert report["structure_recovery"]["score"] == 1.0
    assert "score" not in report


def test_fundamentality_is_reported_without_changing_primary_score():
    from src.features.answer import build_answer
    from src.features.io import load_submission

    problem = load_problem("problems/demo_problem.yaml")
    answer = build_answer(problem, "mechanism_discovery")
    formulas = [item["formula"] for item in answer["mechanisms"]]
    submission = load_submission(
        "; ".join(formulas), task="mechanism_discovery", answer=answer
    )

    class Scorer:
        def compare(self, candidate, *, answer, problem):
            return {
                "score": 0.25,
                "items": [],
                "provider": "test",
                "model": "fake",
            }

    report = validate_result(
        submission,
        answer,
        problem=problem,
        fundamentality_scorer=Scorer(),
    )
    assert report["mechanism_fundamentality"]["score"] == 0.25
    assert "score" not in report


def test_prepare_file_format(tmp_path):
    synthetic_dir = tmp_path / "synthetic"
    _generate_test_data(synthetic_dir)
    args = get_prepare_parser().parse_args([
        "--problems", "problems/demo_problem.yaml",
        "--synthetic-data-dir", str(synthetic_dir),
        "--output-dir", str(tmp_path / "prepared"),
        "--task", "symbolic_regression",
        "--format", "file",
        "--save-answer",
    ])
    assert prepare_main(args) == 0
    output = next((tmp_path / "prepared").glob("*.npz"))
    with np.load(output) as archive:
        assert set(archive.files) == {
            "problem_json", "answer_json", "data_train", "data_id_test", "data_ood_test"
        }


def test_prepare_main_requires_existing_synthetic_data(tmp_path):
    args = get_prepare_parser().parse_args([
        "--problems", "problems/demo_problem.yaml",
        "--synthetic-data-dir", str(tmp_path / "missing"),
        "--output-dir", str(tmp_path / "prepared"),
        "--task", "symbolic_regression",
    ])
    assert prepare_main(args) == 1
    assert not (tmp_path / "prepared").exists()


def _generate_test_data(output_dir):
    args = get_synthetic_parser().parse_args([
        "--problems", "problems/demo_problem.yaml",
        "--output-dir", str(output_dir),
        "--train-samples", "8",
        "--id-test-samples", "4",
        "--ood-test-samples", "4",
        "--pilot-samples", "64",
    ])
    assert synthetic_main(args) == 0


def test_batch_operations_accept_directories(tmp_path):
    problem_dir = tmp_path / "problems"
    problem_dir.mkdir()
    (problem_dir / "demo_problem.yaml").write_text(
        Path("problems/demo_problem.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    synthetic_args = get_synthetic_parser().parse_args([
        "--problems", str(problem_dir),
        "--output-dir", str(tmp_path / "synthetic"),
        "--train-samples", "8",
        "--id-test-samples", "4",
        "--ood-test-samples", "4",
        "--pilot-samples", "64",
    ])
    assert synthetic_main(synthetic_args) == 0
    prepare_args = get_prepare_parser().parse_args([
        "--problems", str(problem_dir),
        "--synthetic-data-dir", str(tmp_path / "synthetic"),
        "--output-dir", str(tmp_path / "prepared"),
        "--task", "mechanism_explanation",
        "--format", "directory",
    ])
    assert prepare_main(prepare_args) == 0
    synthetic_outputs = list((tmp_path / "synthetic").glob("*.npz"))
    assert len(synthetic_outputs) == 1
    with np.load(synthetic_outputs[0]) as data:
        assert set(data.files) == {
            "train", "id_test", "ood_test", "variables", "generation_config"
        }
    assert (tmp_path / "prepared" / "开普勒第三定律_-_原版" / "problem.json").is_file()


def test_answer_is_private_by_default_and_constants_keep_metadata(tmp_path):
    problem = load_problem("problems/demo_problem.yaml")
    data = generate_synthetic_data(
        problem, train_samples=8, id_test_samples=4, ood_test_samples=4, pilot_samples=64
    )
    public = prepare_problem(problem, data, task="mechanism_explanation")
    assert set(public) == {"problem_json", "data_train"}
    private = prepare_problem(problem, data, task="mechanism_explanation", save_answer=True)
    answer = private["answer_json"]
    assert answer["task"] == "mechanism_explanation"
    assert {"a", "M", "m", "G", "π"} <= set(answer["source_variables"])
    assert answer["constants"][0].keys() == {"name", "value", "description", "unit"}


def test_prepare_does_not_delete_redundant_files_and_requires_overwrite(tmp_path):
    synthetic_dir = tmp_path / "synthetic"
    _generate_test_data(synthetic_dir)
    argv = [
        "--problems", "problems/demo_problem.yaml",
        "--synthetic-data-dir", str(synthetic_dir),
        "--output-dir", str(tmp_path / "prepared"),
        "--task", "mechanism_explanation",
    ]
    assert prepare_main(get_prepare_parser().parse_args(argv)) == 0
    output = tmp_path / "prepared" / "开普勒第三定律_-_原版"
    extra = output / "keep-me.txt"
    extra.write_text("user data")
    assert prepare_main(get_prepare_parser().parse_args(argv)) == 1
    assert prepare_main(get_prepare_parser().parse_args(argv + ["--force"])) == 0
    assert extra.read_text() == "user data"


def test_prepare_formats_have_the_same_logical_contents(tmp_path):
    synthetic_dir = tmp_path / "synthetic"
    _generate_test_data(synthetic_dir)
    common = [
        "--problems", "problems/demo_problem.yaml",
        "--synthetic-data-dir", str(synthetic_dir),
        "--task", "symbolic_regression",
        "--save-answer",
    ]
    directory_root = tmp_path / "directory"
    file_root = tmp_path / "file"
    assert prepare_main(get_prepare_parser().parse_args(
        common + ["--output-dir", str(directory_root), "--format", "directory"]
    )) == 0
    assert prepare_main(get_prepare_parser().parse_args(
        common + ["--output-dir", str(file_root), "--format", "file"]
    )) == 0
    directory = next(path for path in directory_root.iterdir() if path.is_dir())
    archive_path = next(file_root.glob("*.npz"))
    with np.load(archive_path) as archive:
        assert json.loads((directory / "problem.json").read_text()) == json.loads(
            str(archive["problem_json"].item())
        )
        assert json.loads((directory / "answer.json").read_text()) == json.loads(
            str(archive["answer_json"].item())
        )
        for key in ("data_train", "data_id_test", "data_ood_test"):
            np.testing.assert_array_equal(np.load(directory / f"{key}.npy"), archive[key])


def test_preparation_report_prints_compact_manifest(tmp_path):
    from src.prepare_problem import format_reports

    output = tmp_path / "task"
    report = {
        "problem_name": "example",
        "path": str(output),
        "format": "directory",
        "files": ["problem.json", "data_train.npy"],
        "overwritten_files": ["data_train.npy"],
        "redundant_files": ["answer.json"],
    }
    rendered = format_reports([report])
    assert "Files" in rendered
    assert "data_train.npy" in rendered and "(Overwritten)" in rendered
    assert "- answer.json" in rendered
    assert str(output / "data_train.npy") not in rendered
