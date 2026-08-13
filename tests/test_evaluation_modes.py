import json

import numpy as np

from src.evaluate_result import (
    _load_private_splits,
    _prediction_metrics,
    evaluate_feedback,
)
from src.features.evaluation import load_public_task, public_answer_context
from src.features.io import load_problem, load_submission
from src.prepare_problem import prepare_problem
from src.synthetic_data import generate_synthetic_data


def _prepared(task):
    problem = load_problem("problems/demo_problem.yaml")
    data = generate_synthetic_data(
        problem,
        train_samples=12,
        id_test_samples=6,
        ood_test_samples=6,
        pilot_samples=64,
    )
    return problem, prepare_problem(problem, data, task=task, save_answer=True)


def test_prediction_metrics_for_perfect_predictions():
    metrics = _prediction_metrics(
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 2.0, 3.0]),
        split="train",
    )
    assert metrics == {
        "split": "train",
        "pearson_r": 1.0,
        "r2": 1.0,
        "mae": 0.0,
        "rmse": 0.0,
        "smape": 0.0,
        "acc": 1.0,
        "all_close": True,
        "tested": 3,
    }


def test_feedback_symbolic_regression_uses_train_only():
    _, artifacts = _prepared("symbolic_regression")
    public = artifacts["problem_json"]
    context = public_answer_context(public)
    submission = load_submission(
        "T = sqrt(4 * π**2 * a**3 / (G * M))",
        task=public["task"],
        answer=context,
    )
    report = evaluate_feedback(submission, public, artifacts["data_train"])
    assert report["evaluation_mode"] == "feedback"
    assert [item["split"] for item in report["data_accuracy"]["splits"]] == ["train"]
    assert report["structure_recovery"] is None


def test_feedback_mechanism_discovery_has_no_reference_recovery():
    _, artifacts = _prepared("mechanism_discovery")
    public = artifacts["problem_json"]
    context = public_answer_context(public)
    submission = load_submission(
        "v = sqrt(G * M / a); T = 2 * π * a / v",
        task=public["task"],
        answer=context,
    )
    class FundamentalityScorer:
        llm_provider = "test-provider"
        llm_model = "test-model"

        def compare(self, candidate, *, answer, problem):
            return {
                "score": 0.8,
                "minimum_item_score": 0.75,
                "mean_item_score": 0.9,
                "bottleneck_weight": 0.7,
                "items": [],
                "provider": self.llm_provider,
                "model": self.llm_model,
            }

    report = evaluate_feedback(
        submission,
        public,
        artifacts["data_train"],
        fundamentality_scorer=FundamentalityScorer(),
    )
    assert report["derivation"] is None
    assert report["data_accuracy"]["splits"][0]["acc"] == 1.0
    assert report["data_accuracy"]["splits"][0]["r2"] == 1.0
    assert report["structure_recovery"] is None
    assert report["mechanism_fundamentality"]["score"] == 0.8
    assert report["mechanism_fundamentality"]["provider"] == "test-provider"
    assert report["mechanism_simplicity"]["mean_ast_nodes_per_item"] > 0


def test_feedback_mechanism_explanation_checks_public_derivation_without_prediction():
    _, artifacts = _prepared("mechanism_explanation")
    public = artifacts["problem_json"]
    context = public_answer_context(public)
    submission = load_submission(
        "v = sqrt(G * M / a); T = 2 * π * a / v",
        task=public["task"],
        answer=context,
    )

    class DerivationChecker:
        def compare(self, candidate, reference, values):
            assert candidate.startswith("T =")
            assert reference == public["phenomenological_formula"]
            return {
                "equivalent": True,
                "score": 1.0,
                "symbolic": {"equivalent": True, "reason": "test"},
                "numeric": {"equivalent": True, "score": 1.0, "tested": 12},
                "llm": {"equivalent": None, "provider": None, "model": None},
            }

    report = evaluate_feedback(
        submission,
        public,
        None,
        derivation_checker=DerivationChecker(),
    )

    assert report["derivation"]["equivalent"] is True
    assert report["data_accuracy"] is None
    assert report["structure_recovery"] is None


def test_public_loader_ignores_private_files(tmp_path):
    _, artifacts = _prepared("symbolic_regression")
    (tmp_path / "problem.json").write_text(
        json.dumps(artifacts["problem_json"]), encoding="utf-8"
    )
    np.save(tmp_path / "data_train.npy", artifacts["data_train"])
    (tmp_path / "answer.json").write_text("not valid JSON", encoding="utf-8")
    problem, train = load_public_task(tmp_path)
    assert problem["task"] == "symbolic_regression"
    np.testing.assert_array_equal(train, artifacts["data_train"])


def test_private_split_loader_includes_train_id_and_ood(tmp_path):
    _, artifacts = _prepared("symbolic_regression")
    archive = tmp_path / "task.npz"
    np.savez(
        archive,
        problem_json=json.dumps(artifacts["problem_json"]),
        answer_json=json.dumps(artifacts["answer_json"]),
        data_train=artifacts["data_train"],
        data_id_test=artifacts["data_id_test"],
        data_ood_test=artifacts["data_ood_test"],
    )

    splits = _load_private_splits(archive, artifacts["answer_json"])

    assert [split for split, _, _ in splits] == ["train", "id_test", "ood_test"]
