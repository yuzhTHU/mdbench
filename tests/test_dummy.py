import json
from pathlib import Path
import numpy as np

from src.algorithms import get_algorithm, get_ask, get_update_parser, list_algorithms
from src.algorithms import dummy
from src.evaluate import expand_probe_reply
from src.scoring import feedback, public_variables


def _problem():
    return {
        "task_description": "A two-input linear demonstration task.",
        "variables": [
            {"name": "y", "description": "output", "unit": "1", "role": "target"},
            {"name": "x1", "description": "first input", "unit": "1", "role": "input"},
            {"name": "x2", "description": "second input", "unit": "1", "role": "input"},
        ],
        "data_columns": ["y", "x1", "x2"],
        "data_layout": "variables_by_samples",
    }


def test_dummy_is_discoverable_and_fits_submission(tmp_path):
    assert "dummy" in list_algorithms()
    assert get_algorithm("dummy") is dummy.run
    assert get_ask("dummy") is dummy.get_ask
    assert get_update_parser("dummy") is None

    problem = _problem()
    problem_file = tmp_path / "problem.json"
    problem_file.write_text(json.dumps(problem))
    x1 = np.array([-2.0, -1.0, 0.5, 2.0, 4.0])
    x2 = np.array([1.0, -3.0, 2.0, 0.25, -1.0])
    train = np.vstack((2 * x1 - 3 * x2, x1, x2))
    train_file = tmp_path / "train.npy"
    np.save(train_file, train)
    submission, checkpoint = dummy.run(None, problem_file, train_file, "unused")
    assert "z1 = a1*x1" in submission
    assert "z2 = a2*x2" in submission
    assert "y = z1 + z2" in submission
    scores = feedback(problem, train, "\n".join(submission))["train"]
    assert scores["numerically_equivalent"] and scores["r2"] == 1
    assert {path.name for path in tmp_path.iterdir()} == {"problem.json", "train.npy"}


def test_dummy_probe_is_determined_by_its_checkpoint(tmp_path):
    problem = _problem()
    problem_file = tmp_path / "problem.json"
    problem_file.write_text(json.dumps(problem))
    train = np.vstack((np.arange(5.0), np.arange(5.0) + 1, np.arange(5.0) + 2))
    train_file = tmp_path / "train.npy"
    np.save(train_file, train)
    _, checkpoint = dummy.run(None, problem_file, train_file, "unused")
    first = dummy.get_ask(None, checkpoint)
    second = dummy.get_ask(None, checkpoint)
    question = "derive h (hidden state) as a function of the input variables: x1, x2."
    reply = first(question, "h", "hidden state")
    assert second(question, "h", "hidden state") == reply
    assert reply in {"h = a1*x1", "h = a2*x2"}

    variables, _, _ = public_variables(_problem())
    sources = [variable for variable in variables if variable.role == "input"]
    expression = expand_probe_reply(
        reply,
        "h",
        sources,
        {"a1": 2, "a2": -3, "z1": 2, "z2": -3},
    )
    assert str(expression) in {"2*x1", "-3*x2"}
