"""Load public task packages and their leakage-safe evaluation context."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _json_value(value: np.ndarray) -> dict[str, Any]:
    return json.loads(str(value.item()))


def load_public_task(path: str | Path) -> tuple[dict[str, Any], np.ndarray]:
    """Load ``problem_json`` and training data from a prepared task package.

    ``path`` may name a prepared directory, its ``problem.json``, or a packed
    NPZ task. Private answer and test artifacts are deliberately ignored.
    """
    path = Path(path)
    if path.is_dir():
        problem_path = path / "problem.json"
        data_path = path / "data_train.npy"
        if not problem_path.is_file() or not data_path.is_file():
            raise ValueError(
                f"Prepared task directory {path} must contain problem.json and "
                "data_train.npy."
            )
        return (
            json.loads(problem_path.read_text(encoding="utf-8")),
            np.load(data_path),
        )
    if path.suffix.lower() == ".npz":
        with np.load(path) as archive:
            missing = {"problem_json", "data_train"} - set(archive.files)
            if missing:
                raise ValueError(
                    f"Prepared task NPZ {path} is missing: {', '.join(sorted(missing))}."
                )
            return _json_value(archive["problem_json"]), archive["data_train"].copy()
    if path.name == "problem.json":
        data_path = path.with_name("data_train.npy")
        if not data_path.is_file():
            raise ValueError(f"Training data not found beside {path}: {data_path}")
        return json.loads(path.read_text(encoding="utf-8")), np.load(data_path)
    raise ValueError(
        "--problem must point to a prepared task directory, problem.json, or task NPZ."
    )


def public_answer_context(problem: dict[str, Any]) -> dict[str, Any]:
    """Build the non-secret schema needed to parse and execute submissions."""
    target = problem["target_variable"]
    variable_names = [item["name"] for item in problem.get("variables", [])]
    auxiliary_names = [
        item["name"] for item in problem.get("auxiliary_input_variables", [])
    ]
    constants = problem.get("constants", [])
    context = {
        "task": problem["task"],
        "problem_name": problem.get("problem_name", ""),
        "problem_description": problem.get("problem_description", ""),
        "target_variable": target,
        "source_variables": [
            *[name for name in variable_names if name != target],
            *auxiliary_names,
            *[item["name"] for item in constants],
        ],
        "constants": constants,
        "variable_descriptions": {
            item["name"]: item.get("description", "")
            for item in problem.get("variables", [])
        } | {
            item["name"]: item.get("description", "")
            for item in problem.get("auxiliary_input_variables", [])
        },
        "intermediate_variables": [],
    }
    if "phenomenological_formula" in problem:
        context["phenomenological_formula"] = problem["phenomenological_formula"]
    return context


def training_values(
    problem: dict[str, Any], data: np.ndarray
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Map rows in ``data_train`` to public names and return target values."""
    names = [item["name"] for item in problem.get("variables", [])]
    names.extend(
        item["name"] for item in problem.get("auxiliary_input_variables", [])
    )
    if data.ndim != 2 or data.shape[0] != len(names):
        raise ValueError(
            f"Training data has shape {data.shape}; expected {len(names)} rows "
            "from the public variable metadata."
        )
    values = {name: data[index] for index, name in enumerate(names)}
    values.update({item["name"]: item["value"] for item in problem.get("constants", [])})
    target = problem["target_variable"]
    if target not in values:
        raise ValueError(f"Training data does not contain target variable {target!r}.")
    return values, np.asarray(values[target], dtype=float)
