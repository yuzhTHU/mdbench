"""Load public task packages and their leakage-safe evaluation context."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _json_value(value: np.ndarray) -> dict[str, Any]:
    return json.loads(str(value.item()))


def load_public_task(path: str | Path) -> tuple[dict[str, Any], np.ndarray | None]:
    """Load ``problem_json`` and optional training data from a task package.

    ``path`` may name a prepared directory, its ``problem.json``, or a packed
    NPZ task. Private answer and test artifacts are deliberately ignored.
    """
    path = Path(path)
    if path.is_dir():
        problem_path = path / "problem.json"
        data_path = path / "data_train.npy"
        if not problem_path.is_file():
            raise ValueError(
                f"Prepared task directory {path} must contain problem.json."
            )
        problem = json.loads(problem_path.read_text(encoding="utf-8"))
        data = np.load(data_path) if data_path.is_file() else None
        if problem.get("input") == "data" and data is None:
            raise ValueError(f"Data-input task {path} must contain data_train.npy.")
        return problem, data
    if path.suffix.lower() == ".npz":
        with np.load(path) as archive:
            missing = {"problem_json"} - set(archive.files)
            if missing:
                raise ValueError(
                    f"Prepared task NPZ {path} is missing: {', '.join(sorted(missing))}."
                )
            problem = _json_value(archive["problem_json"])
            data = archive["data_train"].copy() if "data_train" in archive else None
            if problem.get("input") == "data" and data is None:
                raise ValueError(f"Data-input task NPZ {path} is missing data_train.")
            return problem, data
    if path.name == "problem.json":
        data_path = path.with_name("data_train.npy")
        problem = json.loads(path.read_text(encoding="utf-8"))
        data = np.load(data_path) if data_path.is_file() else None
        if problem.get("input") == "data" and data is None:
            raise ValueError(f"Training data not found beside {path}: {data_path}")
        return problem, data
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
