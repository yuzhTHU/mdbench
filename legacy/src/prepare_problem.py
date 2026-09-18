"""Build leakage-controlled benchmark task packages."""
from __future__ import annotations
import json
import argparse
import numpy as np
import nd2py as nd
from pathlib import Path
from typing import Any
from .core import Problem
from .features.io import load_problem, solve_mechanism_equations
from .features.answer import build_answer
from .utils import confirm_overwrite, discover_yaml_files, safe_name, tag2ansi, logger

DEFAULT_PROBLEM_DIR = Path("data/problem")
DEFAULT_SYNTHETIC_DATA_DIR = Path("data/synthetic_data")
TASKS = {
    "symbolic_regression":   ("data",    "formula"   ),
    "mechanism_explanation": ("formula", "mechanisms"),
    "mechanism_discovery":   ("data",    "mechanisms"),
}
ARTIFACT_FILENAMES = {
    "problem_json": "problem.json",
    "answer_json": "answer.json",
    "data_train": "data_train.npy",
    "data_id_test": "data_id_test.npy",
    "data_ood_test": "data_ood_test.npy",
}


def prepare_problem(
    problem: Problem,
    data: dict[str, np.ndarray] | None,
    *,
    task: str,
    save_answer: bool = False,
    reveal_auxiliary: bool = False,
) -> dict[str, dict[str, Any] | np.ndarray]:
    """Build task artifacts from a problem and pre-generated synthetic data."""
    if not problem.solution:
        problem.solution = solve_mechanism_equations(problem)
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}; choose one of {sorted(TASKS)}.")
    if task == "symbolic_regression" and reveal_auxiliary:
        raise ValueError("reveal_auxiliary is not available for symbolic regression.")
    input_kind, output_kind = TASKS[task]
    if input_kind == "data":
        if data is None:
            raise ValueError(f"Synthetic data is required for task {task!r}.")
        required_splits = {"train", "id_test", "ood_test", "variables"}
        if missing := sorted(required_splits - set(data)):
            raise ValueError(
                f"Synthetic data is missing required arrays: {', '.join(missing)}"
            )
    public = {
        "problem_name": problem.problem_name,
        "problem_description": problem.problem_description,
        "task": task,
        "input": input_kind,
        "expected_output": output_kind,
        "target_variable": problem.target_variable.name,
    }

    # Public variables
    variables = []
    for v in [problem.target_variable, *problem.input_variables]:
        variables.append({
            "name": v.name,
            "description": v.description,
            "unit": v.unit.to_dict() if v.unit is not None else {}
        })
    public["variables"] = variables

    phenomenological_names = {
        node.name
        for node in nd.parse(problem.phenomenological_formula).iter_preorder()
        if isinstance(node, nd.Variable)
    }
    visible_constants = (
        problem.constants
        if reveal_auxiliary and task != "symbolic_regression"
        else [
            constant
            for constant in problem.constants
            if constant.name in phenomenological_names
        ]
    )
    public["constants"] = [{
        "name": constant.name,
        "value": constant.value,
        "description": constant.description,
        "unit": constant.unit.to_dict(),
    } for constant in visible_constants]

    # Optional auxiliary inputs
    if reveal_auxiliary:
        auxiliary_input_variables = []
        for v in problem.auxiliary_input_variables:
            auxiliary_input_variables.append({
                "name": v.name,
                "description": v.description,
                "unit": v.unit.to_dict() if v.unit is not None else {}
            })
        public["auxiliary_input_variables"] = auxiliary_input_variables

    # Optional phenomenological formula
    if input_kind == "formula":
        public["phenomenological_formula"] = f"{problem.target_variable.name} = {problem.phenomenological_formula}"

    artifacts: dict[str, dict[str, Any] | np.ndarray] = {"problem_json": public}
    data_variables = None
    if input_kind == "data":
        data_variables = data["variables"].tolist()
        visible_data_variables = [problem.target_variable, *problem.input_variables]
        if reveal_auxiliary:
            visible_data_variables.extend(problem.auxiliary_input_variables)
        missing_data_variables = [
            variable.name
            for variable in visible_data_variables
            if variable.name not in data_variables
        ]
        if missing_data_variables:
            raise ValueError(
                "Synthetic data does not contain required variables: "
                + ", ".join(missing_data_variables)
            )
        visible_indices = [
            data_variables.index(variable.name) for variable in visible_data_variables
        ]
        artifacts["data_train"] = data["train"][visible_indices]

    if save_answer:
        answer = build_answer(problem, task)
        artifacts["answer_json"] = answer
        if input_kind == "data":
            answer["data_variables"] = data_variables
            artifacts.update({
                "data_id_test": data["id_test"],
                "data_ood_test": data["ood_test"],
            })
    return artifacts


def format_reports(reports: list[dict]) -> str:
    lines = ["[blue bold]MDBench · Problem preparation[reset]"]
    for report_index, report in enumerate(reports):
        if report_index:
            lines.append("[gray]" + "─" * 72 + "[reset]")
        lines.append(f"[cyan bold]{report['problem_name']}[reset]")
        output_path = Path(report["path"])
        overwritten = {Path(path) for path in report.get("overwritten_files", [])}
        if report["format"] == "file":
            status = " [yellow bold](Overwritten)[reset]" if overwritten else ""
            lines.append(f"  [blue bold]Path[reset]: {output_path}{status}")
            continue

        directory = output_path.as_posix().rstrip("/") + "/"
        lines.append(f"  [blue bold]Path[reset]: {directory}")
        lines.append("  [blue bold]Files[reset]:")
        for value in report["files"]:
            path = Path(value)
            status = " [yellow bold](Overwritten)[reset]" if path in overwritten else ""
            lines.append(f"    - {path.name}{status}")
        if redundant := report.get("redundant_files"):
            lines.append("  [yellow bold]Warning: redundant files remain in the output directory:[reset]")
            for path in redundant:
                lines.append(f"    - {Path(path).name}")
    return tag2ansi("\n".join(lines))


def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(description="Prepare a benchmark task package")
    parser.add_argument("--problems", nargs="*", default=["problems"], help="Problem files or directories")
    parser.add_argument("--output-dir", default=str(DEFAULT_PROBLEM_DIR))
    parser.add_argument( "--synthetic-data-dir", default=str(DEFAULT_SYNTHETIC_DATA_DIR), help=(
        "Directory containing synthetic-data NPZ files generated by MDBench"
    ))
    parser.add_argument("--format", choices=["file", "directory"], default="directory")
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--reveal-auxiliary", action=argparse.BooleanOptionalAction, default=False, help=(
        "Reveal auxiliary input variables and mechanism-only constants (mechanism tasks only)"
    ))
    parser.add_argument("--save-answer", action=argparse.BooleanOptionalAction, default=False, help=(
        "Include the private answer"
    ))
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=False, help=(
        "Overwrite planned files without prompting"
    ))
    return parser


def main(args):
    try:
        reports = []
        used_names: set[str] = set()
        for problem_path in discover_yaml_files(args.problems):
            problem = load_problem(str(problem_path))
            problem.solution = solve_mechanism_equations(problem)
            name = safe_name(problem.problem_name)
            if name in used_names:
                raise ValueError(f"Duplicate problem output name: {name}")
            else:
                used_names.add(name)

            data = None
            if TASKS[args.task][0] == "data":
                synthetic_data_path = Path(args.synthetic_data_dir) / f"{name}.npz"
                if not synthetic_data_path.is_file():
                    raise FileNotFoundError(
                        f"Synthetic data not found: {synthetic_data_path}. "
                        "Run 'mdbench synthetic' before preparing data-input tasks."
                    )
                with np.load(synthetic_data_path) as source:
                    data = {key: source[key] for key in source.files}

            artifacts = prepare_problem(
                problem,
                data,
                task=args.task,
                save_answer=args.save_answer,
                reveal_auxiliary=args.reveal_auxiliary,
            )
            save_as = f"{name}.npz" if args.format == "file" else name
            output = Path(args.output_dir) / save_as
            filenames = [ARTIFACT_FILENAMES[key] for key in artifacts]
            if args.format == "directory":
                collisions = [
                    output / filename
                    for filename in filenames
                    if (output / filename).exists()
                ]
            else:
                collisions = [output] if output.exists() else []
            if collisions and not args.force and not confirm_overwrite(collisions):
                raise FileExistsError(
                    "Refusing to overwrite existing files: "
                    + ", ".join(str(path) for path in collisions)
                )

            if args.format == "directory":
                output.mkdir(parents=True, exist_ok=True)
                for key, value in artifacts.items():
                    path = output / ARTIFACT_FILENAMES[key]
                    if key.endswith("_json"):
                        path.write_text(
                            json.dumps(value, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                    else:
                        np.save(path, value)
                expected_names = set(filenames)
                redundant_files = sorted(
                    path.name
                    for path in output.iterdir()
                    if path.name not in expected_names
                )
                overwritten_files = [path.name for path in collisions]
            else:
                output.parent.mkdir(parents=True, exist_ok=True)
                archive = {
                    key: (
                        np.asarray(json.dumps(value, ensure_ascii=False))
                        if key.endswith("_json")
                        else value
                    )
                    for key, value in artifacts.items()
                }
                np.savez_compressed(output, **archive)
                redundant_files = []
                overwritten_files = list(artifacts) if collisions else []

            report = {
                "problem_name": problem.problem_name,
                "problem_path": str(problem_path),
                "path": str(output),
                "format": args.format,
                "files": filenames,
                "overwritten_files": overwritten_files,
                "redundant_files": redundant_files,
            }
            reports.append(report)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.error(tag2ansi(f"[red bold]Error:[reset] {exc}"))
        return 1
    report = format_reports(reports)
    logger.info(report)
    return 0
