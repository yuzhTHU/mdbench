# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""Generate reproducible ID/OOD synthetic datasets for benchmark problems."""
from __future__ import annotations
import json
import argparse
import numpy as np
import nd2py as nd
from pathlib import Path
from .core import Problem
from .features.io import evaluate_solution, load_problem, solve_mechanism_equations
from .features.sampling import RangeInferrer
from .utils import discover_yaml_files, safe_name, tag2ansi, logger

DEFAULT_SYNTHETIC_DATA_DIR = Path("data/synthetic_data")

def _sample_count(value, count: int) -> np.ndarray:
    """Return an nd2py result as one real-valued sample vector."""
    array = np.asarray(value)
    if array.ndim == 0:
        array = np.full(count, array.item())
    try:
        return np.asarray(np.broadcast_to(array, (count,)))
    except ValueError as exc:
        raise ValueError(f"Formula returned shape {array.shape}, expected ({count},).") from exc


def _evaluate(
    problem: Problem,
    samples: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Evaluate the target and mechanism states, returning their validity mask."""
    count = len(next(iter(samples.values()))) if samples else 1
    values = {**samples, **{constant.name: constant.value for constant in problem.constants}}
    valid = np.ones(count, dtype=bool)

    with np.errstate(all="ignore"):
        target = _sample_count(nd.parse(problem.phenomenological_formula).eval(values), count)
        valid &= np.isreal(target) & np.isfinite(target)

        mechanism_values = evaluate_solution(
            problem,
            {**values, problem.target_variable.name: target},
        )
        for variable in problem.intermediate_variables:
            result = _sample_count(mechanism_values[variable.name], count)
            valid &= np.isreal(result) & np.isfinite(result)
            mechanism_values[variable.name] = result

    return np.real(target).astype(float), mechanism_values, valid


def _draw(
    rng: np.random.Generator,
    ranges: dict[str, tuple[float, float]],
    count: int,
) -> dict[str, np.ndarray]:
    drawn = {}
    for name, bounds in ranges.items():
        low, high, *tail = bounds
        distribution = tail[0] if tail else "uniform"
        drawn[name] = (
            np.exp(rng.uniform(np.log(low), np.log(high), count))
            if distribution == "log_uniform" else rng.uniform(low, high, count)
        )
    return drawn


def _generate_split(
    problem: Problem,
    rng: np.random.Generator,
    ranges: dict[str, tuple[float, float]],
    count: int,
    output_variables,
) -> np.ndarray:
    """Use rejection sampling until one complete feature-by-sample matrix is available."""
    accepted_values = {variable.name: [] for variable in output_variables}
    accepted = 0

    for _ in range(50):
        if accepted >= count:
            break
        batch_size = max(1024, 2 * (count - accepted))
        samples = _draw(rng, ranges, batch_size)
        target, mechanism_values, valid = _evaluate(problem, samples)
        if not valid.any():
            continue
        take = min(count - accepted, int(valid.sum()))
        selected = np.flatnonzero(valid)[:take]
        for variable in output_variables:
            if variable.name == problem.target_variable.name:
                values = target
            elif variable.name in samples:
                values = samples[variable.name]
            elif variable.name in mechanism_values:
                values = mechanism_values[variable.name]
            else:
                raise ValueError(
                    f"Cannot generate values for variable {variable.name!r}."
                )
            accepted_values[variable.name].append(
                np.real(values[selected]).astype(float)
            )
        accepted += take

    if accepted < count:
        raise ValueError(
            f"Only generated {accepted}/{count} valid samples after 50 attempts. "
            "Please specify a physically appropriate sampling range for this problem."
        )

    return np.vstack([
        np.concatenate(accepted_values[variable.name])
        for variable in output_variables
    ])


def generate_synthetic_data(
    problem: Problem,
    *,
    seed: int = 20260725,
    train_samples: int = 1000,
    id_test_samples: int = 1000,
    ood_test_samples: int = 1000,
    pilot_samples: int = 10000,
    range_inferrer: RangeInferrer | None = None,
) -> dict[str, np.ndarray]:
    """Generate an NPZ-compatible dictionary without reading or writing files."""
    if min(train_samples, id_test_samples, ood_test_samples, pilot_samples) <= 0:
        raise ValueError("All sample counts must be positive integers.")
    if not problem.solution:
        problem.solution = solve_mechanism_equations(problem)
    if range_inferrer is None:
        range_inferrer = RangeInferrer()

    feature_variables = [
        problem.target_variable,
        *problem.input_variables,
        *problem.auxiliary_input_variables,
    ]
    produced = {variable.name for variable in problem.intermediate_variables}
    source_names = [
        variable.name
        for variable in [*problem.input_variables, *problem.auxiliary_input_variables]
        if variable.name not in produced
    ]
    if len(source_names) != len(set(source_names)):
        duplicated = [name for name in sorted(set(source_names)) if source_names.count(name) > 1]
        raise ValueError(f"Duplicate variable names found while generating synthetic data: {duplicated!r}")

    rng = np.random.default_rng(seed)
    inferred = {
        name: spec
        for name, spec in range_inferrer.infer(problem).items()
        if name in source_names
    }
    fallback_ranges = {name: (spec[0], spec[3], spec[4]) for name, spec in inferred.items()}
    pilot = _draw(rng, fallback_ranges, pilot_samples)
    _, _, valid = _evaluate(problem, pilot)
    valid_count = int(valid.sum())
    logger.info("Pilot sampling: %s/%s samples are finite and real.", valid_count, pilot_samples)
    if valid_count < max(10, len(source_names) + 1):
        raise ValueError(
            f"Only {valid_count}/{pilot_samples} pilot samples are finite. "
            "The configured sampling ranges do not produce enough finite samples."
        )

    low_ranges = {name: (spec[0], spec[1], spec[4]) for name, spec in inferred.items()}
    high_ranges = {name: (spec[2], spec[3], spec[4]) for name, spec in inferred.items()}

    train = _generate_split(problem, rng, low_ranges, train_samples, feature_variables)
    id_test = _generate_split(problem, rng, low_ranges, id_test_samples, feature_variables)
    ood_test = _generate_split(problem, rng, high_ranges, ood_test_samples, feature_variables)

    return {
        "train": train,
        "id_test": id_test,
        "ood_test": ood_test,
        "variables": np.asarray([variable.name for variable in feature_variables]),
        "generation_config": np.asarray(json.dumps({
            "seed": seed,
            "train_samples": train_samples,
            "id_test_samples": id_test_samples,
            "ood_test_samples": ood_test_samples,
            "pilot_samples": pilot_samples,
        })),
    }


def format_reports(reports: list[dict]) -> str:
    lines = ["[blue bold]MDBench · Synthetic data[reset]"]
    for report_index, report in enumerate(reports):
        if report_index:
            lines.append("[gray]" + "─" * 72 + "[reset]")
        lines.append(f"[cyan bold]{report['problem_name']}[reset]")
        lines.append(f"  [blue bold]Path[reset]: {report['path']}")
    return tag2ansi("\n".join(lines))


def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(description="Generate synthetic benchmark data")
    parser.add_argument("--problems", nargs="*", default=["problems"], help="Problem files or directories")
    parser.add_argument("--output-dir", default=str(DEFAULT_SYNTHETIC_DATA_DIR))
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--train-samples", type=int, default=1000)
    parser.add_argument("--id-test-samples", type=int, default=1000)
    parser.add_argument("--ood-test-samples", type=int, default=1000)
    parser.add_argument(
        "--pilot-samples",
        type=int,
        default=10000,
        help="Number of full-range trial samples used to check that formulas produce finite real values",
    )
    return parser


def main(args):
    output_dir = Path(args.output_dir)
    reports = []
    used_names: set[str] = set()
    for problem_path in discover_yaml_files(args.problems):
        problem = load_problem(str(problem_path))
        problem.solution = solve_mechanism_equations(problem)
        name = safe_name(problem.problem_name)
        if name in used_names:
            raise ValueError(f"Duplicate problem output name: {name}")
        used_names.add(name)

        data = generate_synthetic_data(
            problem,
            seed=args.seed,
            train_samples=args.train_samples,
            id_test_samples=args.id_test_samples,
            ood_test_samples=args.ood_test_samples,
            pilot_samples=args.pilot_samples,
        )
        output_path = output_dir / f"{name}.npz"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **data)
        reports.append({
            "problem_name": problem.problem_name,
            "problem_path": str(problem_path),
            "path": str(output_path),
        })
    report = format_reports(reports)
    logger.info(report)
    return 0
