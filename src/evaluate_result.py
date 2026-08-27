"""Score formula and mechanism submissions against a private answer."""
from __future__ import annotations
import json
import argparse
import numpy as np
from pathlib import Path
from .metrics import (
    LLMMechanismFundamentalityScorer,
    NumericEquivalenceChecker,
    StructuralMechanismMatcher,
    MechanismSimplicityScorer,
)
from .core import Problem
from .prepare_problem import TASKS
from .utils import tag2ansi, logger
from .features.sampling import RangeInferrer
from .metrics import HybridFormulaEquivalenceChecker
from .features.io import evaluate_solution, load_submission, solve_mechanism_equations
from .features.evaluation import (
    build_submission_problem,
    load_public_task,
    public_answer_context,
    trace_mechanism_submission,
    training_values,
)


def resolve_task(requested_task: str | None, answer: dict) -> str:
    """Resolve the evaluation task and reject contradictory metadata."""
    answer_task = answer.get("task")
    if requested_task and answer_task and requested_task != answer_task:
        raise ValueError(
            f"Requested task {requested_task!r} does not match answer task {answer_task!r}."
        )
    task = requested_task or answer_task
    if not task:
        raise ValueError("Task is required because the answer does not declare one.")
    return task


def _fundamentality_report(scorer, mechanisms, *, answer, problem=None) -> dict:
    """Return an explicit unavailable report when the configured LLM fails."""
    try:
        return scorer.compare(mechanisms, answer=answer, problem=problem)
    except Exception as exc:
        return {
            "score": None,
            "items": [],
            "provider": getattr(scorer, "llm_provider", None),
            "model": getattr(scorer, "llm_model", None),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _derivation_report(submission, context, values, checker=None) -> dict:
    """Compare a mechanism's derived target with the supplied target relationship."""
    trace = trace_mechanism_submission(submission["mechanisms"], context)
    derived_formula = trace["derived_formula"]
    if derived_formula is not None:
        candidate_formula = f"{context['target_variable']} = {derived_formula}"
        report = (checker or HybridFormulaEquivalenceChecker(max_retries=0)).compare(
            candidate_formula,
            context["phenomenological_formula"],
            values,
        )
        report["derived_formula"] = candidate_formula
        return report

    expected = values.get(context["target_variable"])
    if expected is None:
        raise ValueError(
            "A numerical implicit mechanism requires observed target values for "
            "derivation-equivalence evaluation."
        )
    actual = _mechanism_predictions(submission, context, values)
    metrics = _prediction_metrics(actual, expected, split="numeric_derivation")
    numeric = {
        "equivalent": metrics["all_close"],
        "score": metrics["acc"],
        "tested": metrics["tested"],
        "metrics": metrics,
    }
    return {
        "equivalent": numeric["equivalent"],
        "score": numeric["score"],
        "reason": (
            "No closed-form target was available; equivalence was checked "
            "numerically on observed values."
        ),
        "symbolic": {
            "equivalent": None,
            "reason": "Unavailable for a numerical implicit solution.",
        },
        "numeric": numeric,
        "llm": {
            "equivalent": None,
            "provider": getattr(checker, "llm_provider", None),
            "model": getattr(checker, "llm_model", None),
            "reason": "Unavailable without a closed-form derived equation.",
        },
        "derived_formula": None,
    }


def validate_result(
    submission: dict,
    answer: dict,
    *,
    data_path: str | Path | None = None,
    formula_values: dict | None = None,
    problem: Problem | None = None,
    formula_checker=None,
    derivation_checker=None,
    mechanism_matcher=None,
    fundamentality_scorer=None,
    fundamentality_context: dict | None = None,
) -> dict:
    report = {
        "formula": None,
        "data_accuracy": None,
        "derivation": None,
        "structure_recovery": None,
        "mechanism_fundamentality": None,
        "mechanism_simplicity": None,
    }
    if "phenomenological_formula" in submission:
        values = dict(formula_values or {})
        if not values and data_path:
            with np.load(data_path) as data:
                names = data["variables"].tolist()
                target = answer.get("target_variable", names[0])
                values = {name: data["id_test"][i] for i, name in enumerate(names) if name != target}
        elif problem is not None:
            ranges = RangeInferrer().infer(problem)
            values = {
                variable.name: np.linspace(ranges[variable.name][0], ranges[variable.name][3], 128)
                for variable in problem.input_variables
            }
        constants = answer.get("constants", [])
        if isinstance(constants, dict):
            values.update(constants)
        else:
            values.update({constant["name"]: constant["value"] for constant in constants})
        report["formula"] = (formula_checker or NumericEquivalenceChecker()).compare(
            submission["phenomenological_formula"], answer["phenomenological_formula"], values)
    if "mechanisms" in submission:
        derivation_values = dict(formula_values or {})
        if not derivation_values and problem is not None:
            ranges = RangeInferrer().infer(problem)
            derivation_values = {
                name: np.linspace(specification[0], specification[3], 128)
                for name, specification in ranges.items()
            }
            derivation_values.update({
                constant.name: constant.value for constant in problem.constants
            })
        report["derivation"] = _derivation_report(
            submission, answer, derivation_values, derivation_checker
        )
        report["structure_recovery"] = (mechanism_matcher or StructuralMechanismMatcher()).compare(
            submission["mechanisms"], answer["mechanisms"])
        report["mechanism_simplicity"] = MechanismSimplicityScorer().compare(
            submission["mechanisms"]
        )
        if fundamentality_scorer is not None:
            report["mechanism_fundamentality"] = _fundamentality_report(
                fundamentality_scorer,
                submission["mechanisms"],
                answer=fundamentality_context or answer,
                problem=problem,
            )
    return report


def _prediction_metrics(actual, expected, *, split: str) -> dict:
    """Compute standard regression metrics on paired finite predictions."""
    actual, expected = np.broadcast_arrays(
        np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    )
    finite = np.isfinite(actual) & np.isfinite(expected)
    if not finite.any():
        return {
            "split": split,
            "pearson_r": float("nan"),
            "r2": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
            "smape": float("nan"),
            "acc": 0.0,
            "all_close": False,
            "tested": 0,
        }
    actual, expected = actual[finite], expected[finite]
    residual = actual - expected
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    denominator = np.abs(actual) + np.abs(expected)
    smape = float(np.mean(np.divide(
        2 * np.abs(residual),
        denominator,
        out=np.zeros_like(residual, dtype=float),
        where=denominator != 0,
    )))
    total_variation = float(np.sum((expected - expected.mean()) ** 2))
    r2 = (
        float(1 - np.sum(residual**2) / total_variation)
        if total_variation > 0
        else (1.0 if np.all(residual == 0) else float("nan"))
    )
    pearson_r = (
        float(np.corrcoef(actual, expected)[0, 1])
        if actual.size > 1 and np.std(actual) > 0 and np.std(expected) > 0
        else float("nan")
    )
    close = np.isclose(actual, expected, rtol=1e-6, atol=1e-9)
    return {
        "split": split,
        "pearson_r": pearson_r,
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "smape": smape,
        "acc": float(close.mean()),
        "all_close": bool(close.all()),
        "tested": int(close.size),
    }


def _formula_predictions(formula: str, values: dict) -> np.ndarray:
    import nd2py as nd

    expression = formula.split("=", 1)[-1].strip().replace("^", "**")
    return np.asarray(nd.parse(expression).eval(values), dtype=float)


def _mechanism_predictions(submission: dict, context: dict, values: dict) -> np.ndarray:
    candidate_context = dict(context)
    candidate_context.setdefault(
        "phenomenological_formula", context["target_variable"]
    )
    candidate = build_submission_problem(submission["mechanisms"], candidate_context)
    candidate.solution = solve_mechanism_equations(candidate)
    evaluated = evaluate_solution(candidate, dict(values))
    target = context["target_variable"]
    if target not in evaluated:
        raise ValueError(f"Submitted mechanisms do not produce target variable {target!r}.")
    return np.asarray(evaluated[target], dtype=float)


def evaluate_feedback(
    submission: dict,
    public_problem: dict,
    data_train: np.ndarray | None,
    *,
    fundamentality_scorer=None,
    derivation_checker=None,
) -> dict:
    """Evaluate using only the public task contract and training observations."""
    task = public_problem["task"]
    context = public_answer_context(public_problem)
    if task in {"symbolic_regression", "mechanism_discovery"}:
        if data_train is None:
            raise ValueError(f"Task {task!r} requires public training data.")
        values, expected = training_values(public_problem, data_train)
    else:
        values = {
            item["name"]: item["value"]
            for item in public_problem.get("constants", [])
        }
        expected = None
    report = {
        "evaluation_mode": "feedback",
        "formula": None,
        "data_accuracy": None,
        "derivation": None,
        "structure_recovery": None,
        "mechanism_fundamentality": None,
        "mechanism_simplicity": None,
    }
    if task == "symbolic_regression":
        actual = _formula_predictions(submission["phenomenological_formula"], values)
        report["data_accuracy"] = {
            "splits": [_prediction_metrics(actual, expected, split="train")]
        }
        return report

    if task == "mechanism_discovery":
        actual = _mechanism_predictions(submission, context, values)
        report["data_accuracy"] = {
            "splits": [_prediction_metrics(actual, expected, split="train")]
        }
    elif task == "mechanism_explanation":
        report["derivation"] = _derivation_report(
            submission, context, values, derivation_checker
        )

    report["mechanism_simplicity"] = MechanismSimplicityScorer().compare(
        submission["mechanisms"]
    )
    if fundamentality_scorer is not None:
        report["mechanism_fundamentality"] = _fundamentality_report(
            fundamentality_scorer,
            submission["mechanisms"], answer=context, problem=None
        )
    return report


def load_json(path: str | Path) -> dict:
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path) as data:
            if "answer_json" not in data:
                raise ValueError(f"NPZ file does not contain answer_json: {path}")
            return json.loads(str(data["answer_json"].item()))
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _load_private_splits(path: str | Path, answer: dict) -> list[tuple[str, dict, np.ndarray]]:
    """Load train and hidden test splits colocated with a private answer artifact."""
    path = Path(path)
    public_path = path if path.suffix.lower() == ".npz" else path.parent
    public_problem, data_train = load_public_task(public_path)
    if data_train is None:
        return []
    train_values, train_expected = training_values(public_problem, data_train)
    results = [("train", train_values, train_expected)]
    arrays = {}
    if path.suffix.lower() == ".npz":
        with np.load(path) as archive:
            for split, key in (("id_test", "data_id_test"), ("ood_test", "data_ood_test")):
                if key in archive:
                    arrays[split] = archive[key].copy()
    else:
        for split, filename in (
            ("id_test", "data_id_test.npy"),
            ("ood_test", "data_ood_test.npy"),
        ):
            candidate = path.with_name(filename)
            if candidate.is_file():
                arrays[split] = np.load(candidate)
    names = answer.get("data_variables") or [
        *[item["name"] for item in public_problem.get("variables", [])],
        *[
            item["name"]
            for item in public_problem.get("auxiliary_input_variables", [])
        ],
    ]
    for split, data in arrays.items():
        if data.ndim != 2 or data.shape[0] != len(names):
            raise ValueError(
                f"Private {split} data has shape {data.shape}; expected "
                f"{len(names)} rows from answer metadata."
            )
        values = {name: data[index] for index, name in enumerate(names)}
        values.update({item["name"]: item["value"] for item in answer.get("constants", [])})
        results.append((split, values, np.asarray(values[answer["target_variable"]])))
    return results


def format_reports(report: dict) -> str:
    def percent(value) -> str:
        return f"{100 * float(value):.2f}%"

    lines = ["[blue bold]MDBench · Result evaluation[reset]"]
    if formula := report.get("formula"):
        lines.append(
            "  [cyan]Private phenomenological-equation equivalence[reset]: "
            f"{percent(formula['score'])} (equivalent={formula.get('equivalent')})"
        )
        if symbolic := formula.get("symbolic"):
            lines.append(
                f"    SymPy proof: {symbolic.get('equivalent')} · "
                f"{symbolic.get('reason')}"
            )
        if numeric := formula.get("numeric"):
            lines.append(
                f"  [cyan]Numeric judgment[reset]: {numeric.get('equivalent')} "
                f"(agreement={percent(numeric.get('score', 0.0))}, "
                f"tested={numeric.get('tested', 0)})"
            )
        if llm := formula.get("llm"):
            lines.append(
                f"  [cyan]LLM judgment[reset]: {llm.get('equivalent')} "
                f"(provider={llm.get('provider')}, model={llm.get('model')})"
            )
            for error in llm.get("errors", []):
                lines.append(f"  [red]LLM error[reset]: {error}")
        if formula.get("reason"):
            lines.append(f"  [cyan]Reason[reset]: {formula['reason']}")
    if accuracy := report.get("data_accuracy"):
        splits = accuracy["splits"]
        split_names = {
            "train": "Train",
            "id_test": "In-Domain Test",
            "ood_test": "Out-of-Domain Test",
        }
        lines.append("  [cyan]Prediction Accuracy[reset]:")
        if len(splits) > 1:
            lines.append(
                "    (" + " | ".join(
                    split_names.get(item["split"], item["split"])
                    for item in splits
                ) + ")"
            )

        def metric_row(label, key, formatter):
            values = " | ".join(formatter(item[key]) for item in splits)
            lines.append(f"    {label}: {values}")

        metric_row("Pearson r", "pearson_r", lambda value: f"{value:.6f}")
        metric_row("R2", "r2", lambda value: f"{value:.6f}")
        metric_row("MAE", "mae", lambda value: f"{value:.6g}")
        metric_row("RMSE", "rmse", lambda value: f"{value:.6g}")
        metric_row("sMAPE", "smape", percent)
        metric_row("Acc (rtol=1e-6, atol=1e-9)", "acc", percent)
    if derivation := report.get("derivation"):
        lines.append(
            "  [cyan]Derived-equation equivalence[reset]: "
            f"{percent(derivation['score'])} "
            f"(equivalent={derivation['equivalent']})"
        )
        lines.append(
            f"    Derived equation: {derivation.get('derived_formula') or 'no closed form'}"
        )
        if symbolic := derivation.get("symbolic"):
            lines.append(
                f"    SymPy proof: {symbolic.get('equivalent')} · "
                f"{symbolic.get('reason')}"
            )
        if numeric := derivation.get("numeric"):
            lines.append(
                f"    Numeric agreement: {percent(numeric.get('score', 0.0))} "
                f"({numeric.get('tested', 0)} finite points; "
                f"all agree={numeric.get('equivalent')})"
            )
        if llm := derivation.get("llm"):
            lines.append(
                f"    LLM judgment: {llm.get('equivalent')} "
                f"(provider={llm.get('provider')}, model={llm.get('model')})"
            )
            if llm.get("reason"):
                lines.append(f"      {llm['reason']}")
            for error in llm.get("errors", []):
                lines.append(f"      [red]LLM error[reset]: {error}")
    if recovery := report.get("structure_recovery"):
        lines.append(f"  [cyan]Ground-truth structure recovery[reset]: {percent(recovery['score'])}")
        lines.append(
            f"    formula={percent(recovery['formula_similarity'])}, "
            f"DAG={percent(recovery['dag_similarity'])}"
        )
    if trace := report.get("mechanism_trace"):
        lines.append("  [cyan]Mechanism trace[reset]")
        line_index = 1
        for index, step in enumerate(trace["solution_steps"], 1):
            variables = ", ".join(step["variables"])
            if step["numerical"]:
                lines.append(
                    f"    {line_index}. {variables}: solved numerically as an "
                    "implicit equation system"
                )
                line_index += 1
                continue
            for variable, original, expanded, show_expanded in zip(
                step["variables"],
                step["original_formulas"],
                step["expanded_formulas"],
                step["show_expanded_formulas"],
            ):
                equation = f"{variable} = {original}"
                if show_expanded:
                    equation += f" = {expanded}"
                lines.append(f"    {line_index}. {equation}")
                line_index += 1
    if fundamentality := report.get("mechanism_fundamentality"):
        if fundamentality.get("score") is None:
            lines.append(
                "  [cyan]Mechanism fundamentality[reset]: unavailable "
                f"(evaluator: {fundamentality.get('model')} @ "
                f"{fundamentality.get('provider')})"
            )
            lines.append(f"    [red]{fundamentality.get('error')}[reset]")
            fundamentality = None
    if fundamentality:
        lines.append(
            "  [cyan]Mechanism fundamentality score[reset]: "
            f"{percent(fundamentality['score'])} "
            f"(evaluator: {fundamentality['model']} @ "
            f"{fundamentality['provider']})"
        )
        lines.append(
            "    bottleneck="
            f"{percent(fundamentality['minimum_item_score'])}, "
            f"mean={percent(fundamentality['mean_item_score'])}"
        )
        bottleneck_weight = fundamentality["bottleneck_weight"]
        lines.append(
            "    fundamentality score = "
            f"{bottleneck_weight:.2f} * bottleneck + "
            f"{1 - bottleneck_weight:.2f} * mean"
        )
        for item in fundamentality["items"]:
            lines.append(
                f"    {item['index']}. fundamentality={percent(item['score'])} "
                f"({item['judgment']}): {item['reason']}"
            )
            if item.get("preferred_relation"):
                lines.append(
                    f"      Preferred relationship: {item['preferred_relation']}"
                )
    if simplicity := report.get("mechanism_simplicity"):
        lines.append(
            "  [cyan]Mechanism description complexity[reset] "
            "(reference-free; lower is simpler):"
        )
        lines.append(
            f"    mean={simplicity['mean_ast_nodes_per_item']:.2f} AST nodes/item, "
            f"maximum={simplicity['maximum_ast_nodes']}, "
            f"total={simplicity['total_ast_nodes']} across "
            f"{simplicity['item_count']} relationships"
        )
    return tag2ansi("\n".join(lines))


def get_parser(parser=None):
    preliminary_parser = argparse.ArgumentParser(add_help=False)
    preliminary_parser.add_argument("--evaluation-mode", choices=("feedback", "final"))
    preliminary_parser.add_argument("--task", choices=sorted(TASKS))
    preliminary_source = preliminary_parser.add_mutually_exclusive_group()
    preliminary_source.add_argument("--answer")
    preliminary_source.add_argument("--problem")
    preliminary_args, _ = preliminary_parser.parse_known_args()

    if parser is None:
        parser = argparse.ArgumentParser(description="Evaluate a benchmark submission")
    parser.add_argument("--evaluation-mode", choices=("feedback", "final"), required=True, help=(
        "'feedback' uses only a prepared public task and training data; "
        "'final' uses a private answer and hidden test artifacts"
    ))
    parser.add_argument("--task", choices=sorted(TASKS))
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False, help=(
        "Show a concise equation chain; append a fully expanded right-hand "
        "side when an intermediate-dependent formula has a closed form"
    ))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--answer", help=(
        "Private answer JSON or prepared NPZ (final mode only)"
    ))
    source.add_argument("--problem", help=(
        "Prepared public task directory, problem.json, or task NPZ "
        "(feedback mode only)"
    ))
    if preliminary_args.evaluation_mode == "feedback":
        if not preliminary_args.problem:
            parser.error("--evaluation-mode feedback requires --problem")
        if preliminary_args.answer:
            parser.error("--answer is not accepted in feedback mode")
    if preliminary_args.evaluation_mode == "final":
        if preliminary_args.problem:
            parser.error("--problem is not accepted in final mode")
        if not preliminary_args.answer:
            parser.error("--evaluation-mode final requires --answer")


    task = preliminary_args.task
    if task is None and preliminary_args.answer:
        task = load_json(preliminary_args.answer).get("task")
    if task is None and preliminary_args.problem:
        task = load_public_task(preliminary_args.problem)[0].get("task")

    if task in ["symbolic_regression"]:
        parser.add_argument("--submission", type=str, required=True, help=(
            "A phenomenological formula string, optionally written as "
            "'target_variable = formula', or a plain-text file containing "
            "the formula on one non-empty line."
        ))
    else:
        parser.add_argument("--submission", type=str, required=True, help=(
            "Mechanism equations separated by semicolons, or a plain-text "
            "file containing one mechanism equation per non-empty line. "
            "Each relationship must use 'variable = formula'; a zero-valued "
            "left side is not currently supported."
        ))

    if preliminary_args.evaluation_mode == "final" and task in ["symbolic_regression", "mechanism_discovery"]:
        parser.add_argument("--data", help=(
            "Optional NPZ dataset supplying numerical values for formula evaluation."
        ))

    if task == "symbolic_regression":
        parser.add_argument("--llm-provider", default="openrouter", help=(
            "LLM provider used for semantic formula-equivalence judgment."
        ))
        parser.add_argument("--llm-model", default="deepseek/deepseek-v4-flash", help=(
            "LLM model used for semantic formula-equivalence judgment."
        ))
    elif task in ["mechanism_explanation", "mechanism_discovery"]:
        parser.add_argument("--llm-provider", default="deepseek", help=(
            "LLM provider used for mechanism fundamentality and private "
            "derived-equation equivalence judgments."
        ))
        parser.add_argument("--llm-model", default="deepseek-v4-flash", help=(
            "LLM model used for mechanism fundamentality and private "
            "derived-equation equivalence judgments."
        ))
    return parser


def main(args):
    try:
        if args.evaluation_mode == "feedback":
            if not args.problem or args.answer:
                raise ValueError("Feedback evaluation requires --problem and forbids --answer.")
            public_problem, data_train = load_public_task(args.problem)
            answer = public_answer_context(public_problem)
            task = resolve_task(args.task, answer)
            problem = None
        else:
            if not args.answer or args.problem:
                raise ValueError("Final evaluation requires --answer and forbids --problem.")
            answer = load_json(args.answer)
            task = resolve_task(args.task, answer)
            problem = None
            private_splits = _load_private_splits(args.answer, answer)
            public_path = (
                Path(args.answer)
                if Path(args.answer).suffix.lower() == ".npz"
                else Path(args.answer).parent
            )
            public_problem, _ = load_public_task(public_path)
            fundamentality_context = public_answer_context(public_problem)
        formula_checker = None
        if task == "symbolic_regression":
            formula_checker = HybridFormulaEquivalenceChecker(
                llm_provider=args.llm_provider,
                llm_model=args.llm_model,
            )
        fundamentality_scorer = None
        derivation_checker = None
        if task in {"mechanism_explanation", "mechanism_discovery"}:
            fundamentality_scorer = LLMMechanismFundamentalityScorer(
                llm_provider=args.llm_provider,
                llm_model=args.llm_model,
            )
            derivation_checker = HybridFormulaEquivalenceChecker(
                llm_provider=args.llm_provider,
                llm_model=args.llm_model,
            )
        submission = load_submission(args.submission, task=task, answer=answer)
        if args.evaluation_mode == "feedback":
            report = evaluate_feedback(
                submission,
                public_problem,
                data_train,
                fundamentality_scorer=fundamentality_scorer,
                derivation_checker=derivation_checker,
            )
        else:
            report = validate_result(
                submission,
                answer, data_path=getattr(args, "data", None), problem=problem,
                formula_values=(private_splits[0][1] if private_splits else None),
                formula_checker=formula_checker,
                derivation_checker=derivation_checker,
                fundamentality_scorer=fundamentality_scorer,
                fundamentality_context=fundamentality_context,
            )
            report["evaluation_mode"] = "final"
            if task in {"symbolic_regression", "mechanism_discovery"}:
                split_reports = []
                for split, values, expected in private_splits:
                    if "phenomenological_formula" in submission:
                        actual = _formula_predictions(submission["phenomenological_formula"], values)
                    else:
                        actual = _mechanism_predictions(submission, answer, values)
                    split_reports.append(_prediction_metrics(actual, expected, split=split))
                report["data_accuracy"] = {"splits": split_reports}
        if "mechanisms" in submission:
            report["mechanism_trace"] = trace_mechanism_submission(submission["mechanisms"], answer)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.error(tag2ansi(f"[red bold]Error:[reset] {exc}"))
        return 1
    report = format_reports(report)
    logger.info(report)
    return 0
