# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""Validate the structure, dimensions and derivation of a problem."""
import argparse
import nd2py as nd
from collections import Counter
from .core import Problem, UNIT
from .features.units import unit_inference
from .features.sampling import RangeInferrer
from .utils import discover_yaml_files, tag2ansi, logger
from .features.validation import (
    LLMMechanismFundamentalityChecker,
    NumericDerivationChecker,
)
from .features.io import load_problem, solve_mechanism_equations


class ValidationError(ValueError):
    pass


def _run_check(name, function) -> dict:
    try:
        return {"name": name, "ok": True, "detail": function()}
    except (ValidationError, ValueError, KeyError, ArithmeticError) as exc:
        return {"name": name, "ok": False, "detail": str(exc)}


def _check_variables(problem: Problem) -> str:
    all_vars = [*problem.all_variables, *problem.constants]
    all_var_names = [var.name for var in all_vars]
    if len(all_var_names) != len(set(all_var_names)):
        duplicates = sorted([item for item, count in Counter(all_var_names).items() if count > 1])
        raise ValidationError(f"Duplicate names across variables and constants: {', '.join(duplicates)}.")

    all_input_vars = [*problem.input_variables, *problem.auxiliary_input_variables]
    if missing_units := sorted(var.name for var in all_input_vars if var.unit is None):
        raise ValidationError(
            "Externally supplied variables must declare units: "
            f"{', '.join(missing_units)}."
        )

    all_derived_vars = [problem.target_variable, *problem.intermediate_variables]
    if sampled_internal := sorted(var.name for var in all_derived_vars if var.sampling is not None):
        raise ValidationError(
            "Mechanism-produced variables must not declare sampling specifications: "
            f"{', '.join(sampled_internal)}."
        )

    phenomeno_formula = nd.parse(problem.phenomenological_formula)
    constant_names = {constant.name for constant in problem.constants}
    expected = {
        var.name
        for var in phenomeno_formula.iter_preorder()
        if isinstance(var, nd.Variable)
    } - constant_names
    if (names := {var.name for var in problem.input_variables}) != expected:
        raise ValidationError(
            f"Input variables must be {sorted(expected)}, got {sorted(names)}."
        )

    used_var_names = {problem.target_variable.name, *expected}
    for item in problem.mechanism:
        used_var_names.add(item.variable)
        for var in nd.parse(item.formula).iter_preorder():
            if isinstance(var, nd.Variable):
                used_var_names.add(var.name)
    if unused := sorted(set(all_var_names) - used_var_names):
        raise ValidationError(
            "Declared variables or constants are unused by the phenomenological "
            f"or mechanism equations: {', '.join(unused)}."
        )

    return (
        f"Registered {1} target variable, "
        f"{len(problem.input_variables)} input variables, "
        f"{len(problem.intermediate_variables)} intermediate variables, "
        f"{len(problem.auxiliary_input_variables)} auxiliary input variables, "
        f"and "
        f"{len(problem.constants)} constants."
    )


def _check_sampling(problem: Problem) -> str:
    ranges = RangeInferrer().infer(problem)
    return f"Sampling specifications are valid for {len(ranges)} source variables."


def _check_fundamentality(problem: Problem, checker) -> str:
    """Check whether each mechanism is more fundamental than the target law.

    ``checker`` is deliberately injected so this heuristic judgment can later be
    replaced by another LLM, a physics knowledge base, or a curated evaluator.
    It accepts a ``Problem`` and returns the JSON-compatible report documented by
    ``MechanismFundamentalityChecker``.
    """
    report = checker.check(problem)
    rejected = [item for item in report["items"] if item["judgment"] == "not_fundamental"]
    uncertain = [item for item in report["items"] if item["judgment"] == "uncertain"]

    if rejected:
        details = []
        for item in rejected:
            detail = f"Mechanism {item['index']}: {item['reason']}"
            if item["preferred_relation"]:
                detail += f" Preferred relationship: {item['preferred_relation']}"
            details.append(detail)
        raise ValidationError(" ".join(details))

    for item in uncertain:
        logger.warning(
            f"Mechanism {item['index']} requires manual fundamentality review: "
            f"{item['reason']}"
        )

    return (
        f"Reviewed {len(report['items'])} mechanism relationships with "
        f"{report['provider']}/{report['model']}; "
        f"{len(uncertain)} require manual review."
    )


def _check_solution(problem: Problem, derivation_checker=None) -> str:
    """Solve all intermediates and verify that the solution derives the target."""
    problem.solution = solve_mechanism_equations(problem)

    solved_vars = {var for item in problem.solution for var in item.variables}
    intermediate_vars = {var.name for var in problem.intermediate_variables}
    if missing := sorted(intermediate_vars - solved_vars):
        raise ValidationError(
            f"No computation rule was solved for intermediate variables: "
            f"{', '.join(missing)}."
        )

    occurrences = {var.name: [] for var in problem.intermediate_variables}
    for idx, item in enumerate(problem.mechanism):
        names = {item.variable}
        for var in nd.parse(item.formula).iter_preorder():
            if isinstance(var, nd.Variable):
                names.add(var.name)
        for name in names:
            if name in occurrences:
                occurrences[name].append(idx)
    unused = []
    for name, indices in occurrences.items():
        if name == problem.target_variable.name:
            pass
        elif len(indices) > 1:
            pass
        elif len(indices) == 1:
            unused.append(name)
        else:
            raise ValidationError(f"Variable {name!r} is never used.")
    if unused:
        logger.warning(
            f"Mechanism-produced variables are never used by a later relationship: "
            f"{', '.join(unused)}."
        )

    if derivation_checker is None:
        derivation_checker = NumericDerivationChecker()
    derivation_report = derivation_checker.check(problem)
    if not derivation_report["equivalent"]:
        raise ValidationError("Mechanism solution does not derive the phenomenological equation.")

    numerical = sum(not item.formulas for item in problem.solution)
    return (
        f"Solved {len(problem.mechanism)} equations as "
        f"{len(problem.solution)} ordered solution steps; "
        f"{numerical} require numerical evaluation. "
        f"{derivation_report['detail']}"
    )


def _check_units(problem: Problem) -> str:
    """Check units of the phenomenological, mechanism, and solved formulas."""
    all_vars = [*problem.all_variables, *problem.constants]
    variable_by_name = {var.name: var for var in all_vars}
    errors = []

    target_unit = problem.target_variable.unit
    if target_unit is None:
        errors.append(
            f"Target variable {problem.target_variable.name!r} must declare a unit."
        )
    else:
        actual, inference_errors = unit_inference(
            problem.phenomenological_formula,
            all_vars,
        )
        if inference_errors:
            errors.extend(
                f"Phenomenological equation: {error}"
                for error in inference_errors
            )
        elif actual != target_unit.unit_dict:
            errors.append(
                "Phenomenological equation has unit "
                f"{UNIT(actual)}, but target {problem.target_variable.name!r} "
                f"has unit {target_unit}."
            )

    mechanism_skipped = 0
    for index, item in enumerate(problem.mechanism, 1):
        equation_names = {item.variable} | {
            node.name
            for node in nd.parse(item.formula).iter_preorder()
            if isinstance(node, nd.Variable)
        }
        if any(variable_by_name[name].unit is None for name in equation_names):
            mechanism_skipped += 1
            continue
        try:
            left_unit, left_errors = unit_inference(item.variable, all_vars)
            right_unit, right_errors = unit_inference(item.formula, all_vars)
            inference_errors = left_errors + right_errors
            if inference_errors:
                raise ValidationError("; ".join(inference_errors))
            if left_unit != right_unit:
                errors.append(
                    f"Mechanism {index} ({item.formula_description}) has mismatched "
                    f"units {UNIT(left_unit)} and {UNIT(right_unit)}."
                )
        except (ValidationError, ValueError) as exc:
            errors.append(f"Mechanism {index} ({item.formula_description}): {exc}")

    if not problem.solution:
        errors.append("No mechanism solution steps were produced.")

    solution_checked = 0
    for item in problem.solution:
        if not item.formulas:
            continue
        label = ", ".join(item.variables)
        for target_name, formula in zip(item.variables, item.formulas):
            if variable_by_name[target_name].unit is None:
                continue
            formula_names = {
                node.name
                for node in nd.parse(formula).iter_preorder()
                if isinstance(node, nd.Variable)
            }
            if any(variable_by_name[name].unit is None for name in formula_names):
                continue
            actual, inference_errors = unit_inference(formula, all_vars)
            expected = variable_by_name[target_name].unit
            if inference_errors:
                errors.append(f"Solution step [{label}]: " + "; ".join(inference_errors))
            elif actual != expected.unit_dict:
                errors.append(
                    f"Solution step [{label}] formula for {target_name!r} "
                    f"has unit {UNIT(actual)}, but expected {expected}."
                )
            solution_checked += 1

    if errors:
        raise ValidationError("; ".join(errors))
    mechanism_checked = len(problem.mechanism) - mechanism_skipped
    return (
        "Phenomenological equation, "
        f"{mechanism_checked} checkable mechanism equations, and "
        f"{solution_checked} closed-form solution formulas have valid units; "
        f"{mechanism_skipped} mechanism equations use variables without declared units."
    )


def format_reports(reports: list[dict]) -> str:
    lines = ["[blue bold]MDBench · Problem validation[reset]"]
    for report in reports:
        if report['ok']:
            lines.append(f"[cyan bold]{report['problem_name']}[reset]: [green]✓ ALL PASS[reset]")
        else:
            lines.append("[gray]" + "─" * 72 + "[reset]")
            lines.append(f"[cyan bold]{report['problem_name']}[reset]")
            for check in report["checks"]:
                marker = "[green]✓ PASS[reset]" if check["ok"] else "[red]✗ FAIL[reset]"
                lines.append(f"  {marker}  {check['name']}")
                lines.append(f"          {check['detail']}")
    return tag2ansi("\n".join(lines))


def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(description="Validate benchmark problem files")
    parser.add_argument("--problems", nargs="*", default=["problems"], help="Problem files or directories")
    parser.add_argument("--check-fundamentality", action="store_true", help=(
        "Send the problem context and mechanism equations to an LLM and "
        "check whether each relationship is sufficiently fundamental"
    ))
    parser.add_argument("--llm-provider", default="deepseek", help=(
        "LLM provider used by --check-fundamentality (default: deepseek)"
    ))
    parser.add_argument("--llm-model", default="deepseek-v4-flash", help=(
        "LLM model used by --check-fundamentality "
        "(default: deepseek-v4-flash)"
    ))
    return parser


def main(args):
    reports = []
    fundamentality_checker = LLMMechanismFundamentalityChecker(
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
    ) if args.check_fundamentality else None
    for path in discover_yaml_files(args.problems):
        problem = load_problem(path, solve=False)
        checks = []
        # Check declarations, uniqueness, and formula usage.
        checks.append(_run_check(
            "Variable definitions", 
            lambda: _check_variables(problem)
        ))
        if fundamentality_checker is not None:
            # Ask an LLM whether each relationship is more fundamental.
            checks.append(_run_check(
                "Mechanism fundamentality",
                lambda: _check_fundamentality(problem, fundamentality_checker),
            ))
        # Resolve mechanisms into executable solution steps.
        checks.append(_run_check(
            "Mechanism equation solving",
            lambda: _check_solution(problem),
        ))
        # Check units in target, mechanism, and solved formulas.
        checks.append(_run_check(
            "Physical units",
            lambda: _check_units(problem),
        ))
        # Check training and OOD sampling specifications.
        checks.append(_run_check(
            "Sampling specifications",
            lambda: _check_sampling(problem)
        ))
        passed = sum(check["ok"] for check in checks)
        failed = len(checks) - passed
        reports.append({
            "ok": all(check["ok"] for check in checks),
            "problem_name": problem.problem_name,
            "checks": checks,
            "passed": passed,
            "failed": failed,
        })
    logger.info(format_reports(reports))
    return 0 if all(report["ok"] for report in reports) else 1
