"""Executable examples of how mechanism equations are loaded and validated."""
import math

import pytest
from src.features.evaluation import build_submission_problem
from src.features.io import (
    evaluate_solution,
    load_submission,
    parse_mechanism_equation,
    solve_mechanism_equations,
)
from src.features.io.solve_mechanism_equations import _evaluate_numerically


def test_load_submission_reads_semicolon_separated_equations_verbatim():
    answer = {"source_variables": ["a", "k"], "target_variable": "F"}

    submission = load_submission(
        "r = a; F = k * r^2",
        task="mechanism_discovery",
        answer=answer,
    )

    assert [item["formula"] for item in submission["mechanisms"]] == [
        "r = a",
        "F = k * r^2",
    ]


def test_load_submission_reads_one_equation_per_nonempty_text_line(tmp_path):
    path = tmp_path / "submission.txt"
    path.write_text("r = a\n\nF = k * r^2\n", encoding="utf-8")
    answer = {"source_variables": ["a", "k"], "target_variable": "F"}

    submission = load_submission(
        str(path), task="mechanism_discovery", answer=answer
    )

    assert [item["formula"] for item in submission["mechanisms"]] == [
        "r = a",
        "F = k * r^2",
    ]


def test_two_equations_are_solved_symbolically_for_two_unresolved_variables():
    answer = {"source_variables": ["x"], "target_variable": "a"}
    submission = load_submission(
        "a + b = 3 * x; a - b = x",
        task="mechanism_discovery",
        answer=answer,
    )
    problem = build_submission_problem(submission["mechanisms"], answer)
    problem.solution = solve_mechanism_equations(problem)
    solution = problem.solution[0]

    assert dict(zip(solution.variables, solution.formulas, strict=True)) == {
        "a": "2 * x",
        "b": "x",
    }


def test_coupled_equations_are_solved_in_successive_dependency_groups():
    answer = {"source_variables": ["x"], "target_variable": "y"}
    submission = load_submission(
        "a + b = 3*x; a - b = x; "
        "c + d = 5*b; c - d = b; "
        "y = c + d",
        task="mechanism_discovery",
        answer=answer,
    )
    problem = build_submission_problem(submission["mechanisms"], answer)
    problem.solution = solve_mechanism_equations(problem)

    solved_formulas = [
        dict(zip(step.variables, step.formulas, strict=True))
        for step in problem.solution
    ]
    assert solved_formulas == [
        {"a": "2 * x", "b": "x"},
        {"c": "3 * b", "d": "2 * b"},
        {"y": "c + d"},
    ]


def test_equation_without_a_closed_form_falls_back_to_numerical_solving():
    answer = {"source_variables": ["x"], "target_variable": "a"}
    submission = load_submission(
        "a = cos(a) + x",
        task="mechanism_discovery",
        answer=answer,
    )
    problem = build_submission_problem(submission["mechanisms"], answer)
    problem.solution = solve_mechanism_equations(problem)

    assert problem.solution[0].formulas == []

    result = evaluate_solution(problem, {"x": 0.2})
    assert result["a"][0] == pytest.approx(math.cos(result["a"][0]) + 0.2)


def test_equation_with_no_solution_fails_during_numerical_evaluation():
    answer = {"source_variables": [], "target_variable": "a"}
    submission = load_submission(
        "a = a + 1",
        task="mechanism_discovery",
        answer=answer,
    )
    problem = build_submission_problem(submission["mechanisms"], answer)
    problem.solution = solve_mechanism_equations(problem)

    with pytest.raises(ValueError) as error:
        evaluate_solution(problem, {})

    assert "Numerical solution failed for mechanism group [1]" in str(error.value)
    assert "maximum scaled residual is 1" in str(error.value)


def test_implicit_solver_seeds_intermediates_from_known_target():
    m = 9.1093837e-31
    q = 1.602176634e-19
    eps = 8.8541878128e-12
    hbar = 1.054571817e-34
    magnetic_field = 5.0e4
    coefficient = m * q**2 / (4 * math.pi * eps)
    radius = 2 * hbar**2 / (
        coefficient
        + math.sqrt(coefficient**2 + 4 * q * magnetic_field * hbar**3)
    )
    equations = [
        parse_mechanism_equation(
            "F_C = q * q / (4 * 3.141592653589793 * eps * r**2)"
        ),
        parse_mechanism_equation("F_B = q * v * B"),
        parse_mechanism_equation("F_in = m * v**2 / r"),
        parse_mechanism_equation("F_in = F_C + F_B"),
        parse_mechanism_equation("L = m * v * r"),
    ]

    _, solved_radius, _, _, _ = _evaluate_numerically(
        ["F_C", "r", "F_B", "v", "F_in"],
        equations,
        [1, 3, 4, 5, 6],
        {
            "m": m,
            "q": q,
            "eps": eps,
            "B": magnetic_field,
            "L": hbar,
            "r": radius,
        },
    )

    assert solved_radius[0] == pytest.approx(radius, rel=1e-8)


def test_load_submission_rejects_fewer_equations_than_unresolved_variables():
    answer = {"source_variables": ["x"], "target_variable": "a"}

    with pytest.raises(ValueError) as error:
        load_submission(
            "a + b = x",
            task="mechanism_discovery",
            answer=answer,
        )

    assert str(error.value) == (
        "Cannot solve remaining mechanism [1]: the system is underdetermined "
        "(1 equation for 2 unresolved variables: a, b)."
    )


def test_load_submission_requires_exactly_one_equals_sign_per_equation():
    answer = {"source_variables": ["x"], "target_variable": "a"}

    with pytest.raises(ValueError, match="exactly one '='"):
        load_submission(
            "a = x = 1",
            task="mechanism_discovery",
            answer=answer,
        )
