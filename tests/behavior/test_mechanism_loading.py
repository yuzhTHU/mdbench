"""Executable examples of how mechanism equations are loaded and validated."""
import math

import pytest
from src.features.evaluation import build_submission_problem
from src.features.io import evaluate_solution, load_submission, solve_mechanism_equations


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
