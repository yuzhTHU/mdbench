"""Fundamental mechanism relationships are normalized into executable groups."""

import nd2py as nd
import numpy as np
import pytest

from src.core import MechanismItem, Problem, UNIT, VariableSpec
from src.features.io import evaluate_solution, load_problem, solve_mechanism_equations
from src.features.validation import NumericDerivationChecker


def _variable(name):
    return VariableSpec(name, name, UNIT({}))


def _equation(left, right, description="relationship"):
    return MechanismItem(left, right, description)


def _implicit_problem():
    y, x, b = map(_variable, ("y", "x", "b"))
    x.sampling = {"min": 1, "max": 3, "ood_boundary": 2, "distribution": "uniform"}
    return Problem(
        "implicit", "implicit", "2 * x", y, [x], [b],
        [_equation("y", "x + b"), _equation("b", "y / 2")],
    )


def test_implicit_cycle_is_solved_symbolically():
    problem = _implicit_problem()
    solution = solve_mechanism_equations(problem)
    problem.solution = solution
    assert problem.solution is solution
    item = solution[0]
    assert item.variables == ["y", "b"]
    assert item.formulas
    result = item.function({"x": np.asarray([2.0, 3.0])})
    assert np.allclose(result[0], [4.0, 6.0])
    assert np.allclose(result[1], [2.0, 3.0])


def test_implicit_equations_need_not_be_contiguous():
    y, x, a, b, c = map(_variable, ("y", "x", "a", "b", "c"))
    problem = Problem(
        "noncontiguous",
        "noncontiguous",
        "4 * x + 1",
        y,
        [x],
        [a, b, c],
        [
            _equation("a", "x + b"),
            _equation("c", "a + 1"),
            _equation("b", "a / 2"),
            _equation("y", "a + c"),
        ],
    )

    solution = solve_mechanism_equations(problem)
    problem.solution = solution
    assert [item.variables for item in solution] == [["a", "b"], ["c"], ["y"]]
    assert NumericDerivationChecker().check(problem)["equivalent"]


def test_symbolically_resolved_cycle_derives_target():
    assert NumericDerivationChecker().check(_implicit_problem())["equivalent"]


def test_rewritten_balance_closes_pending_drag_relationship():
    variables = [
        _variable(name) for name in ("v", "F", "k", "F_d")
    ]
    variables[1].sampling = {"min": 1, "max": 3, "ood_boundary": 2, "distribution": "uniform"}
    variables[2].sampling = {"min": 1, "max": 3, "ood_boundary": 2, "distribution": "uniform"}
    problem = Problem(
        "balance", "balance", "sqrt(F / k)",
        variables[0], variables[1:3], variables[3:],
        [_equation("F_d", "k * v ** 2", "drag law"),
         _equation("F_d", "F", "force balance")],
    )
    solution = solve_mechanism_equations(problem)
    problem.solution = solution
    assert [item.variables for item in solution] == [["F_d"], ["v"]]
    values = evaluate_solution(
        problem,
        {"F": np.asarray([4.0, 9.0]), "k": np.asarray([1.0, 1.0])},
    )
    assert np.allclose(values["F_d"], [4.0, 9.0])
    assert np.allclose(values["v"], [2.0, 3.0])


def test_triangular_equations_are_solved_in_dependency_order():
    y, x, a1, a2 = map(_variable, ("y", "x", "a1", "a2"))
    problem = Problem(
        "triangular", "triangular", "4 * x + 2", y, [x], [a1, a2],
        [_equation("a2", "2 * a1"), _equation("a1", "x + 1"),
         _equation("y", "a2 + 2 * x")],
    )
    solution = solve_mechanism_equations(problem)
    assert [item.variables for item in solution] == [["a1"], ["a2"], ["y"]]
    assert NumericDerivationChecker().check(problem)["equivalent"]


def test_explicit_steps_are_stored_in_execution_order():
    problem = load_problem("demo_problem.yaml")
    solution = solve_mechanism_equations(problem)
    assert [item.variables for item in solution] == [
        ["r"], ["F"], ["acc"], ["v"], ["T"],
    ]
    assert all(item.formulas for item in solution)


def test_self_reference_uses_numerical_solver_when_no_closed_form_exists():
    a, x = _variable("a"), _variable("x")
    problem = Problem(
        "fixed point", "fixed point", "x", a, [x], [],
        [_equation("a", "cos(a) + x", "fixed point")],
    )
    item = solve_mechanism_equations(problem)[0]
    assert item.formulas == []
    result = item.function({"x": np.asarray([0.1, 0.2]), "a": np.asarray([1.0, 1.0])})[0]
    assert np.allclose(result, np.cos(result) + [0.1, 0.2], atol=1e-8)


def test_single_implicit_variable_is_solved_symbolically_when_unique():
    a, x = _variable("a"), _variable("x")
    problem = Problem(
        "single symbolic",
        "single symbolic",
        "x",
        a,
        [x],
        [],
        [_equation("a", "(x + a) / 2")],
    )

    item = solve_mechanism_equations(problem)[0]
    assert item.variables == ["a"]
    assert item.formulas
    assert np.allclose(item.function({"x": np.array([2.0, 3.0])})[0], [2.0, 3.0])


def test_multiple_symbolic_roots_choose_the_positive_branch():
    a, x = _variable("a"), _variable("x")
    problem = Problem(
        "multiple roots",
        "multiple roots",
        "sqrt(x)",
        a,
        [x],
        [],
        [_equation("x", "a ** 2")],
    )

    item = solve_mechanism_equations(problem)[0]
    assert item.formulas
    x_values = np.array([4.0, 9.0])
    result = item.function({"x": x_values})[0]
    assert np.allclose(result, [2.0, 3.0])
    assert np.all(result > 0)
    assert np.allclose(result**2, x_values)


def test_two_variable_transcendental_system_is_solved_numerically():
    a, b, x = map(_variable, ("a", "b", "x"))
    problem = Problem(
        "two numerical",
        "two numerical",
        "x",
        a,
        [x],
        [b],
        [
            _equation("a", "x + cos(b) / 10"),
            _equation("b", "sin(a) / 10"),
        ],
    )

    item = solve_mechanism_equations(problem)[0]
    assert item.variables == ["a", "b"]
    assert item.formulas == []
    x_values = np.array([0.1, 0.2])
    a_values, b_values = item.function({"x": x_values})
    assert np.allclose(a_values, x_values + np.cos(b_values) / 10, atol=1e-8)
    assert np.allclose(b_values, np.sin(a_values) / 10, atol=1e-8)


def test_three_variable_implicit_system_is_solved_symbolically():
    a, b, c, x = map(_variable, ("a", "b", "c", "x"))
    problem = Problem(
        "three symbolic",
        "three symbolic",
        "x",
        a,
        [x],
        [b, c],
        [
            _equation("a", "x + b / 10 + c / 10"),
            _equation("b", "a / 5 + c / 10"),
            _equation("c", "a / 10 + b / 5"),
        ],
    )

    item = solve_mechanism_equations(problem)[0]
    assert item.variables == ["a", "b", "c"]
    assert len(item.formulas) == 3
    a_values, b_values, c_values = item.function({"x": np.array([1.0, 2.0])})
    assert np.allclose(a_values, [1.0, 2.0] + b_values / 10 + c_values / 10)
    assert np.allclose(b_values, a_values / 5 + c_values / 10)
    assert np.allclose(c_values, a_values / 10 + b_values / 5)


def test_three_variable_transcendental_system_is_solved_numerically():
    a, b, c, x = map(_variable, ("a", "b", "c", "x"))
    problem = Problem(
        "three numerical",
        "three numerical",
        "x",
        a,
        [x],
        [b, c],
        [
            _equation("a", "x + cos(b) / 10 + c / 10"),
            _equation("b", "sin(a) / 5 + c / 10"),
            _equation("c", "cos(a + b) / 10"),
        ],
    )

    item = solve_mechanism_equations(problem)[0]
    assert item.variables == ["a", "b", "c"]
    assert item.formulas == []
    x_values = np.array([0.1, 0.2])
    a_values, b_values, c_values = item.function({"x": x_values})
    assert np.allclose(
        a_values,
        x_values + np.cos(b_values) / 10 + c_values / 10,
        atol=1e-8,
    )
    assert np.allclose(b_values, np.sin(a_values) / 5 + c_values / 10, atol=1e-8)
    assert np.allclose(c_values, np.cos(a_values + b_values) / 10, atol=1e-8)


def test_underdetermined_block_is_rejected():
    y, x, b = map(_variable, ("y", "x", "b"))
    problem = Problem(
        "underdetermined", "underdetermined", "x", y, [x], [b],
        [_equation("y", "x + b")],
    )
    with pytest.raises(ValueError, match="underdetermined"):
        solve_mechanism_equations(problem)


def test_solution_items_have_disjoint_variables_and_parseable_formulas():
    solution = solve_mechanism_equations(_implicit_problem())
    flattened = [name for item in solution for name in item.variables]
    assert flattened == ["y", "b"]
    assert len(flattened) == len(set(flattened))
    assert all(
        isinstance(nd.parse(formula), nd.Symbol)
        for item in solution for formula in item.formulas
    )
