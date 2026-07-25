from src.core import ConstantSpec, MechanismItem, Problem, UNIT, VariableSpec
import nd2py as nd
import numpy as np

from src.validate_problem import (
    _check_sampling,
    _check_solution,
    _check_units,
    _check_variables,
    _run_check,
)


def _validate(problem):
    checks = [
        _run_check("Variable definitions", lambda: _check_variables(problem)),
        _run_check(
            "Mechanism equation solving",
            lambda: _check_solution(problem),
        ),
        _run_check("Physical units", lambda: _check_units(problem)),
        _run_check("Sampling specifications", lambda: _check_sampling(problem)),
    ]
    return {"ok": all(check["ok"] for check in checks), "checks": checks}


def test_loaded_formula_format_is_accepted():
    from src.features.io import load_problem

    result = _validate(load_problem("problems/demo_problem.yaml"))
    assert len(result["checks"]) == 4


def test_pi_and_e_are_dimensionless_constants():
    target = VariableSpec("y", "target", UNIT({}))
    constants = [
        ConstantSpec("pi", "pi", UNIT({}), np.pi),
        ConstantSpec("e", "Euler's number", UNIT({}), np.e),
    ]
    problem = Problem(
        problem_name="constants",
        problem_description="constants",
        phenomenological_formula="pi + e",
        target_variable=target,
        input_variables=[],
        intermediate_variables=[],
        mechanism=[MechanismItem("y", "pi + e", "constants")],
        constants=constants,
    )

    result = _validate(problem)
    assert result["ok"], result


def test_undeclared_math_constant_is_rejected():
    target = VariableSpec("y", "target", UNIT({}))
    problem = Problem(
        problem_name="undeclared constant",
        problem_description="undeclared constant",
        phenomenological_formula="pi",
        target_variable=target,
        input_variables=[],
        intermediate_variables=[],
        mechanism=[MechanismItem("y", "pi", "undeclared constant")],
    )

    result = _validate(problem)
    assert not result["ok"]
    assert any("pi" in check["detail"] for check in result["checks"] if not check["ok"])


def test_rejects_declared_constant_replaced_by_numeric_literal():
    target = VariableSpec("y", "target", UNIT({}))
    x = VariableSpec(
        "x", "input", UNIT({}),
        sampling={"min": 0.1, "max": 10.0, "ood_boundary": 5.0, "distribution": "uniform"},
    )
    k = ConstantSpec("k", "constant", UNIT({}), value=2.0)
    problem = Problem(
        problem_name="fixed constant",
        problem_description="fixed constant",
        phenomenological_formula="k * x",
        target_variable=target,
        input_variables=[x],
        intermediate_variables=[],
        mechanism=[MechanismItem("y", "2 * x", "substituted constant")],
        constants=[k],
    )

    result = _validate(problem)
    assert not result["ok"]
    assert "unused" in result["checks"][0]["detail"]


def test_unused_declared_constant_is_rejected():
    from src.features.io import load_problem

    problem = load_problem("problems/demo_problem.yaml")
    problem.constants.append(ConstantSpec("unused", "unused", UNIT({}), 1.0))
    result = _validate(problem)
    variable_check = next(
        check for check in result["checks"] if check["name"] == "Variable definitions"
    )
    assert not variable_check["ok"]
    assert "unused" in variable_check["detail"]


def test_intermediate_variable_may_omit_unit():
    from src.features.io import load_problem

    problem = load_problem("problems/demo_problem.yaml")
    next(variable for variable in problem.intermediate_variables if variable.name == "F").unit = None
    result = _validate(problem)
    assert result["ok"], result


def test_solution_must_derive_the_target_variable():
    target = VariableSpec("y", "target", UNIT({}))
    x = VariableSpec(
        "x",
        "input",
        UNIT({}),
        sampling={
            "min": 0.1,
            "max": 10.0,
            "ood_boundary": 5.0,
            "distribution": "uniform",
        },
    )
    intermediate = VariableSpec("a", "intermediate", UNIT({}))
    problem = Problem(
        problem_name="missing target",
        problem_description="mechanism does not derive y",
        phenomenological_formula="x",
        target_variable=target,
        input_variables=[x],
        intermediate_variables=[intermediate],
        mechanism=[MechanismItem("a", "x", "derive only a")],
    )

    check = _run_check("Mechanism equation solving", lambda: _check_solution(problem))
    assert not check["ok"]
    assert "does not derive the phenomenological equation" in check["detail"]


def test_effective_force_is_used_without_warning(caplog):
    import logging
    from src.features.io import load_problem

    problem = load_problem("problems/分层流体小球终端速度-二次阻力改版.yaml")
    with caplog.at_level(logging.WARNING, logger="src.utils.logger"):
        detail = _check_solution(problem)
    assert "Mechanism derivation tested" in detail
    assert "never used" not in caplog.text
