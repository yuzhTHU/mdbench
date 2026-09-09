"""Expected behaviour of src.features.units.unit_inference."""

import pytest

from src.core import UNIT, VariableSpec


@pytest.fixture
def variables():
    return [
        VariableSpec("length", "length", UNIT({"m": 1})),
        VariableSpec("time", "time", UNIT({"s": 1})),
        VariableSpec("area", "area", UNIT({"m": 2})),
        VariableSpec("ratio", "ratio", UNIT({})),
    ]


def infer(formula, variables):
    # Import here so that an incompatible nd2py API is reported as a test
    # failure instead of preventing pytest from collecting the test file.
    from src.features.units import unit_inference

    unit, errors = unit_inference(formula, variables)
    return unit.unit_dict if isinstance(unit, UNIT) else unit, errors


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        ("length", {"m": 1}),
        ("2", {}),
        ("length * length", {"m": 2}),
        ("length / time", {"m": 1, "s": -1}),
        ("length ** 3", {"m": 3}),
        ("sqrt(area)", {"m": 1}),
        ("-length", {"m": 1}),
        ("abs(length)", {"m": 1}),
        ("1 / length", {"m": -1}),
    ],
)
def test_infers_units(formula, expected, variables):
    unit, errors = infer(formula, variables)
    assert unit == expected
    assert errors == []


def test_addition_requires_matching_units(variables):
    unit, errors = infer("length + time", variables)
    assert errors
    assert "same units" in errors[0]


def test_dimensionless_function_rejects_dimensional_input(variables):
    unit, errors = infer("sin(length)", variables)
    assert unit == {}
    assert errors
    assert "dimensionless" in errors[0]


def test_dimensionless_function_accepts_dimensionless_input(variables):
    unit, errors = infer("sin(ratio)", variables)
    assert unit == {}
    assert errors == []


def test_variable_exponent_accepts_dimensionless_base_and_exponent(variables):
    unit, errors = infer("ratio ** (1 / ratio)", variables)
    assert unit == {}
    assert errors == []


def test_variable_exponent_rejects_dimensional_base(variables):
    unit, errors = infer("length ** ratio", variables)
    assert unit == {}
    assert errors
    assert "requires a dimensionless base" in errors[0]


def test_unknown_variable_raises_clear_error(variables):
    with pytest.raises(ValueError, match="not found"):
        infer("unknown", variables)


def test_numeric_factor_is_inferred_dimensionless_when_terms_already_match():
    from src.features.units import unit_inference

    force_variables = [
        VariableSpec("F_down1", "force", UNIT({"kg": 1, "m": 1, "s": -2})),
        VariableSpec("F_down2", "force", UNIT({"kg": 1, "m": 1, "s": -2})),
    ]
    unit, errors, warnings = unit_inference(
        "F_down1 + 0.1 * F_down2",
        force_variables,
        include_warnings=True,
    )

    assert unit == {"kg": 1, "m": 1, "s": -2}
    assert errors == []
    assert warnings == []


def test_numeric_factor_unit_is_inferred_from_expected_result():
    from src.features.units import unit_inference

    unit, errors, warnings = unit_inference(
        "0.1 * x",
        [VariableSpec("x", "length", UNIT({"m": 1}))],
        expected_unit=UNIT({"K": 1}),
        include_warnings=True,
    )

    assert unit == {"K": 1}
    assert errors == []
    assert len(warnings) == 1
    assert "m^-1 K" in warnings[0]


def test_numeric_factor_unit_is_inferred_from_dimensionless_function_argument():
    from src.features.units import unit_inference

    unit, errors, warnings = unit_inference(
        "exp(0.1 * mass)",
        [VariableSpec("mass", "mass", UNIT({"kg": 1}))],
        include_warnings=True,
    )

    assert unit == {}
    assert errors == []
    assert len(warnings) == 1
    assert "kg^-1" in warnings[0]


def test_coupled_numeric_units_remain_unreported_when_not_uniquely_identifiable():
    from src.features.units import unit_inference

    unit, errors, warnings = unit_inference(
        "3.0e15 / (12.5 * radius**2)",
        [VariableSpec("radius", "radius", UNIT({"m": 1}))],
        expected_unit=UNIT({"kg": 1, "s": -3}),
        include_warnings=True,
    )

    assert unit == {"kg": 1, "s": -3}
    assert errors == []
    assert warnings == []
