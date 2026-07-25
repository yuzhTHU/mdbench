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
