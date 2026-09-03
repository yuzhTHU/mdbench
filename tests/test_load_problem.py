import pytest
import nd2py as nd

from src.core import UNIT
from src.features.io import load_problem
from src.utils import parse_unit


def test_utils_package_exports_functions():
    from src.features import load_problem as exported_load_problem
    from src.utils import tag2ansi

    assert callable(exported_load_problem)
    assert callable(tag2ansi)


def test_unit_string_format():
    assert str(UNIT({})) == "1 (dimensionless)"
    assert str(UNIT({"kg": 1, "m": 1, "s": -2})) == "kg m s^-2"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", {}),
        ("1", {}),
        ("dimensionless", {}),
        ("kg m s^-2", {"kg": 1, "m": 1, "s": -2}),
        ("kg*m/s^2", {"kg": 1, "m": 1, "s": -2}),
        ("m**3 / kg / s**2", {"m": 3, "kg": -1, "s": -2}),
        ("m³ kg⁻¹ s⁻²", {"m": 3, "kg": -1, "s": -2}),
    ],
)
def test_parse_unit(text, expected):
    assert parse_unit(text) == expected


@pytest.mark.parametrize("text", ["metre", "m//s", "*m", "m**kg"])
def test_parse_unit_rejects_invalid_notation(text):
    with pytest.raises(ValueError, match="Invalid unit"):
        parse_unit(text)


def test_load_problem_builds_unit_objects():
    problem = load_problem("demo_problem.yaml")
    assert all(isinstance(variable.unit, UNIT) for variable in problem.all_variables)
    assert all(isinstance(variable.unit, UNIT) for variable in problem.auxiliary_input_variables)
    assert all(isinstance(constant.unit, UNIT) for constant in problem.constants)
    assert all(isinstance(item.variable, str) for item in problem.mechanism)
    assert all(isinstance(item.formula, str) for item in problem.mechanism)
    assert problem.target_variable.name == "T"
    assert {variable.name for variable in problem.input_variables} == {"a", "M"}
    assert {variable.name for variable in problem.intermediate_variables} == {"r", "F", "acc", "v"}
    assert {variable.name for variable in problem.auxiliary_input_variables} == {"m"}
    assert {constant.name: constant.value for constant in problem.constants} == {
        "π": pytest.approx(3.141592653589793),
        "G": pytest.approx(6.67430e-11),
    }

    nodes = list(nd.parse(problem.phenomenological_formula).iter_preorder())
    assert any(isinstance(node, nd.Number) and node.value == 4 for node in nodes)
    assert {node.name for node in nodes if isinstance(node, nd.Variable)} >= {"π", "G"}


def test_variable_roles_come_from_sections_not_sampling(tmp_path):
    source = open("demo_problem.yaml", encoding="utf-8").read()
    source = source.replace(
        "      description: 行星圆轨道半径\n      unit: m\n",
        "      description: 行星圆轨道半径\n      unit: m\n"
        "      sampling: {min: 1, max: 3, ood_boundary: 2, distribution: uniform}\n",
        1,
    ).replace(
        "      sampling: {min: 1.0e20, max: 1.0e28, ood_boundary: 1.0e24, distribution: log_uniform}\n",
        "",
        1,
    )
    path = tmp_path / "explicit-roles.yaml"
    path.write_text(source, encoding="utf-8")
    problem = load_problem(str(path))
    assert "r" in {variable.name for variable in problem.intermediate_variables}
    assert "m" in {variable.name for variable in problem.auxiliary_input_variables}


def test_load_problem_rejects_legacy_formula_key(tmp_path):
    source = open("demo_problem.yaml", encoding="utf-8").read()
    path = tmp_path / "legacy.yaml"
    path.write_text(source.replace("phenomenological_formula:", "phenomical_formula:"), encoding="utf-8")
    with pytest.raises(ValueError, match="phenomenological_formula"):
        load_problem(str(path))


def test_load_problem_rejects_legacy_unified_variable_section(tmp_path):
    source = open("demo_problem.yaml", encoding="utf-8").read()
    source = source.replace(
        "variable_description:", "phenomenological_variable_description:", 1
    )
    path = tmp_path / "legacy-auxiliary.yaml"
    path.write_text(source, encoding="utf-8")
    with pytest.raises(ValueError, match="variable_description"):
        load_problem(str(path))


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            "    name: T\n",
            "    name: T\n    typo: unexpected\n",
            "Invalid variable_description item",
        ),
        (
            "    description: Orbital period of the planet around the star\n",
            "",
            "Invalid variable_description item",
        ),
        ("- name: π\n", "- name: π\n  typo: unexpected\n", "Invalid constants item"),
        ("  value: 3.141592653589793\n", "", "Invalid constants item"),
    ],
)
def test_dataclass_constructors_validate_spec_fields(
    tmp_path, old, new, message
):
    source = open("demo_problem.yaml", encoding="utf-8").read()
    path = tmp_path / "invalid-spec.yaml"
    path.write_text(source.replace(old, new, 1), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_problem(str(path))


def test_load_problem_allows_forward_reference_to_later_mechanism_lhs(tmp_path):
    path = tmp_path / "implicit.yaml"
    path.write_text("""
problem_name: implicit
problem_description: implicit equations
phenomenological_formula: a = 2 * x
variable_description:
  target: {name: a, description: target, unit: 1}
  inputs:
    - name: x
      description: input
      unit: 1
      sampling: {min: 1, max: 3, ood_boundary: 2, distribution: uniform}
  intermediates:
    - {name: b, description: internal variable}
  auxiliary_inputs: []
mechanism:
  - formula: a = x + b
    formula_description: first constraint
  - formula: b = a / 2
    formula_description: second constraint
""", encoding="utf-8")
    problem = load_problem(str(path))
    assert [item.equation for item in problem.mechanism] == ["a = x + b", "b = a / 2"]
    assert next(variable for variable in problem.intermediate_variables if variable.name == "b").unit is None


def test_mechanism_item_rejects_legacy_variable_metadata(tmp_path):
    source = open("demo_problem.yaml", encoding="utf-8").read()
    source = source.replace(
        "  formula_description: The orbital radius equals the semi-major axis for a circular orbit",
        "  formula_description: The orbital radius equals the semi-major axis for a circular orbit\n  unit: m",
        1,
    )
    path = tmp_path / "legacy-mechanism-metadata.yaml"
    path.write_text(source, encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected keys: unit"):
        load_problem(str(path))


def test_mechanism_item_rejects_zero_left_side(tmp_path):
    source = open("demo_problem.yaml", encoding="utf-8").read()
    source = source.replace("formula: r = a", "formula: 0 = r - a", 1)
    path = tmp_path / "zero-left-side.yaml"
    path.write_text(source, encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid mechanism variable name: '0'"):
        load_problem(str(path))
