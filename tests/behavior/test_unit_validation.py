"""Physical dimensions are constraints; numeric literals can carry any unit."""
import warnings

import pytest

from src.validate_problem import load_task, validate_task

TASK_YAML = """\
task_name: Response - Variant 1
task_description: Relate the input length to the output area.
mutation: Add an offset to the hidden response.
mechanism_model:
  - {formula: 'h = 2*x + 3', role: constitutive/component relations, description: Hidden response.}
  - {formula: 'y = h*x', role: conservation/balance relations, description: Output balance.}
phenomenal_model: y = (2*x + 3)*x
variables:
  - {name: y, description: Output area., unit: 'm^2', role: target}
  - name: x
    description: Input length.
    unit: m
    role: input
    sampling: {min: 1, max: 8, ood_boundary: 4, distribution: uniform}
  - {name: h, description: Hidden length., unit: m, role: internal}
mechanism_probes:
  - {probe: h, description: Hidden length., answer: h}
"""


def test_numeric_offsets_may_carry_length_units_without_emitting_warnings(tmp_path):
    path = tmp_path / 'Response - Variant 1.yaml'
    # In h=3*x+3, the two occurrences of 3 may have different units:
    # a dimensionless multiplier and a length-valued offset.
    path.write_text(TASK_YAML.replace('2*x', '3*x'))

    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter('always')
        task = load_task(path, validate=False)
        report = validate_task(task, path=path)

    assert report['ok'] is True
    assert report['warnings'] == []
    assert emitted == []


def test_an_omitted_internal_unit_is_inferred_from_the_full_mechanism(tmp_path):
    path = tmp_path / 'Response - Variant 1.yaml'
    path.write_text(TASK_YAML.replace('Hidden length., unit: m,', 'Hidden length.,'))

    task = load_task(path, validate=False)
    report = validate_task(task, path=path)

    assert report['ok'] is True
    assert report['inferred_units'] == {'h': 'm^1'}


def test_variable_addition_cannot_mix_length_and_time_units(tmp_path):
    path = tmp_path / 'Response - Variant 1.yaml'
    path.write_text(TASK_YAML
        .replace("'h = 2*x + 3'", "'t = 1'", 1)
        .replace('mechanism_model:\n', "mechanism_model:\n  - {formula: 'h = x+t', role: constitutive/component relations, description: Hidden response.}\n")
        .replace('phenomenal_model: y = (2*x + 3)*x', 'phenomenal_model: y = (x+1)*x')
        .replace('variables:\n', 'variables:\n  - {name: t, description: Internal time., unit: s, role: internal}\n'))

    # Algebra alone agrees with the declared answer. x+t is nevertheless invalid.
    with pytest.raises(ValueError, match='Inconsistent physical dimensions'):
        load_task(path)


def test_algebraic_cancellation_does_not_hide_a_dimensionally_invalid_sum(tmp_path):
    path = tmp_path / 'Response - Variant 1.yaml'
    path.write_text(TASK_YAML
        .replace("'h = 2*x + 3'", "'h = x+t-t'")
        .replace('mechanism_model:\n', "mechanism_model:\n  - {formula: 't = 1', role: constitutive/component relations, description: Internal time.}\n")
        .replace('phenomenal_model: y = (2*x + 3)*x', 'phenomenal_model: y = x^2')
        .replace('variables:\n', 'variables:\n  - {name: t, description: Internal time., unit: s, role: internal}\n'))

    # x+t-t simplifies to x; validation must still inspect the original x+t sum.
    with pytest.raises(ValueError, match='Inconsistent physical dimensions'):
        load_task(path)


@pytest.mark.parametrize(('argument', 'valid'), [('x', False), ('x/2.0', True)])
def test_exp_requires_a_dimensionless_argument_but_a_numeric_scale_can_supply_it(tmp_path, argument, valid):
    path = tmp_path / 'Response - Variant 1.yaml'
    path.write_text(TASK_YAML
        .replace('h = 2*x + 3', f'h = exp({argument})')
        .replace('y = (2*x + 3)*x', f'y = exp({argument})*x')
        .replace("unit: 'm^2', role: target", 'unit: m, role: target')
        .replace('Hidden length., unit: m,', 'Hidden response., unit: 1,'))

    if valid:
        # 2.0 can carry m, so x/2.0 is dimensionless; no literal warning is allowed.
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter('always')
            load_task(path)
        assert emitted == []
    else:
        with pytest.raises(ValueError, match='Inconsistent physical dimensions'):
            load_task(path)
