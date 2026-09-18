"""Executable algebra contracts: explicit solutions, arbitrary order, no guessing."""
import pytest
import sympy as sp
import yaml

from src.validate_problem import load_task

TASK_YAML = """\
task_name: Coupled Response - Variant 1
task_description: Relate the input to a coupled response.
mutation: Couple two successive pairs of internal states.
mechanism_model:
  - {formula: 'a + b - 3*x = 0', role: conservation/balance relations, description: First balance.}
  - {formula: 'a - b = x', role: constitutive/component relations, description: First response.}
  - {formula: 'c + d = 5*b', role: conservation/balance relations, description: Second balance.}
  - {formula: 'c - d = b', role: constitutive/component relations, description: Second response.}
  - {formula: 'y = c + d', role: conservation/balance relations, description: Output.}
phenomenal_model: y = 5*x
variables:
  - {name: y, description: Output., unit: m, role: target}
  - name: x
    description: Input.
    unit: m
    role: input
    sampling: {min: 1, max: 8, ood_boundary: 4, distribution: uniform}
  - {name: a, description: First state., unit: m, role: internal}
  - {name: b, description: Second state., unit: m, role: internal}
  - {name: c, description: Third state., unit: m, role: internal}
  - {name: d, description: Fourth state., unit: m, role: internal}
mechanism_probes:
  - {probe: b, description: Second state., answer: b}
  - {probe: d, description: Fourth state., answer: d}
"""


def test_coupled_mechanism_solves_every_internal_and_target_in_terms_of_inputs(tmp_path):
    path = tmp_path / 'Coupled Response - Variant 1.yaml'
    path.write_text(TASK_YAML)

    task = load_task(path)

    x = sp.Symbol('x', positive=True)  # All configured samples are positive.
    assert task.solution == {'a': 2*x, 'b': x, 'c': 3*x, 'd': 2*x, 'y': 5*x}


def test_equation_and_variable_order_do_not_change_the_mathematical_solution(tmp_path):
    path = tmp_path / 'Coupled Response - Variant 1.yaml'
    path.write_text(TASK_YAML)
    before = load_task(path)
    reordered = yaml.safe_load(TASK_YAML)
    reordered['mechanism_model'].reverse()
    reordered['variables'].reverse()
    path.write_text(yaml.safe_dump(reordered))

    after = load_task(path)

    assert after.solution == before.solution


@pytest.mark.parametrize('definition', ['k = 2', '2 = k', 'k - 2 = 0', '2*k = 4'])
def test_unlisted_numeric_constants_allow_mathematically_equivalent_equations(tmp_path, definition):
    path = tmp_path / 'Constant Response - Variant 1.yaml'
    path.write_text(f'''\
task_name: Constant Response - Variant 1
task_description: Relate length to work through an internal force.
mutation: Modify the force coefficient.
mechanism_model:
  - {{formula: '{definition}', role: constitutive/component relations, description: Coefficient.}}
  - {{formula: 'F = k*x', role: constitutive/component relations, description: Force response.}}
  - {{formula: 'y = F*x', role: conservation/balance relations, description: Work.}}
phenomenal_model: y = k*x^2
variables:
  - {{name: y, description: Work., unit: J, role: target}}
  - name: x
    description: Length.
    unit: m
    role: input
    sampling: {{min: 1, max: 8, ood_boundary: 4, distribution: uniform}}
  - {{name: F, description: Force., unit: N, role: internal}}
mechanism_probes:
  - {{probe: F, description: Force., answer: F}}
''')

    task = load_task(path)

    x = sp.Symbol('x', positive=True)
    assert task.solution == {'k': 2, 'F': 2*x, 'y': 2*x**2}
    assert [variable.name for variable in task.variables] == ['y', 'x', 'F']


def test_correct_equation_count_does_not_make_undetermined_internal_states_valid(tmp_path):
    path = tmp_path / 'Coupled Response - Variant 1.yaml'
    raw = yaml.safe_load(TASK_YAML)
    raw['variables'] = [v for v in raw['variables'] if v['name'] not in ('c', 'd')]
    raw['mechanism_model'] = [
        {'formula': formula, 'role': 'conservation/balance relations', 'description': 'Balance.'}
        for formula in ['a+b=x', 'y=a+b', '2*y=2*x']
    ]
    raw['phenomenal_model'] = 'y=x'
    raw['mechanism_probes'] = [{'probe': 'a', 'description': 'First state.', 'answer': 'a'}]
    path.write_text(yaml.safe_dump(raw))

    # y=x is unique, but a and b are not. All three equations and all states matter.
    with pytest.raises(ValueError, match='no unique explicit algebraic solution'):
        load_task(path)


def test_multiple_real_branches_cannot_be_selected_from_the_declared_answer(tmp_path):
    path = tmp_path / 'Coupled Response - Variant 1.yaml'
    raw = yaml.safe_load(TASK_YAML)
    raw['variables'] = [v for v in raw['variables'] if v['name'] in ('x', 'a', 'y')]
    next(v for v in raw['variables'] if v['name'] == 'x')['unit'] = 'm^2'
    raw['mechanism_model'] = [
        {'formula': formula, 'role': 'constitutive/component relations', 'description': 'Response.'}
        for formula in ['a^2=x', 'y=a']
    ]
    raw['phenomenal_model'] = 'y=sqrt(x)'
    raw['mechanism_probes'] = [{'probe': 'a', 'description': 'First state.', 'answer': 'a'}]
    path.write_text(yaml.safe_dump(raw))

    # Positive inputs still allow a=+sqrt(x) and a=-sqrt(x).
    with pytest.raises(ValueError, match='no unique explicit algebraic solution'):
        load_task(path)


def test_an_explicit_sqrt_in_the_mechanism_defines_a_unique_branch(tmp_path):
    path = tmp_path / 'Coupled Response - Variant 1.yaml'
    raw = yaml.safe_load(TASK_YAML)
    raw['variables'] = [v for v in raw['variables'] if v['name'] in ('x', 'a', 'y')]
    next(v for v in raw['variables'] if v['name'] == 'x')['unit'] = 'm^2'
    raw['mechanism_model'] = [
        {'formula': formula, 'role': 'constitutive/component relations', 'description': 'Response.'}
        for formula in ['a=sqrt(x)', 'y=a']
    ]
    raw['phenomenal_model'] = 'y=sqrt(x)'
    raw['mechanism_probes'] = [{'probe': 'a', 'description': 'First state.', 'answer': 'a'}]
    path.write_text(yaml.safe_dump(raw))

    task = load_task(path)

    x = sp.Symbol('x', positive=True)
    assert task.solution == {'a': sp.sqrt(x), 'y': sp.sqrt(x)}


def test_inconsistent_equations_fail_even_when_the_equation_count_is_correct(tmp_path):
    path = tmp_path / 'Coupled Response - Variant 1.yaml'
    path.write_text(TASK_YAML.replace('a - b = x', 'a + b = 3*x + 1'))

    with pytest.raises(ValueError, match='no unique explicit algebraic solution'):
        load_task(path)


@pytest.mark.parametrize(('phenomenal', 'error'), [
    ('y = 6*x', 'does not imply the declared phenomenal_model'),
    ('5*x = y', 'must put the target on the left'),
    ('y = 5*b', 'RHS must use precisely the declared input variables'),
])
def test_phenomenal_model_must_be_an_implied_target_formula_over_inputs(tmp_path, phenomenal, error):
    path = tmp_path / 'Coupled Response - Variant 1.yaml'
    path.write_text(TASK_YAML.replace('phenomenal_model: y = 5*x', f'phenomenal_model: {phenomenal}'))

    with pytest.raises(ValueError, match=error):
        load_task(path)
