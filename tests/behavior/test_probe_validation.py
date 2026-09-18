"""Probe contracts: unobservable, relevant, consistent and input-only after expansion."""
import pytest
import sympy as sp

from src.validate_problem import expand_expression, load_task

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

# Auxiliary a changes h but cancels from s and the observable y. The probe s
# remains input-only. This exercises the distinction between a warning and error.
AUXILIARY_YAML = """\
task_name: Auxiliary Response - Variant 1
task_description: Relate input length to output area.
mutation: Introduce an unobserved control in a hidden response.
mechanism_model:
  - {formula: 'h = x+a', role: constitutive/component relations, description: Hidden response.}
  - {formula: 's = h-a', role: conservation/balance relations, description: Balanced state.}
  - {formula: 'y = s*x', role: conservation/balance relations, description: Output.}
phenomenal_model: y = x^2
variables:
  - {name: y, description: Output area., unit: 'm^2', role: target}
  - name: x
    description: Input length.
    unit: m
    role: input
    sampling: {min: 1, max: 8, ood_boundary: 4, distribution: uniform}
  - name: a
    description: Auxiliary control.
    unit: m
    role: auxiliary
    sampling: {min: 1, max: 3, ood_boundary: 2, distribution: uniform}
  - {name: h, description: Hidden length., unit: m, role: internal}
  - {name: s, description: Balanced length., unit: m, role: internal}
mechanism_probes:
  - {probe: s, description: Balanced length., answer: s}
"""


@pytest.mark.parametrize('answer', ['h', '2*x+3', 'h = 2*x+3'])
def test_probe_answer_can_be_an_internal_name_expression_or_equation(tmp_path, answer):
    path = tmp_path / 'Response - Variant 1.yaml'
    path.write_text(TASK_YAML.replace('answer: h}', f"answer: '{answer}'}}"))

    task = load_task(path)
    expanded = expand_expression(task.mechanism_probes[0].answer, task, lhs='h')

    x = sp.Symbol('x', positive=True)
    assert expanded == 2*x + 3
    assert expanded.free_symbols == {x}


@pytest.mark.parametrize('probe', ['x', 'y'])
def test_observable_input_and_target_variables_cannot_be_mechanism_probes(tmp_path, probe):
    path = tmp_path / 'Response - Variant 1.yaml'
    path.write_text(TASK_YAML.replace('probe: h,', f'probe: {probe},'))

    with pytest.raises(ValueError, match='Probe cannot be observable'):
        load_task(path)


def test_probe_answer_must_match_the_hidden_state_implied_by_the_mechanism(tmp_path):
    path = tmp_path / 'Response - Variant 1.yaml'
    path.write_text(TASK_YAML.replace('answer: h}', "answer: '2*x'}"))

    # h=2*x omits the offset in the submitted standard mechanism h=2*x+3.
    with pytest.raises(ValueError, match='Probe answer disagrees with mechanism'):
        load_task(path)


def test_a_solvable_hidden_state_cannot_be_a_probe_if_it_does_not_derive_the_target(tmp_path):
    path = tmp_path / 'Response - Variant 1.yaml'
    path.write_text(TASK_YAML.replace('y = h*x', 'y = (2*x+3)*x'))

    # h still has a unique value, but the target equation no longer uses it.
    with pytest.raises(ValueError, match='does not participate in deriving the target'):
        load_task(path)


def test_auxiliary_variables_warn_but_all_states_are_solved_and_input_only_probes_remain_valid(tmp_path):
    path = tmp_path / 'Auxiliary Response - Variant 1.yaml'
    path.write_text(AUXILIARY_YAML)

    with pytest.warns(UserWarning, match='Auxiliary variables'):
        task = load_task(path)

    x, a = sp.symbols('x a', positive=True)
    assert task.solution == {'h': x+a, 's': x, 'y': x**2}
    assert expand_expression('s', task) == x


def test_a_probe_that_still_depends_on_auxiliary_variables_is_rejected(tmp_path):
    path = tmp_path / 'Auxiliary Response - Variant 1.yaml'
    path.write_text(AUXILIARY_YAML.replace(
        'probe: s, description: Balanced length., answer: s',
        'probe: h, description: Hidden length., answer: h',
    ))

    with pytest.warns(UserWarning, match='Auxiliary variables'):
        with pytest.raises(ValueError, match='input variables only'):
            load_task(path)


def test_constant_internal_probes_warn_without_invalidating_the_task(tmp_path):
    path = tmp_path / 'Response - Variant 1.yaml'
    path.write_text(TASK_YAML
        .replace('mechanism_model:\n', "mechanism_model:\n  - {formula: 'k = 2', role: constitutive/component relations, description: Coefficient.}\n")
        .replace('h = 2*x + 3', 'h = k*x + 3')
        .replace('variables:\n', 'variables:\n  - {name: k, description: Coefficient., unit: 1, role: internal}\n')
        .replace('probe: h, description: Hidden length., answer: h', 'probe: k, description: Coefficient., answer: k'))

    with pytest.warns(UserWarning, match='Constant probe k'):
        task = load_task(path)

    assert expand_expression('k', task) == 2


def test_an_undeclared_derived_probe_warns_that_relevance_needs_manual_review(tmp_path):
    path = tmp_path / 'Response - Variant 1.yaml'
    path.write_text(TASK_YAML.replace(
        'probe: h, description: Hidden length., answer: h',
        "probe: q, description: Twice the hidden length., answer: '2*h'",
    ))

    with pytest.warns(UserWarning, match='outside declared internal variables'):
        task = load_task(path)

    x = sp.Symbol('x', positive=True)
    assert expand_expression(task.mechanism_probes[0].answer, task, lhs='q') == 4*x + 6
