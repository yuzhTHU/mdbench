import pytest
import sympy as sp
from src.validate_problem import (ValidationError, task_from_dict, validate_task, solve_model,
                                  parse_expression, check_units, expand_expression, load_task)
from src.core import VariableSpec

def test_bundled_demo_probes_expand_to_its_inputs(demo):
    validate_task(demo)
    inputs = {variable.name for variable in demo.by_role('input')}
    for probe in demo.mechanism_probes:
        expression = expand_expression(probe.answer, demo, lhs=probe.probe)
        assert {str(symbol) for symbol in expression.free_symbols} <= inputs


def test_sampling_rejects_nonfinite_limits(simple_raw):
    task = task_from_dict(simple_raw)
    task.variables[1].sampling['min'] = float('nan')
    with pytest.raises(ValidationError, match='Sampling min must be finite numeric'):
        validate_task(task)

def test_declared_units_of_a_product_must_match_the_target(simple_raw):
    task = task_from_dict(simple_raw)
    task.variables[0].unit = 's'  # h*x has m^2, with no numeric coefficient to absorb units.
    with pytest.raises(ValidationError, match='dimensions'):
        check_units(task)

def test_model_coupled_equations_and_alternative_names():
    sources = [VariableSpec('x', 'Input', '1', 'input')]
    result = solve_model(['u+w=3*x', 'u-w=x', 'y=u*w'], sources, required=['y'])
    x = sp.Symbol('x', real=True)
    assert result['y'] == 2*x**2
    for formulas in (['y*y=x'], ['y=u*x'], ['y=x', 'y=2*x'], ['x=2', 'y=x']):
        with pytest.raises(ValidationError): solve_model(formulas, sources, required=['y'])

def test_no_untrusted_code_execution():
    for text in ('__import__("os").system("touch /tmp/unsafe")', 'x.__class__', 'x[0]',
                 '[x for x in range(4)]', 'float("nan")', 'sin(x,y)', '2**100000', 'True'):
        with pytest.raises(ValidationError): parse_expression(text)


def test_proposal_scientific_notation_and_duplicate_yaml(simple_raw, tmp_path):
    import yaml
    path = tmp_path / (simple_raw['task_name'] + '.yaml')
    text = yaml.safe_dump(simple_raw).replace('min: 1.0', 'min: 1.0e0')
    path.write_text(text)
    assert load_task(path).variables[1].sampling['min'] == 1.0
    path.write_text(text + '\ntask_name: Other - Original\n')
    with pytest.raises(ValidationError, match='Duplicate YAML'): load_task(path)
