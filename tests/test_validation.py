import pytest
import sympy as sp
from pathlib import Path
from src import validate_problem as validator
from src.scoring import symbolic_equivalent
from src.validate_problem import (ValidationError, task_from_dict, validate_task, solve_model,
                                  parse_expression, check_units, expand_expression, load_task,
                                  discover_tasks)
from src.core import VariableSpec

def test_bundled_demo_probes_expand_to_its_inputs(demo):
    validate_task(demo)
    inputs = {variable.name for variable in demo.by_role('input')}
    for probe in demo.mechanism_probes:
        expression = expand_expression(probe.answer, demo, lhs=probe.probe)
        assert {str(symbol) for symbol in expression.free_symbols} <= inputs


def test_task_discovery_is_recursive(tmp_path):
    root_task = tmp_path / 'root.yaml'
    nested_task = tmp_path / 'domain' / 'family' / 'nested.yml'
    archived_task = tmp_path / 'domain' / 'ignored.yaml.archived'
    nested_task.parent.mkdir(parents=True)
    for path in (root_task, nested_task, archived_task):
        path.write_text('task_name: placeholder\n')

    assert discover_tasks([tmp_path]) == [nested_task, root_task]


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


def test_model_coupled_equations_and_alternative_names(monkeypatch):
    sources = [VariableSpec('x', 'Input', '1', 'input')]
    calls = []
    original_solve = validator._solve_block
    def recording_solve(equations, *args, **kwargs):
        calls.append(len(equations) if isinstance(equations, (list, tuple)) else 1)
        return original_solve(equations, *args, **kwargs)
    monkeypatch.setattr(validator, '_solve_block', recording_solve)
    result = solve_model(['u+w=3*x', 'u-w=x', 'y=u*w'], sources, required=['y'])
    x = sp.Symbol('x', real=True)
    assert result['y'] == 2*x**2
    assert calls[:2] == [2, 1]  # Solve {u,w} first, then the downstream y equation.
    for formulas in (['y*y=x'], ['y=u*x'], ['y=x', 'y=2*x'], ['x=2', 'y=x']):
        with pytest.raises(ValidationError): solve_model(formulas, sources, required=['y'])


def test_explicit_decimal_power_constants_are_solved_by_substitution():
    sources = [VariableSpec('A0', 'Acid', '1', 'input'),
               VariableSpec('B0', 'Base', '1', 'input')]
    result = solve_model([
        'Ka = 10^(-4.7447274948967)',
        'Hplus = Ka * A0 / B0',
        'pH = -log(Hplus)/log(10)',
    ], sources, required=['pH'])
    assert {str(symbol) for symbol in result['pH'].free_symbols} == {'A0', 'B0'}
    assert abs(float(result['Ka']) - 1.8e-5) < 1e-16


def test_symbolic_equivalence_quickly_rejects_decimal_approximation():
    x = sp.Symbol('x', positive=True)
    predicted = sp.log(x) / sp.log(10) + sp.Rational('4.7447274948966935')
    truth = -sp.log(sp.Rational(9, 500000) / x) / sp.log(10)
    assert not symbolic_equivalent(predicted, truth)
    assert symbolic_equivalent((x + 1) ** 2, x**2 + 2*x + 1)

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


def test_resolved_substitution_preserves_dependencies_and_bound_variables(monkeypatch):
    x, y, z = sp.symbols('x y z', real=True)
    dependent = {x: y + 1, y: sp.Integer(2)}
    assert validator._substitute(x, dependent) == x.subs(dependent) == 3
    bound = sp.Integral(x, (x, 0, 1)) + x
    assert validator._substitute(bound, {x: y}) == bound.subs({x: y})
    expression, mapping = x*y + x, {x: z + 1, y: z**2}
    expected = expression.subs(mapping)

    def unexpected_subs(*args, **kwargs):
        pytest.fail('Resolved independent symbols should be replaced in one pass.')

    monkeypatch.setattr(sp.Basic, 'subs', unexpected_subs)
    assert validator._substitute(expression, mapping) == expected


@pytest.mark.parametrize('use_flint', [False, True])
def test_exact_zero_proofs_and_fallback_do_not_accept_nearby_nonidentities(monkeypatch, use_flint):
    if use_flint and validator.fmpq_mpoly_ctx is None:
        pytest.skip('Optional python-flint is not installed.')
    if not use_flint:
        monkeypatch.setattr(validator, 'fmpq_mpoly_ctx', None)
    x, y = sp.symbols('x y', positive=True)
    rational = (x**2-y**2)/(x-y) - (x+y)
    function = sp.exp(x) + sp.sqrt(y)
    opaque = (function**2-1)/(function-1) - (function+1)
    assert validator._is_zero(rational)
    assert validator._is_zero(opaque)
    # Treating sin/cos as independent symbols is insufficient; simplify must
    # still handle their relationship instead of rejecting a valid identity.
    assert validator._is_zero(sp.sin(x)**2 + sp.cos(x)**2 - 1)
    assert not validator._is_zero(rational + sp.Rational(1, 10**30))
    assert not validator._is_zero(sp.sqrt(x**2 + 1) - x)


def test_linear_fast_path_keeps_large_rhs_compact_and_does_not_call_general_solve(monkeypatch):
    sources = [VariableSpec('x', 'Input', '1', 'input')]

    def unexpected_solve(*args, **kwargs):
        pytest.fail('Explicit and coupled linear equations do not need general solve.')

    monkeypatch.setattr(sp, 'solve', unexpected_solve)
    result = solve_model(['u+w=3*exp(x)', 'u-w=exp(x)',
                          'y=(u+w+1)^40'], sources, required=['y'])
    x = sp.Symbol('x', real=True)
    assert result['y'] == (3*sp.exp(x) + 1)**40


def test_nonlinear_branch_checks_remain_in_effect():
    sources = [VariableSpec('x', 'Input', '1', 'input', sampling={'min': 1})]
    assert solve_model(['y^3=x'], sources)['y'] == sp.Symbol('x', positive=True)**sp.Rational(1, 3)
    for equations in (['y^2=x'], ['y^2=-1'], ['u^2+v^2=0', 'y=u+v']):
        with pytest.raises(ValidationError):
            solve_model(equations, sources, required=['y'])


def test_underdetermined_shortcut_is_restricted_to_audited_sympy_version(monkeypatch):
    sources = [VariableSpec('x', 'Input', '1', 'input')]
    calls = []
    original = sp.solve

    def recording_solve(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(sp, 'solve', recording_solve)
    monkeypatch.setattr(sp, '__version__', 'future-version')
    with pytest.raises(ValidationError):
        solve_model(['u+v=x', 'y=u+v'], sources, required=['y'])
    assert calls  # Unreviewed versions must use the existing solver contract.


def test_complex_equation_can_constrain_two_real_unknowns():
    # Counting equations before real/imaginary splitting would reject this
    # complete solution incorrectly, even though the expression is rational.
    assert solve_model(['u+sqrt(-1)*v=0'], []) == {'u': 0, 'v': 0}


@pytest.mark.parametrize(('family', 'variant'), [
    ('Thermoelastic Heating Response', '1-2'),
    ('Drude Transport', '1-2-5'),
])
def test_previously_slow_tasks_validate_without_global_sympy_patches(family, variant):
    before = (sp.solve, sp.simplify, sp.Basic.subs, sp.Add.as_real_imag,
              sp.Mul.as_real_imag, sp.Pow.as_real_imag)
    path = Path(__file__).resolve().parents[1] / 'tasks' / 'material' / family / f'{family} - Variant {variant}.yaml'
    task = load_task(path, validate=False)
    report = validate_task(task, path=path)
    assert report['ok']
    assert sum(len(text) for text in report['solution'].values()) < 20000
    assert before == (sp.solve, sp.simplify, sp.Basic.subs, sp.Add.as_real_imag,
                      sp.Mul.as_real_imag, sp.Pow.as_real_imag)
