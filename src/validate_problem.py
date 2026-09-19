"""Load, validate and solve proposal tasks and submitted algebraic models.

Formula parsing uses an AST allowlist, never eval/sympify on untrusted text.
Algebraic ambiguity is an error: no fitting, branch guessing or numeric fallback.
"""
from __future__ import annotations
import argparse
import ast
from itertools import combinations
import json
import math
import re
import warnings
from pathlib import Path
import numpy as np
import sympy as sp
import yaml
from .core import Task, VariableSpec, MechanismItem, MechanismProbe, MECHANISM_ROLES

class ValidationError(ValueError):
    pass

NAME = re.compile(r'[A-Za-z_][A-Za-z0-9_]*\Z')
FUNCTIONS = {'sqrt': sp.sqrt, 'exp': sp.exp, 'log': sp.log, 'ln': sp.log,
             'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan, 'abs': sp.Abs,
             'Abs': sp.Abs, 'asin': sp.asin, 'acos': sp.acos, 'atan': sp.atan,
             'sinh': sp.sinh, 'cosh': sp.cosh, 'tanh': sp.tanh}


def expression_tree(text: str) -> ast.expr:
    if not isinstance(text, str) or not text.strip() or len(text) > 10000:
        raise ValidationError('Formula must be a nonempty string of at most 10000 characters.')
    try:
        tree = ast.parse(text.replace('^', '**').replace('π', 'pi').strip(), mode='eval').body
    except (SyntaxError, RecursionError) as exc:
        raise ValidationError(f'Invalid mathematical expression: {text!r}') from exc
    if sum(1 for _ in ast.walk(tree)) > 1000:
        raise ValidationError('Expression is too complex.')
    return tree


def _convert(node: ast.expr, symbols: dict[str, sp.Symbol]) -> sp.Expr:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        if not math.isfinite(node.value):
            raise ValidationError('Nonfinite numeric literal.')
        return sp.Rational(str(node.value))
    if isinstance(node, ast.Name) and NAME.fullmatch(node.id):
        return symbols.setdefault(node.id, sp.Symbol(node.id, real=True))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _convert(node.operand, symbols)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp):
        left, right = _convert(node.left, symbols), _convert(node.right, symbols)
        if isinstance(node.op, ast.Add): return left + right
        if isinstance(node.op, ast.Sub): return left - right
        if isinstance(node.op, ast.Mult): return left * right
        if isinstance(node.op, ast.Div): return left / right
        if isinstance(node.op, ast.Pow):
            if right.is_number and (not right.is_real or abs(float(right)) > 100):
                raise ValidationError('Numeric power must be real and at most 100 in magnitude.')
            return left ** right
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in FUNCTIONS and len(node.args) == 1 and not node.keywords):
        return FUNCTIONS[node.func.id](_convert(node.args[0], symbols))
    raise ValidationError(f'Unsupported mathematical syntax: {ast.dump(node)}')


def parse_expression(text: str, symbols=None) -> sp.Expr:
    symbols = {} if symbols is None else symbols
    return _convert(expression_tree(text), symbols)


def split_equation(text: str) -> tuple[str, str]:
    if not isinstance(text, str) or text.count('=') != 1:
        raise ValidationError(f'Expected one equality: {text!r}')
    left, right = text.split('=')
    if not left.strip() or not right.strip():
        raise ValidationError('Both sides of an equation must be nonempty.')
    return left.strip(), right.strip()


def parse_equation(text: str, symbols=None) -> sp.Expr:
    symbols = {} if symbols is None else symbols
    left, right = split_equation(text)
    return parse_expression(left, symbols) - parse_expression(right, symbols)


def symbols_for(variables: list[VariableSpec]) -> dict[str, sp.Symbol]:
    result = {}
    for v in variables:
        positive = v.sampling is not None and v.sampling['min'] > 0
        result[v.name] = sp.Symbol(v.name, positive=True) if positive else sp.Symbol(v.name, real=True)
    return result


def solve_model(formulas: list[str], source_variables: list[VariableSpec], *, required=()) -> dict[str, sp.Expr]:
    """Solve all non-source symbols explicitly in terms of source symbols.

    Submissions define their own internal names, without access to answer metadata.
    All equations must hold, including redundant equations and source-only constraints.
    """
    if not formulas or len(formulas) > 100:
        raise ValidationError('Supply 1 to 100 equations.')
    symbols = symbols_for(source_variables)
    sources = set(symbols.values())
    residuals = [parse_equation(f, symbols) for f in formulas]
    unknowns = sorted(set().union(*(r.free_symbols for r in residuals)) - sources, key=str)
    if not unknowns:
        raise ValidationError('Model does not define any internal or target variable.')
    solved: dict[sp.Symbol, sp.Expr] = {}
    pending = list(residuals)

    def complete_solutions(equations, variables, *, simplify=True):
        """Return complete explicit real candidates for one equation block."""
        try:
            if len(equations) == len(variables) == 1:
                variable = variables[0]
                candidates = [
                    {variable: expression}
                    for expression in sp.solve(
                        equations[0], variable, check=True, simplify=simplify)
                ]
            else:
                candidates = sp.solve(equations, variables, dict=True, check=True,
                                      simplify=simplify)
        except (NotImplementedError, ValueError):
            return []
        complete = []
        for candidate in candidates:
            if set(candidate) != set(variables):
                continue
            # SymPy can return block members in terms of one another. Expand the
            # finite dependency chain before checking that only sources remain.
            for _ in range(len(variables)):
                candidate = {v: e.subs(candidate) for v, e in candidate.items()}
            if simplify:
                candidate = {v: sp.simplify(e) for v, e in candidate.items()}
            if all(e.free_symbols <= sources and e.is_real is not False
                   for e in candidate.values()):
                if candidate not in complete:
                    complete.append(candidate)
        return complete

    # Repeatedly select the first strictly solvable smallest square block.
    while pending:
        # Keep substituted expressions structural. Simplifying products with a
        # decimal-derived rational power such as 10^(-4.7447) can be extremely
        # expensive even when the equation is already explicit.
        reduced_pending = [residual.subs(solved) for residual in pending]
        resolved = []
        for index, reduced in enumerate(reduced_pending):
            if reduced.free_symbols <= sources:
                if reduced != 0 and sp.simplify(reduced) != 0:
                    raise ValidationError(f'Inconsistent equation or constraint on inputs: {reduced} = 0')
                resolved.append(index)
        if resolved:
            resolved = set(resolved)
            pending = [residual for index, residual in enumerate(pending)
                       if index not in resolved]
            continue

        progress = False
        for size in range(1, len(pending) + 1):
            for indices in combinations(range(len(pending)), size):
                equations = [reduced_pending[index] for index in indices]
                variables = sorted(
                    set().union(*(equation.free_symbols for equation in equations)) - sources,
                    key=str)
                if len(variables) != size:
                    continue
                # N=1 retains the no-simplification fast path needed for exact
                # rational representations of decimal exponents.
                complete = complete_solutions(
                    equations, variables, simplify=size != 1)
                if len(complete) != 1:
                    continue
                solved.update(complete[0])
                selected = set(indices)
                pending = [residual for index, residual in enumerate(pending)
                           if index not in selected]
                progress = True
                break
            if progress:
                break
        if not progress:
            break
    if pending:
        remaining = [v for v in unknowns if v not in solved]
        try:
            candidates = sp.solve([r.subs(solved) for r in pending], remaining, dict=True, check=True)
        except (NotImplementedError, ValueError) as exc:
            raise ValidationError(f'Cannot solve algebraic model: {exc}') from exc
        complete = []
        for candidate in candidates:
            if set(candidate) == set(remaining):
                for _ in range(len(remaining)):
                    candidate = {v: e.subs(candidate) for v, e in candidate.items()}
                candidate = {v: sp.simplify(e) for v, e in candidate.items()}
                if (all(e.free_symbols <= sources and e.is_real is not False
                        for e in candidate.values()) and candidate not in complete):
                    complete.append(candidate)
        if len(complete) != 1:
            raise ValidationError('Model has no unique explicit algebraic solution (underdetermined, inconsistent, or multiple branches).')
        solved.update(complete[0])
    if any((value := r.subs(solved)) != 0 and sp.simplify(value) != 0
           for r in residuals):
        raise ValidationError('Solved model fails an original equation.')
    result = {str(v): e for v, e in solved.items()}
    if missing := set(required) - set(result):
        raise ValidationError(f'Model does not solve required variables: {sorted(missing)}')
    return result


def expand_expression(text: str, task: Task, *, lhs: str | None = None) -> sp.Expr:
    symbols = symbols_for(task.by_role('input', 'auxiliary'))
    symbols.update({name: sp.Symbol(name, real=True) for name in task.solution})
    if '=' in text:
        left, text = split_equation(text)
        if lhs is not None and left != lhs:
            raise ValidationError(f'Expected {lhs} on the left side.')
    expression = parse_expression(text, symbols)
    expression = sp.simplify(expression.subs({symbols[n]: e for n, e in task.solution.items()}))
    allowed = {symbols[v.name] for v in task.by_role('input', 'auxiliary')}
    if expression.free_symbols - allowed:
        raise ValidationError('Probe expression must expand to input and auxiliary variables only.')
    return expression


def _fields(value, required, optional=(), context='object'):
    if not isinstance(value, dict):
        raise ValidationError(f'{context} must be a mapping.')
    missing, extra = set(required) - value.keys(), value.keys() - set(required) - set(optional)
    if missing or extra:
        raise ValidationError(f'{context}: missing fields {sorted(missing)}, unexpected fields {sorted(extra)}')


def _text(value, label, *, empty=False):
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValidationError(f'{label} must be a string{ "" if empty else " with nonempty content"}.')


def task_from_dict(raw: dict) -> Task:
    _fields(raw, ('task_name', 'task_description', 'mutation', 'mechanism_model',
                  'phenomenal_model', 'variables', 'mechanism_probes'), context='task')
    for label in ('task_name', 'task_description', 'mutation', 'phenomenal_model'):
        _text(raw[label], label, empty=label == 'mutation')
    for label in ('variables', 'mechanism_model', 'mechanism_probes'):
        if not isinstance(raw[label], list):
            raise ValidationError(f'{label} must be a list.')
    variables = []
    for item in raw['variables']:
        _fields(item, ('name', 'description', 'role'), ('unit', 'sampling'), 'variable')
        for label in ('name', 'description', 'role'): _text(item[label], f'variable.{label}')
        if not NAME.fullmatch(item['name']) or item['name'] in FUNCTIONS:
            raise ValidationError(f'Invalid/reserved variable name: {item["name"]}')
        if item['role'] not in ('target', 'input', 'internal', 'auxiliary'):
            raise ValidationError(f'Invalid variable role: {item["role"]}')
        if item.get('unit') is not None and not isinstance(item['unit'], (str, int)):
            raise ValidationError('Unit must be an SI unit string or 1.')
        unit = str(item['unit']) if item.get('unit') is not None else None
        variables.append(VariableSpec(item['name'], item['description'], unit, item['role'], item.get('sampling')))
    mechanism = []
    for item in raw['mechanism_model']:
        _fields(item, ('formula', 'role', 'description'), context='mechanism')
        for label in ('formula', 'role', 'description'): _text(item[label], f'mechanism.{label}')
        if item['role'] not in MECHANISM_ROLES:
            raise ValidationError(f'Invalid mechanism role: {item["role"]}')
        mechanism.append(MechanismItem(item['formula'], parse_equation(item['formula']), item['role'], item['description']))
    probes = []
    for item in raw['mechanism_probes']:
        _fields(item, ('probe', 'description', 'answer'), context='probe')
        for label in ('probe', 'description', 'answer'): _text(item[label], f'probe.{label}')
        if not NAME.fullmatch(item['probe']): raise ValidationError('Invalid probe name.')
        probes.append(MechanismProbe(**item))
    return Task(raw['task_name'], raw['task_description'], raw['mutation'], mechanism,
                raw['phenomenal_model'], variables, probes)



class TaskLoader(yaml.SafeLoader):
    """Recognize proposal-style scientific notation (YAML 1.2), reject duplicate keys."""


def _unique_mapping(loader, node, deep=False):
    loader.flatten_mapping(node)
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValidationError(f'Duplicate YAML field: {key}')
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


TaskLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)
TaskLoader.add_implicit_resolver('tag:yaml.org,2002:float',
    re.compile(r'^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)[eE][+-]?[0-9]+$'),
    list('+-0123456789.'))

def load_task(path: str | Path, *, validate=True, check_filename=True) -> Task:
    path = Path(path)
    raw = json.loads(path.read_text()) if path.suffix == '.json' else yaml.load(path.read_text(), Loader=TaskLoader)
    task = task_from_dict(raw)
    if validate: validate_task(task, path=path if check_filename else None)
    return task


def validate_sampling(variable: VariableSpec):
    sampling = variable.sampling
    if variable.role not in ('input', 'auxiliary'):
        if sampling is not None: raise ValidationError('Internal/target variables cannot declare sampling.')
        return
    _fields(sampling, ('min', 'max', 'ood_boundary', 'distribution'), context=f'{variable.name}.sampling')
    for key in ('min', 'max', 'ood_boundary'):
        value = sampling[key]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValidationError(f'Sampling {key} must be finite numeric.')
    if not sampling['min'] < sampling['ood_boundary'] < sampling['max']:
        raise ValidationError('Sampling requires min < ood_boundary < max.')
    if sampling['distribution'] not in ('uniform', 'log_uniform'):
        raise ValidationError('Supported distributions: uniform, log_uniform.')
    if sampling['distribution'] == 'log_uniform' and sampling['min'] <= 0:
        raise ValidationError('log_uniform sampling requires positive bounds.')


# Dimensional constraints are built from raw syntax, before SymPy folds literals.
BASES = ('kg', 'm', 's', 'A', 'K', 'mol', 'cd')
DERIVED_UNITS = {
    'N': {'kg': 1, 'm': 1, 's': -2}, 
    'J': {'kg': 1, 'm': 2, 's': -2},
    'W': {'kg': 1, 'm': 2, 's': -3}, 
    'Pa': {'kg': 1, 'm': -1, 's': -2},
    'Hz': {'s': -1},
    'C': {'s': 1, 'A': 1}, 
    'V': {'kg': 1, 'm': 2, 's': -3, 'A': -1},
    'ohm': {'kg': 1, 'm': 2, 's': -3, 'A': -2}, 
    'rad': {}
}


def parse_unit(text: str) -> dict[str, sp.Rational]:
    if text.strip().lower() in ('1', 'dimensionless', '1 (dimensionless)', '-'):
        return {}
    result = {}
    for i, section in enumerate(text.replace('**', '^').replace('·', '*').split('/')):
        parts = section.replace('*', ' ').split()
        if not parts: raise ValidationError(f'Invalid unit: {text}')
        for part in parts:
            match = re.fullmatch(r'([A-Za-z]+)(?:\^([-+]?\d+(?:\.\d+)?))?', part)
            if not match: raise ValidationError(f'Invalid unit: {text}')
            name, power = match.groups()
            if name not in BASES and name not in DERIVED_UNITS:
                raise ValidationError(f'Unknown unit: {name}')
            power = sp.Rational(power or '1') * (-1 if i else 1)
            for base, exponent in ({name: 1} if name in BASES else DERIVED_UNITS[name]).items():
                result[base] = result.get(base, 0) + power * exponent
    return {base: value for base, value in result.items() if value}


def check_units(task: Task) -> dict[str, str]:
    equations = []
    dimensions = {v.name: sp.Dummy(v.name) for v in task.variables}

    def dimension(node):
        if isinstance(node, ast.Name):
            return dimensions.setdefault(node.id, sp.Dummy(node.id))
        if isinstance(node, ast.Constant): return sp.Dummy('literal')
        if isinstance(node, ast.UnaryOp): return dimension(node.operand)
        if isinstance(node, ast.BinOp):
            left, right = dimension(node.left), dimension(node.right)
            if isinstance(node.op, (ast.Add, ast.Sub)):
                equations.append(left - right)
                return left
            if isinstance(node.op, ast.Mult): return left + right
            if isinstance(node.op, ast.Div): return left - right
            if isinstance(node.op, ast.Pow):
                equations.append(right)
                exponent = _convert(node.right, {})
                if exponent.is_number: return exponent * left
                equations.append(left)
                return sp.S.Zero
        if isinstance(node, ast.Call):
            child = dimension(node.args[0])
            if node.func.id in ('abs', 'Abs'): return child
            if node.func.id == 'sqrt': return child / 2
            equations.append(child)
            return sp.S.Zero
        raise ValidationError('Unsupported unit syntax.')

    texts = [m.formula_str for m in task.mechanism_model] + [task.phenomenal_model]
    for probe in task.mechanism_probes:
        texts.append(probe.answer if '=' in probe.answer else f'{probe.probe} = {probe.answer}')
    for text in texts:
        left, right = split_equation(text)
        # Validate allowlist even when called independently.
        parse_equation(text)
        equations.append(dimension(expression_tree(left)) - dimension(expression_tree(right)))
    known = {v.name: parse_unit(v.unit) for v in task.variables if v.unit is not None}
    results = {}
    all_symbols = set(dimensions.values()) | set().union(*(e.free_symbols for e in equations))
    ordered = sorted(all_symbols, key=sp.default_sort_key)
    for base in BASES:
        constraints = equations + [dimensions[name] - units.get(base, 0) for name, units in known.items()]
        solution = sp.linsolve(constraints, ordered)
        if solution == sp.EmptySet:
            raise ValidationError(f'Inconsistent physical dimensions ({base}).')
        values = next(iter(solution))
        results[base] = dict(zip(ordered, values))
    inferred = {}
    for v in task.variables:
        if v.unit is not None: continue
        powers = {base: results[base][dimensions[v.name]] for base in BASES}
        if any(value.free_symbols for value in powers.values()): continue
        inferred[v.name] = ' '.join(f'{base}^{power}' for base, power in powers.items() if power) or '1'
    return inferred


def validate_task(task: Task, *, path: Path | None = None, seen: set[str] | None = None) -> dict:
    if not re.fullmatch(r'[A-Za-z0-9 -]+', task.task_name):
        raise ValidationError('task_name may contain only letters, digits, spaces and hyphens.')
    if not re.fullmatch(r'.+ - (Original|Variant [1-9]\d*(?:-[1-9]\d*)*)', task.task_name):
        raise ValidationError('task_name must end in " - Original" or " - Variant 1[-2...]".')
    if path is not None and path.stem != task.task_name:
        raise ValidationError(f'task_name must equal filename stem: {path.name}')
    if seen is not None:
        if task.task_name in seen: raise ValidationError(f'Duplicate task_name: {task.task_name}')
        seen.add(task.task_name)
    names = [v.name for v in task.variables]
    if len(set(names)) != len(names): raise ValidationError('Duplicate variable names.')
    target = task.target
    for v in task.variables:
        validate_sampling(v)
        if v.role in ('target', 'input', 'auxiliary') and v.unit is None:
            raise ValidationError(f'Observed variable {v.name} must declare a unit.')
    notices = []
    def warn(message):
        notices.append(message)
        warnings.warn(message, UserWarning, stacklevel=2)
    if task.by_role('auxiliary'): warn('Auxiliary variables are strongly discouraged by proposal.md.')
    for label, text in [('task_description', task.task_description), ('mutation', task.mutation)]:
        if len(text.split()) > 30: warn(f'{label} exceeds the recommended 30 words.')
    sources = task.by_role('input', 'auxiliary')
    symbols = symbols_for(sources)
    left, right = split_equation(task.phenomenal_model)
    if left != target.name: raise ValidationError('phenomenal_model must put the target on the left.')
    phenomenal = parse_expression(right, symbols)
    input_symbols = {symbols[v.name] for v in task.by_role('input')}
    required = [v.name for v in task.by_role('internal', 'target')]
    formulas = [m.formula_str for m in task.mechanism_model]
    # The proposal also permits unlisted numeric constants defined by equations.
    residuals = [parse_equation(formula) for formula in formulas]
    numeric_constants = set()
    for residual in residuals:
        # Constant definitions obey the same equality contract as every other
        # mechanism equation: k=2, 2=k and k-2=0 are interchangeable.
        constant_symbols = residual.free_symbols
        if len(constant_symbols) != 1:
            continue
        symbol = next(iter(constant_symbols))
        if str(symbol) in names:
            continue
        try:
            values = sp.solve(residual, symbol, check=True)
        except (NotImplementedError, ValueError):
            continue
        if len(values) == 1 and not values[0].free_symbols and values[0].is_real is not False:
            numeric_constants.add(str(symbol))
    all_used = set().union(*(set(map(str, residual.free_symbols)) for residual in residuals))
    if unknown := all_used - set(names) - numeric_constants:
        raise ValidationError(f'Undeclared nonconstant variables: {sorted(unknown)}')
    if len(formulas) != len(required) + len(numeric_constants):
        raise ValidationError('Equation count must equal internal count + one target + unlisted numeric constant definitions.')
    if unused := set(required) - all_used:
        raise ValidationError(f'Unused derived variables: {sorted(unused)}')
    task.solution = solve_model(formulas, sources, required=required)
    constants = {sp.Symbol(name, real=True): expr for name, expr in task.solution.items() if not expr.free_symbols}
    phenomenal = sp.simplify(phenomenal.subs(constants))
    if phenomenal.free_symbols != input_symbols:
        raise ValidationError('phenomenal_model RHS must use precisely the declared input variables, after expanding numeric constants.')
    if sp.simplify(task.solution[target.name] - phenomenal) != 0:
        raise ValidationError('Mechanism does not imply the declared phenomenal_model.')
    probe_names = [p.probe for p in task.mechanism_probes]
    if not probe_names: warn('Task has no mechanism probes; mechanism recovery cannot be scored.')
    if len(set(probe_names)) != len(probe_names): raise ValidationError('Duplicate mechanism probes.')
    # Structural participation is checked by removing equations involving a probe.
    # This is a relevance check, not a DAG-recovery metric.
    for probe in task.mechanism_probes:
        if probe.probe in [v.name for v in task.observed]: raise ValidationError('Probe cannot be observable.')
        answer = expand_expression(probe.answer, task, lhs=probe.probe)
        if probe.probe in task.solution:
            if sp.simplify(answer - task.solution[probe.probe]) != 0:
                raise ValidationError(f'Probe answer disagrees with mechanism: {probe.probe}')
            if not answer.free_symbols: warn(f'Constant probe {probe.probe} is discouraged.')
            reduced = [f for f in formulas if probe.probe not in set(map(str, parse_equation(f).free_symbols))]
            try:
                alternative = solve_model(reduced, sources, required=[target.name])
            except ValidationError:
                alternative = None
            if alternative is not None and sp.simplify(alternative[target.name] - phenomenal) == 0:
                raise ValidationError(f'Probe {probe.probe} does not participate in deriving the target.')
        else:
            warn(f'Probe {probe.probe} is outside declared internal variables; review its relevance manually.')
    inferred = check_units(task)
    return {'task_name': task.task_name, 'ok': True, 'warnings': notices,
            'solution': {n: str(e) for n, e in task.solution.items()}, 'inferred_units': inferred}


def discover_tasks(paths):
    result = []
    for value in paths:
        path = Path(value)
        if path.is_dir():
            candidates = [*path.rglob('*.yaml'), *path.rglob('*.yml')]
            result.extend(candidate for candidate in candidates
                          if 'legacy' not in candidate.relative_to(path).parts[:-1])
        elif path.is_file(): result.append(path)
        else: raise ValidationError(f'Task path does not exist: {path}')
    if not result: raise ValidationError('No task YAML files found.')
    return sorted(set(result))


def get_parser(parser=None):
    parser = parser or argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', nargs='+', default=['tasks'])
    return parser


def main(args):
    reports, seen, families = [], set(), {}
    for path in discover_tasks(args.tasks):
        try:
            task = load_task(path, validate=False)
            report = validate_task(task, path=path, seen=seen)
            family = task.task_name.rsplit(' - ', 1)[0]
            if family in families and families[family] != task.task_description:
                raise ValidationError('Task family descriptions must be identical across variants.')
            families[family] = task.task_description
            reports.append(report)
        except (ValueError, TypeError, yaml.YAMLError) as exc:
            reports.append({'path': str(path), 'ok': False, 'error': str(exc)})
    print(json.dumps(reports, indent=2))
    return int(any(not r['ok'] for r in reports))

if __name__ == '__main__':
    raise SystemExit(main(get_parser().parse_args()))
