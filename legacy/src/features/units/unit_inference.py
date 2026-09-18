"""Context-aware physical-unit inference for nd2py expressions."""
from __future__ import annotations

from typing import Dict, List

import nd2py as nd
import sympy as sp
from nd2py.core.base_visitor import Visitor

from ...core import UNIT, VariableSpec


UnitDict = dict[str, int | float]


def _clean(unit: UnitDict) -> UnitDict:
    return {name: exponent for name, exponent in unit.items() if exponent != 0}


def _combine(left: UnitDict, right: UnitDict, sign: int = 1) -> UnitDict:
    result = {}
    for name in set(left) | set(right):
        exponent = left.get(name, 0) + sign * right.get(name, 0)
        if exponent != 0:
            result[name] = exponent
    return result


def _scale(unit: UnitDict, factor: int | float) -> UnitDict:
    return _clean({name: exponent * factor for name, exponent in unit.items()})


class _ContextualUnitInference:
    """Solve unit constraints by repeated forward and backward tree passes.

    Numeric literals start with an unknown unit. Operator constraints propagate
    known units upward and downward until a fixed point is reached. Only truly
    unconstrained literals then fall back to dimensionless.
    """

    _PRESERVING_UNARY = {"Neg", "Abs", "Identity"}
    _SCALED_UNARY = {
        "Inv": -1,
        "Pow2": 2,
        "Pow3": 3,
        "Sqrt": 0.5,
        "SqrtAbs": 0.5,
    }
    _SAME_UNIT_BINARY = {"Add", "Sub"}

    def __init__(
        self,
        root: nd.Symbol,
        units: Dict[str, UNIT | None],
        *,
        warning_aware: bool = False,
    ):
        self.root = root
        self.nodes = list(root.iter_preorder())
        self.inferred: dict[int, UnitDict] = {}
        self.errors: list[str] = []
        self._error_set: set[str] = set()
        self._warning_aware = warning_aware

        for node in self.nodes:
            kind = type(node).__name__
            if isinstance(node, nd.Variable):
                if node.name not in units:
                    raise ValueError(
                        f"Unit inference failed for node {node}: Variable "
                        f"{node.name} not found in provided units."
                    )
                if units[node.name] is None:
                    raise ValueError(
                        f"Unit inference failed for node {node}: Variable "
                        f"{node.name} has no declared or inferred unit."
                    )
                self.inferred[id(node)] = _clean(dict(units[node.name].unit_dict))
            elif kind == "Empty":
                raise ValueError(
                    f"Unit inference failed for node {node}: Empty node encountered."
                )

    def _unit(self, node) -> UnitDict | None:
        return self.inferred.get(id(node))

    def _error(self, message: str) -> None:
        if message not in self._error_set:
            self._error_set.add(message)
            self.errors.append(message)

    def _set(self, node, unit: UnitDict, conflict: str) -> bool:
        unit = _clean(unit)
        current = self._unit(node)
        if current is None:
            self.inferred[id(node)] = unit
            return True
        if current != unit:
            self._error(conflict.format(actual=UNIT(current), expected=UNIT(unit)))
        return False

    def _same(self, node, related) -> bool:
        known = [(item, self._unit(item)) for item in related]
        known = [(item, unit) for item, unit in known if unit is not None]
        if not known:
            return False
        reference = known[0][1]
        changed = False
        message = (
            f"Unit inference failed for node {node}: Expected operands with the "
            "same units, but got {actual} and {expected}."
        )
        for item in related:
            changed |= self._set(item, reference, message)
        return changed

    def _dimensionless(self, node, subject: str) -> bool:
        return self._set(
            node,
            {},
            f"Unit inference failed for node {node}: {subject}, but got "
            + "{actual}.",
        )

    def _visit_constraint(self, node) -> bool:
        kind = type(node).__name__
        operands = list(node.operands)
        changed = False

        if isinstance(node, (nd.Variable, nd.Number)):
            return False

        if kind in self._SAME_UNIT_BINARY:
            return self._same(node, [node, *operands])

        if kind == "Mul":
            left, right = operands
            parent_unit, left_unit, right_unit = map(self._unit, (node, left, right))
            message = (
                f"Unit inference failed for node {node}: Multiplicative unit "
                "constraints conflict (got {actual}, expected {expected})."
            )
            if left_unit is not None and right_unit is not None:
                changed |= self._set(node, _combine(left_unit, right_unit), message)
            if parent_unit is not None and left_unit is not None:
                changed |= self._set(right, _combine(parent_unit, left_unit, -1), message)
            if parent_unit is not None and right_unit is not None:
                changed |= self._set(left, _combine(parent_unit, right_unit, -1), message)
            return changed

        if kind == "Div":
            numerator, denominator = operands
            parent_unit, numerator_unit, denominator_unit = map(
                self._unit, (node, numerator, denominator)
            )
            message = (
                f"Unit inference failed for node {node}: Divisive unit constraints "
                "conflict (got {actual}, expected {expected})."
            )
            if numerator_unit is not None and denominator_unit is not None:
                changed |= self._set(
                    node, _combine(numerator_unit, denominator_unit, -1), message
                )
            if parent_unit is not None and denominator_unit is not None:
                changed |= self._set(
                    numerator, _combine(parent_unit, denominator_unit), message
                )
            if parent_unit is not None and numerator_unit is not None:
                changed |= self._set(
                    denominator, _combine(numerator_unit, parent_unit, -1), message
                )
            return changed

        if kind in self._PRESERVING_UNARY:
            return self._same(node, [node, operands[0]])

        if kind in self._SCALED_UNARY:
            child = operands[0]
            factor = self._SCALED_UNARY[kind]
            parent_unit, child_unit = self._unit(node), self._unit(child)
            message = (
                f"Unit inference failed for node {node}: Unit constraints conflict "
                "(got {actual}, expected {expected})."
            )
            if child_unit is not None:
                changed |= self._set(node, _scale(child_unit, factor), message)
            if parent_unit is not None and factor != 0:
                changed |= self._set(child, _scale(parent_unit, 1 / factor), message)
            return changed

        if kind == "Pow":
            base, exponent = operands
            changed |= self._dimensionless(exponent, "Exponent must be dimensionless")
            if isinstance(exponent, nd.Number):
                factor = exponent.value
                parent_unit, base_unit = self._unit(node), self._unit(base)
                message = (
                    f"Unit inference failed for node {node}: Power unit constraints "
                    "conflict (got {actual}, expected {expected})."
                )
                if base_unit is not None:
                    changed |= self._set(node, _scale(base_unit, factor), message)
                if parent_unit is not None and factor != 0:
                    changed |= self._set(base, _scale(parent_unit, 1 / factor), message)
            else:
                changed |= self._dimensionless(
                    base, "A non-numeric exponent requires a dimensionless base"
                )
                changed |= self._dimensionless(
                    node, "A variable power's result must be dimensionless"
                )
            return changed

        if len(operands) == 1:
            # Remaining unary functions (exp, log, trig, etc.) require and
            # return dimensionless quantities.
            changed |= self._dimensionless(
                operands[0], "Function argument must be dimensionless"
            )
            changed |= self._dimensionless(
                node, "Function result must be dimensionless"
            )
            return changed

        if len(operands) == 2:
            # Preserve the old visitor's convention for other binary functions.
            return self._same(node, [node, *operands])

        self._error(
            f"Please implement contextual unit constraints for {type(node).__name__}."
        )
        return False

    def _linear_constraints(self, symbols, expected_unit, base):
        """Build scalar exponent constraints for one SI base dimension."""
        equations = []
        for node in self.nodes:
            symbol = symbols[id(node)]
            kind = type(node).__name__
            operands = list(node.operands)
            if isinstance(node, nd.Variable):
                equations.append(symbol - self._unit(node).get(base, 0))
            elif isinstance(node, nd.Number):
                # A Number is deliberately unconstrained unless its syntactic
                # role (for example, a power exponent) constrains it below.
                continue
            elif kind in self._SAME_UNIT_BINARY:
                equations.extend(symbol - symbols[id(item)] for item in operands)
            elif kind == "Mul":
                equations.append(
                    symbol - symbols[id(operands[0])] - symbols[id(operands[1])]
                )
            elif kind == "Div":
                equations.append(
                    symbol - symbols[id(operands[0])] + symbols[id(operands[1])]
                )
            elif kind in self._PRESERVING_UNARY:
                equations.append(symbol - symbols[id(operands[0])])
            elif kind in self._SCALED_UNARY:
                equations.append(
                    symbol
                    - self._SCALED_UNARY[kind] * symbols[id(operands[0])]
                )
            elif kind == "Pow":
                exponent = operands[1]
                equations.append(symbols[id(exponent)])
                if isinstance(exponent, nd.Number):
                    equations.append(
                        symbol - exponent.value * symbols[id(operands[0])]
                    )
                else:
                    equations.extend((symbols[id(operands[0])], symbol))
            elif len(operands) == 1:
                equations.extend((symbols[id(operands[0])], symbol))
            elif len(operands) == 2:
                equations.extend(symbol - symbols[id(item)] for item in operands)
        if expected_unit is not None:
            equations.append(symbols[id(self.root)] - expected_unit.get(base, 0))
        return equations

    def _solve_linear_unknowns(self, expected_unit: UnitDict | None) -> None:
        """Resolve units uniquely implied by the complete expression context."""
        bases = set(expected_unit or {})
        for node in self.nodes:
            if isinstance(node, nd.Variable):
                bases.update(self._unit(node))
        if not bases:
            # The empty dimensional vector is still a meaningful constraint.
            bases = {"__dimensionless_probe__"}

        symbols = {
            id(node): sp.Symbol(f"unit_{index}")
            for index, node in enumerate(self.nodes)
        }
        solutions = {}
        for base in bases:
            equations = self._linear_constraints(symbols, expected_unit, base)
            if not equations:
                solutions[base] = tuple(symbols.values())
                continue
            solution_set = sp.linsolve(equations, list(symbols.values()))
            if solution_set == sp.EmptySet:
                self._error(
                    "Unit inference failed: expression unit constraints are inconsistent."
                )
                return
            solutions[base] = next(iter(solution_set))

        for index, node in enumerate(self.nodes):
            values = {base: solution[index] for base, solution in solutions.items()}
            if any(value.free_symbols for value in values.values()):
                continue
            unit = {}
            for base, value in values.items():
                if base == "__dimensionless_probe__" or value == 0:
                    continue
                numeric = float(value)
                unit[base] = int(numeric) if numeric.is_integer() else numeric
            self._set(
                node,
                unit,
                f"Unit inference failed for node {node}: Linear and local unit "
                "constraints conflict (got {actual}, expected {expected}).",
            )

    def solve(self, expected_unit: UnitDict | None = None):
        if expected_unit is not None:
            self._set(
                self.root,
                expected_unit,
                f"Unit inference failed for node {self.root}: Expression has unit "
                "{actual}, but its context requires {expected}.",
            )

        for _ in range(max(1, 2 * len(self.nodes))):
            changed = False
            for node in reversed(self.nodes):
                changed |= self._visit_constraint(node)
            if not changed:
                break

        # Local propagation may stall when several unknown literals are
        # coupled. Solve all remaining dimensional-exponent constraints over
        # the complete tree, then propagate the newly unique results.
        self._solve_linear_unknowns(expected_unit)
        for _ in range(max(1, len(self.nodes))):
            changed = False
            for node in reversed(self.nodes):
                changed |= self._visit_constraint(node)
            if not changed:
                break

        # Preserve the historical public API for context-free calls. In the
        # warning-aware validation path, unresolved literals remain wildcards:
        # they are allowed to carry any unit and cannot be warned about safely.
        if expected_unit is None and not self._warning_aware:
            for node in self.nodes:
                if isinstance(node, nd.Number) and self._unit(node) is None:
                    self.inferred[id(node)] = {}
            for _ in range(max(1, len(self.nodes))):
                changed = False
                for node in reversed(self.nodes):
                    changed |= self._visit_constraint(node)
                if not changed:
                    break

        warnings = []
        for node in self.nodes:
            if isinstance(node, nd.Number) and (unit := self._unit(node)):
                warnings.append(
                    f"Numeric literal {node.value:g} was inferred to have physical "
                    f"unit {UNIT(unit)} rather than 1 (dimensionless)."
                )
        return self._unit(self.root) or {}, self.errors, warnings


class UnitInferenceVisitor(Visitor):
    """Compatibility facade around contextual unit inference."""

    def __call__(self, node: nd.Symbol, units: Dict[str, UNIT]):
        unit, errors, _ = _ContextualUnitInference(node, units).solve()
        return unit, errors


def unit_inference(
    formula: str | nd.Symbol,
    variables: List[VariableSpec],
    *,
    expected_unit: UNIT | UnitDict | None = None,
    include_warnings: bool = False,
):
    """Infer a unit, optionally using the expression's required context unit."""
    if isinstance(formula, str):
        formula = nd.parse(formula)
    expected = (
        expected_unit.unit_dict if isinstance(expected_unit, UNIT) else expected_unit
    )
    units = {variable.name: variable.unit for variable in variables}
    engine = _ContextualUnitInference(formula, units, warning_aware=include_warnings)
    unit, errors, warnings = engine.solve(expected)
    if include_warnings:
        return unit, errors, warnings
    return unit, errors
