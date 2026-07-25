import nd2py as nd
from typing import List, Dict
from nd2py.core.base_visitor import yield_nothing, Visitor

from ...core import VariableSpec, UNIT

class UnitInferenceVisitor(Visitor):
    def __call__(self, node: nd.Symbol, units: Dict[str, UNIT]):
        y = super().__call__(node=node, units=units)
        return y
    
    def visit_Mul(self, node, *args, **kwargs):
        unit1, error1 = yield (node.operands[0], args, kwargs)
        unit2, error2 = yield (node.operands[1], args, kwargs)
        unit = {}
        for k in set(unit1) | set(unit2):
            if (v := unit1.get(k, 0) + unit2.get(k, 0)) != 0:
                unit[k] = v
        return unit, error1 + error2
    
    def visit_Div(self, node, *args, **kwargs):
        unit1, error1 = yield (node.operands[0], args, kwargs)
        unit2, error2 = yield (node.operands[1], args, kwargs)
        unit = {}
        for k in set(unit1) | set(unit2):
            if (v := unit1.get(k, 0) - unit2.get(k, 0)) != 0:
                unit[k] = v
        return unit, error1 + error2

    def visit_Pow(self, node, *args, **kwargs):
        unit, error = yield (node.operands[0], args, kwargs)
        if type(node.operands[1]).__name__ != 'Number':
            exponent_unit, exponent_error = yield (node.operands[1], args, kwargs)
            error += exponent_error
            if exponent_unit:
                error.append(
                    f"Unit inference failed for node {node}: "
                    f"Exponent must be dimensionless, but got {exponent_unit}."
                )
            if unit:
                error.append(
                    f"Unit inference failed for node {node}: A non-numeric "
                    f"exponent requires a dimensionless base, but got {unit}."
                )
            return {}, error
        else:
            exp = node.operands[1].value
            return {k: v * exp for k, v in unit.items()}, error

    def visit_Pow2(self, node, *args, **kwargs):
        unit, error = yield (node.operands[0], args, kwargs)
        return {k: v * 2 for k, v in unit.items()}, error

    def visit_Pow3(self, node, *args, **kwargs):
        unit, error = yield (node.operands[0], args, kwargs)
        return {k: v * 3 for k, v in unit.items()}, error

    def visit_Sqrt(self, node, *args, **kwargs):
        unit, error = yield (node.operands[0], args, kwargs)
        return {k: v * 0.5 for k, v in unit.items()}, error

    def visit_SqrtAbs(self, node, *args, **kwargs):
        unit, error = yield (node.operands[0], args, kwargs)
        return {k: v * 0.5 for k, v in unit.items()}, error

    def visit_Neg(self, node, *args, **kwargs):
        unit, error = yield (node.operands[0], args, kwargs)
        return unit, error

    def visit_Abs(self, node, *args, **kwargs):
        unit, error = yield (node.operands[0], args, kwargs)
        return unit, error

    def visit_Identity(self, node, *args, **kwargs):
        unit, error = yield (node.operands[0], args, kwargs)
        return unit, error

    def visit_Inv(self, node, *args, **kwargs):
        unit, error = yield (node.operands[0], args, kwargs)
        return {k: -v for k, v in unit.items()}, error

    def visit_Variable(self, node, *args, **kwargs):
        yield from yield_nothing()
        units = kwargs.get("units", {})
        if node.name not in units:
            raise ValueError(f"Unit inference failed for node {node}: Variable {node.name} not found in provided units.")
        if units[node.name] is None:
            raise ValueError(
                f"Unit inference failed for node {node}: Variable {node.name} "
                "has no declared or inferred unit."
            )
        return units[node.name].unit_dict, []
    
    def visit_Number(self, node, *args, **kwargs):
        yield from yield_nothing()
        return {}, []
    
    def visit_Empty(self, node, *args, **kwargs):
        yield from yield_nothing()
        raise ValueError(f"Unit inference failed for node {node}: Empty node encountered.")
    
    def generic_visit(self, node, *args, **kwargs):
        yield from yield_nothing()
        if node.n_operands == 0:
            raise ValueError(f"Please implement {type(self).__name__}.visit_{type(node).__name__}")
        elif node.n_operands == 1:
            unit, error = yield (node.operands[0], args, kwargs)
            if unit != {}:
                error.append(f"Unit inference failed for node {node}: Expected dimensionless operands, but got {unit}.")
            return {}, error
        elif node.n_operands == 2:
            unit1, error1 = yield (node.operands[0], args, kwargs)
            unit2, error2 = yield (node.operands[1], args, kwargs)
            error = error1 + error2
            if unit1 != unit2:
                error.append(f"Unit inference failed for node {node}: Expected operands with the same units, but got {unit1} and {unit2}.")
            return unit1, error
        else:
            raise ValueError(f"Please implement {type(self).__name__}.visit_{type(node).__name__}")
        


def unit_inference(formula: str | nd.Symbol, variables: List[VariableSpec]):
    if isinstance(formula, str):
        formula = nd.parse(formula)
    
    foo = UnitInferenceVisitor()
    units = {var.name: var.unit for var in variables}
    unit, error = foo(formula, units=units)
    return unit, error
