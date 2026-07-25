# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
from typing import List, Dict, Literal, get_args
from dataclasses import dataclass, field
from .solution import SolutionItem

SI = Literal['kg', 'm', 's', 'A', 'K', 'mol', 'cd']

class UNIT:
    def __init__(self, unit_dict: Dict[SI, int | float]):
        self.unit_dict = unit_dict

    def __repr__(self):
        return f"UNIT({self.unit_dict})"

    def __str__(self):
        if not self.unit_dict:
            return "1 (dimensionless)"
        return " ".join(
            name if exponent == 1 else f"{name}^{exponent:g}"
            for name, exponent in sorted(self.unit_dict.items(), key=lambda x: get_args(SI).index(x[0]))
        )
    
    def __eq__(self, other):
        if not isinstance(other, UNIT):
            return NotImplemented
        return self.unit_dict == other.unit_dict

    def to_dict(self) -> Dict[str, int | float]:
        """Return the dependency-free interchange representation."""
        return dict(self.unit_dict)

@dataclass
class VariableSpec:
    name: str
    description: str
    unit: UNIT | None
    sampling: Dict[str, float | str] | None = field(default=None, kw_only=True)


@dataclass
class ConstantSpec(VariableSpec):
    name: str
    description: str
    unit: UNIT
    value: int | float


@dataclass
class MechanismItem:
    variable: str
    formula: str  # An nd2py-compatible right-hand-side expression.
    formula_description: str

    @property
    def equation(self) -> str:
        """Return the complete mechanism equation."""
        return f"{self.variable} = {self.formula}"


@dataclass
class Problem:
    problem_name: str
    problem_description: str
    phenomenological_formula: str  # nd2py-compatible right-hand side.
    target_variable: VariableSpec
    input_variables: List[VariableSpec]
    intermediate_variables: List[VariableSpec]
    mechanism: List[MechanismItem]
    auxiliary_input_variables: List[VariableSpec] = field(default_factory=list)
    constants: List[ConstantSpec] = field(default_factory=list)
    solution: List[SolutionItem] = field(default_factory=list)

    @property
    def all_variables(self) -> List[VariableSpec]:
        """All non-constant variables in their schema order."""
        return [
            self.target_variable,
            *self.input_variables,
            *self.intermediate_variables,
            *self.auxiliary_input_variables,
        ]
