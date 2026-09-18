"""Task instance schema from proposal.md; independent of legacy interfaces."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import sympy as sp

VariableRole = Literal['target', 'input', 'internal', 'auxiliary']
MECHANISM_ROLES = (
    'governing/dynamical laws',
    'conservation/balance relations',
    'constitutive/component relations',
    'kinematic/geometric constraints',
    'auxiliary/regime constraints',
)

@dataclass
class VariableSpec:
    name: str
    description: str
    unit: str | None
    role: VariableRole
    sampling: dict[str, float | str] | None = None

@dataclass
class MechanismItem:
    formula_str: str
    formula: sp.Expr  # LHS - RHS
    role: str
    description: str

@dataclass
class MechanismProbe:
    probe: str
    description: str
    answer: str

@dataclass
class Task:
    task_name: str
    task_description: str
    mutation: str
    mechanism_model: list[MechanismItem]
    phenomenal_model: str
    variables: list[VariableSpec]
    mechanism_probes: list[MechanismProbe]
    solution: dict[str, sp.Expr] = field(default_factory=dict)

    def by_role(self, *roles: str) -> list[VariableSpec]:
        return [v for v in self.variables if v.role in roles]

    @property
    def target(self) -> VariableSpec:
        targets = self.by_role('target')
        if len(targets) != 1:
            raise ValueError('Exactly one target variable is required.')
        return targets[0]

    @property
    def observed(self) -> list[VariableSpec]:
        # This order is also the row order in every exported NPY array.
        return [self.target, *self.by_role('input', 'auxiliary')]
