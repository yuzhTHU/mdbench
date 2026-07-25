"""Build a readable dependency graph for a solved mechanism system."""
from __future__ import annotations

from dataclasses import dataclass

import nd2py as nd

from ...core import Problem


def _variables(formula: str) -> list[str]:
    names = []
    for node in nd.parse(formula).iter_preorder():
        if isinstance(node, nd.Variable) and node.name not in names:
            names.append(node.name)
    return names


def _quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


@dataclass(frozen=True)
class _Edge:
    source: str
    target: str
    implicit: bool = False
    numerical: bool = False


class MechanismGraphBuilder:
    """Convert a solved problem into a layered Graphviz dependency graph.

    Explicit solution steps form a left-to-right DAG. Variables solved from a
    self-referential or coupled equation system are enclosed in one highlighted
    cluster and connected as a directed cycle.
    """

    @staticmethod
    def _is_implicit(problem: Problem, variables: list[str]) -> bool:
        targets = set(variables)
        if len(targets) > 1:
            return True
        target = variables[0]
        return any(
            item.variable == target and target in _variables(item.formula)
            for item in problem.mechanism
        )

    @staticmethod
    def _implicit_dependencies(
        problem: Problem,
        targets: set[str],
        known: set[str],
    ) -> set[str]:
        dependencies = set()
        for item in problem.mechanism:
            names = {item.variable, *_variables(item.formula)}
            if names & targets and names <= known | targets:
                dependencies.update(names - targets)
        return dependencies

    def build(
        self,
        problem: Problem,
        *,
        include_constants: bool = True,
    ) -> str:
        if not problem.solution:
            raise ValueError(
                "Problem has no solution. Call solve_mechanism_equations(problem) first."
            )

        constant_names = {constant.name for constant in problem.constants}
        auxiliary_names = {
            variable.name for variable in problem.auxiliary_input_variables
        }
        input_names = {variable.name for variable in problem.input_variables}
        source_names = input_names | auxiliary_names
        if include_constants:
            source_names |= constant_names

        implicit_steps = {
            index
            for index, item in enumerate(problem.solution)
            if self._is_implicit(problem, item.variables)
        }
        implicit_names = {
            variable
            for index in implicit_steps
            for variable in problem.solution[index].variables
        }

        edges: set[_Edge] = set()
        known_names = set(source_names)
        for index, item in enumerate(problem.solution):
            targets = set(item.variables)
            if item.formulas:
                for target, formula in zip(item.variables, item.formulas, strict=True):
                    for dependency in _variables(formula):
                        if dependency != target:
                            edges.add(_Edge(dependency, target))
            else:
                for dependency in self._implicit_dependencies(
                    problem,
                    targets,
                    known_names,
                ):
                    for target in item.variables:
                        edges.add(_Edge(dependency, target, numerical=True))

            if index in implicit_steps:
                if len(item.variables) == 1:
                    variable = item.variables[0]
                    edges.add(
                        _Edge(
                            variable,
                            variable,
                            implicit=True,
                            numerical=not item.formulas,
                        )
                    )
                else:
                    cycle = [*item.variables, item.variables[0]]
                    for source, target in zip(cycle[:-1], cycle[1:], strict=True):
                        edges.add(
                            _Edge(
                                source,
                                target,
                                implicit=True,
                                numerical=not item.formulas,
                            )
                        )
            known_names.update(targets)

        if not include_constants:
            edges = {
                edge
                for edge in edges
                if edge.source not in constant_names
                and edge.target not in constant_names
            }

        visible_names = {variable.name for variable in problem.all_variables}
        if include_constants:
            visible_names |= constant_names
        visible_names |= {
            name for edge in edges for name in (edge.source, edge.target)
        }
        node_ids = {
            name: f"n{index}" for index, name in enumerate(sorted(visible_names))
        }

        lines = [
            "digraph mechanism {",
            '  graph [rankdir=LR, bgcolor="white", pad="0.35", nodesep="0.55", '
            'ranksep="1.05", overlap=false, splines=spline, outputorder=edgesfirst, '
            'newrank=true, compound=true];',
            '  node [shape=box, style="rounded,filled", fillcolor="#FFFFFF", '
            'color="#94A3B8", fontname="DejaVu Sans", fontsize=11, '
            'fontcolor="#0F172A", margin="0.18,0.11", penwidth=1.2];',
            '  edge [color="#64748B", penwidth=1.15, arrowsize=0.72, '
            'arrowhead=vee];',
        ]

        for name in sorted(visible_names):
            attributes = [f"label={_quote(name)}"]
            if name == problem.target_variable.name:
                attributes += [
                    "shape=doublecircle",
                    'fillcolor="#FEF3C7"',
                    'color="#D97706"',
                    "penwidth=1.8",
                ]
            elif name in implicit_names:
                attributes += [
                    "shape=circle",
                    'fillcolor="#F3E8FF"',
                    'color="#8B5CF6"',
                    "penwidth=1.6",
                ]
            elif name in constant_names:
                attributes += [
                    "shape=diamond",
                    'fillcolor="#F8FAFC"',
                    'color="#94A3B8"',
                    "fontsize=9",
                ]
            elif name in auxiliary_names:
                attributes += [
                    "shape=ellipse",
                    'style="dashed,filled"',
                    'fillcolor="#ECFEFF"',
                    'color="#0891B2"',
                ]
            elif name in input_names:
                attributes += [
                    "shape=ellipse",
                    'fillcolor="#EFF6FF"',
                    'color="#3B82F6"',
                ]
            lines.append(f"  {node_ids[name]} [{', '.join(attributes)}];")

        if source_names:
            members = "; ".join(
                node_ids[name] for name in sorted(source_names) if name in node_ids
            )
            lines.append(f"  {{ rank=source; {members}; }}")

        for index in sorted(implicit_steps):
            item = problem.solution[index]
            members = [node_ids[name] for name in item.variables]
            lines.append(f"  subgraph cluster_implicit_{index + 1} {{")
            lines.append(
                '    label="Implicit solution"; labelloc="t"; fontsize=10; '
                'fontname="DejaVu Sans"; fontcolor="#6D28D9"; color="#C4B5FD"; '
                'fillcolor="#FAF5FF"; style="rounded,filled"; penwidth=1.2;'
            )
            lines.append("    rank=same; " + "; ".join(members) + ";")
            lines.append("  }")

        representatives = [
            node_ids[item.variables[0]]
            for item in problem.solution
            if item.variables
        ]
        for previous, current in zip(representatives, representatives[1:]):
            lines.append(
                f"  {previous} -> {current} "
                '[style=invis, weight=30, minlen=1];'
            )

        for edge in sorted(
            edges,
            key=lambda value: (
                value.implicit,
                value.numerical,
                value.source,
                value.target,
            ),
        ):
            attributes = []
            if edge.implicit:
                attributes += [
                    'color="#8B5CF6"',
                    "penwidth=1.8",
                    "arrowsize=0.8",
                    "constraint=false",
                ]
            if edge.numerical:
                attributes.append("style=dashed")
            suffix = f" [{', '.join(attributes)}]" if attributes else ""
            lines.append(
                f"  {node_ids[edge.source]} -> {node_ids[edge.target]}{suffix};"
            )

        lines.append("}")
        return "\n".join(lines) + "\n"
