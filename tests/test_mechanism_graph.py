from src.features.io import load_problem, solve_mechanism_equations
from src.features.visualization import MechanismGraphBuilder


def test_graph_points_sources_through_drag_to_target():
    problem = load_problem("problems/settling-terminal-velocity-quadratic-drag.yaml")
    problem.solution = solve_mechanism_equations(problem)
    dot = MechanismGraphBuilder().build(problem)

    node_by_label = {}
    for line in dot.splitlines():
        if " [label=\"" in line:
            node_id = line.strip().split()[0]
            label = line.split('label="', 1)[1].split('"', 1)[0]
            node_by_label[label] = node_id

    assert f"{node_by_label['F_g']} -> {node_by_label['F_d']}" in dot
    assert f"{node_by_label['F_b']} -> {node_by_label['F_d']}" in dot
    assert f"{node_by_label['F_d']} -> {node_by_label['v_T']}" in dot
    assert f"{node_by_label['C_d']} -> {node_by_label['v_T']}" in dot
    assert "outputorder=edgesfirst" in dot
    assert "rankdir=LR" in dot
    assert "Step " not in dot


def test_graph_can_hide_constants():
    problem = load_problem("problems/demo_problem.yaml")
    problem.solution = solve_mechanism_equations(problem)
    dot = MechanismGraphBuilder().build(problem, include_constants=False)
    assert 'label="G"' not in dot
    assert 'label="π"' not in dot


def test_implicit_solution_is_drawn_as_a_cycle():
    from src.core import MechanismItem, Problem, UNIT, VariableSpec

    variables = {
        name: VariableSpec(name, name, UNIT({}))
        for name in ("x", "a", "b", "y")
    }
    problem = Problem(
        "implicit",
        "implicit",
        "4 * x",
        variables["y"],
        [variables["x"]],
        [variables["a"], variables["b"]],
        [
            MechanismItem("a", "x + b", "first relation"),
            MechanismItem("b", "a / 2", "second relation"),
            MechanismItem("y", "2 * a", "target relation"),
        ],
    )
    problem.solution = solve_mechanism_equations(problem)
    dot = MechanismGraphBuilder().build(problem)

    node_by_label = {}
    for line in dot.splitlines():
        if " [label=\"" in line:
            node_id = line.strip().split()[0]
            label = line.split('label="', 1)[1].split('"', 1)[0]
            node_by_label[label] = node_id

    assert 'label="Implicit solution"' in dot
    assert f"{node_by_label['a']} -> {node_by_label['b']}" in dot
    assert f"{node_by_label['b']} -> {node_by_label['a']}" in dot


def test_numerical_solution_uses_dashed_edges():
    from src.core import MechanismItem, Problem, UNIT, VariableSpec

    a = VariableSpec("a", "a", UNIT({}))
    x = VariableSpec("x", "x", UNIT({}))
    problem = Problem(
        "numerical",
        "numerical",
        "x",
        a,
        [x],
        [],
        [MechanismItem("a", "cos(a) + x", "fixed point")],
    )
    problem.solution = solve_mechanism_equations(problem)
    dot = MechanismGraphBuilder().build(problem)

    assert "style=dashed" in dot
    assert 'label="Implicit solution"' in dot
