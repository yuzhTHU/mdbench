from src.evaluate_result import format_reports
from src.features.answer import build_answer
from src.features.evaluation import trace_mechanism_submission
from src.features.io import load_problem


def test_trace_expands_ordered_mechanism_steps_to_target():
    problem = load_problem("problems/demo_problem.yaml", solve=False)
    answer = build_answer(problem, "mechanism_explanation")

    trace = trace_mechanism_submission(answer["mechanisms"], answer)

    assert trace["submitted_equations"] == [
        item["formula"] for item in answer["mechanisms"]
    ]
    assert [step["variables"] for step in trace["solution_steps"]] == [
        ["r"], ["F"], ["acc"], ["v"], ["T"]
    ]
    assert "G" in trace["derived_formula"]
    assert "M" in trace["derived_formula"]
    assert not trace["uses_numerical_solver"]


def test_trace_marks_numerically_solved_implicit_system():
    answer = {
        "task": "mechanism_explanation",
        "target_variable": "y",
        "source_variables": ["x"],
        "phenomenological_formula": "y = x",
        "constants": [],
        "intermediate_variables": [],
        "mechanisms": [],
    }
    mechanisms = [
        {"formula": "a = cos(a) + x", "formula_description": "implicit"},
        {"formula": "y = a", "formula_description": "target"},
    ]

    trace = trace_mechanism_submission(mechanisms, answer)

    assert trace["uses_numerical_solver"]
    assert trace["solution_steps"][0]["variables"] == ["a"]
    assert trace["solution_steps"][0]["numerical"]
    assert trace["solution_steps"][0]["formulas"] == []


def test_verbose_report_formats_mechanism_trace():
    report = {
        "derivation": {"score": 0.5, "equivalent": False},
        "structure_recovery": {
            "score": 0.5,
            "formula_similarity": 0.6,
            "dag_similarity": 0.4,
        },
        "mechanism_trace": {
            "submitted_equations": ["a = cos(a) + x", "y = a"],
            "solution_steps": [
                {
                    "variables": ["a"],
                    "formulas": [],
                    "original_formulas": ["cos(a) + x"],
                    "expanded_formulas": [],
                    "show_expanded_formulas": [False],
                    "numerical": True,
                },
                {
                    "variables": ["y"],
                    "formulas": ["a"],
                    "original_formulas": ["a"],
                    "expanded_formulas": ["a"],
                    "show_expanded_formulas": [False],
                    "numerical": False,
                },
            ],
            "target_variable": "y",
            "derived_formula": "a",
            "uses_numerical_solver": True,
        },
        "mechanism_simplicity": {
            "mean_ast_nodes_per_item": 5.0,
            "maximum_ast_nodes": 6,
            "total_ast_nodes": 10,
            "item_count": 2,
        },
    }

    rendered = format_reports(report)

    assert "Verbose mechanism trace" in rendered
    assert "solved numerically as an implicit equation system" in rendered
    assert "2. y = a" in rendered
    assert "50.00%" in rendered
    assert "Mechanism description complexity" in rendered
