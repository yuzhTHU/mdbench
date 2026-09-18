from src.metrics import StructuralMechanismMatcher


def _items(*equations):
    return [{"formula": equation} for equation in equations]


def test_ignores_variable_names_numeric_values_and_item_order():
    reference = _items("a = 2 * x", "b = sin(a) + y", "z = b / 3")
    candidate = _items("out = q / 99", "q = sin(hidden) + v", "hidden = 7 * u")

    report = StructuralMechanismMatcher().compare(candidate, reference)

    assert report["score"] == 1.0
    assert report["formula_similarity"] == 1.0
    assert report["dag_similarity"] == 1.0


def test_penalizes_changed_formula_and_dependency_structure():
    reference = _items("a = x + y", "b = a * y", "z = sin(b)")
    candidate = _items("p = u + v", "q = u * v", "out = exp(q)")

    report = StructuralMechanismMatcher().compare(candidate, reference)

    assert 0.0 < report["score"] < 1.0
    assert report["dag_similarity"] < 1.0
    assert report["formula_similarity"] < 1.0


def test_unmatched_relationship_reduces_recovery():
    reference = _items("a = x", "y = a")
    candidate = _items("p = u", "q = p", "out = q")

    assert StructuralMechanismMatcher().compare(candidate, reference)["score"] < 1.0


def test_implicit_self_dependency_affects_graph_recovery():
    reference = _items("a = cos(a) + x", "y = a")
    candidate = _items("p = cos(x) + x", "out = p")

    report = StructuralMechanismMatcher().compare(candidate, reference)

    assert report["dag_similarity"] < 1.0
