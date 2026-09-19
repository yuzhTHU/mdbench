"""Readable scientific examples for the eight constructed task families."""
from itertools import combinations
from pathlib import Path
import json

import numpy as np
import pytest

from src.synthetic_data import evaluate_expression, generate_synthetic_data
from src.validate_problem import load_task


ROOT = Path(__file__).resolve().parents[2]
CATALOGUE = json.loads((ROOT / 'playground/chemical_families/catalogue.json').read_text())
FAMILIES = sorted({entry['task_name'].rsplit(' - ', 1)[0] for entry in CATALOGUE})
TASK_FILES = {
    path.name: path
    for path in (ROOT / 'tasks' / 'chemistry').rglob('*')
    if path.is_file()
}


def task_path(task_name):
    return TASK_FILES[task_name + '.yaml']


def catalogue_path(entry):
    return TASK_FILES[Path(entry['path']).name]


@pytest.mark.parametrize('family', FAMILIES)
def test_each_family_has_blueprint_and_every_combination_up_to_five(family):
    expected = {()} | {c for n in range(1, 6) for c in combinations(range(1, 6), n)}
    entries = [entry for entry in CATALOGUE if entry['task_name'].startswith(family + ' - ')]
    assert len(entries) == 32
    assert {tuple(entry['mutations']) for entry in entries} == expected
    for entry in entries:
        path = catalogue_path(entry)
        task = load_task(path, validate=False, check_filename=False)
        assert path.name.removesuffix('.archived') == task.task_name + '.yaml'


@pytest.mark.parametrize('family', FAMILIES)
def test_variants_preserve_every_public_description_and_sampling_range(family):
    reference = load_task(task_path(family + ' - Original'), validate=False)
    for entry in CATALOGUE:
        if not entry['task_name'].startswith(family + ' - '):
            continue
        task = load_task(catalogue_path(entry), validate=False, check_filename=False)
        assert task.task_description == reference.task_description
        assert task.variables == reference.variables
        assert not task.by_role('auxiliary')
        assert len(task.mechanism_model) == len(task.by_role('internal')) + 1


@pytest.mark.parametrize('family,inputs,expected', [
    ('Lindemann Activation', {'A': 0.3, 'M': 0.4},
     {'j': 1.7 * 0.3 * 0.4, 'u': 1.7 * 0.3 * 0.4 / (0.8 * 0.4 + 2.3),
      'r': 2.3 * 1.7 * 0.3 * 0.4 / (0.8 * 0.4 + 2.3)}),
    ('Michaelis Menten Catalysis', {'S': 0.3, 'E': 1e-5},
     {'e': 1e-5 * 3 / (3 + 1.4 * 0.3), 'c': 1e-5 * 1.4 * 0.3 / (3 + 1.4 * 0.3),
      'r': 2.1 * 1e-5 * 1.4 * 0.3 / (3 + 1.4 * 0.3)}),
    ('Competitive Enzyme Inhibition', {'S': 0.3, 'I': 0.4, 'E': 1e-5},
     {'e': 1e-5 / (1 + 1.4 * 0.3 / 3 + 1.1 * 0.4 / 0.7),
      'q': 1e-5 * (1.1 * 0.4 / 0.7) / (1 + 1.4 * 0.3 / 3 + 1.1 * 0.4 / 0.7),
      'r': 2.1 * 1e-5 * (1.4 * 0.3 / 3) / (1 + 1.4 * 0.3 / 3 + 1.1 * 0.4 / 0.7)}),
    ('Langmuir Surface Adsorption', {'C': 0.3},
     {'v': 0.9 / (0.9 + 1.4 * 0.3), 'theta': 1.4 * 0.3 / (0.9 + 1.4 * 0.3)}),
    ('Surface Bimolecular Catalysis', {'A': 0.3, 'B': 0.4},
     {'a': 1.3 * 0.3 / (1 + 1.3 * 0.3 + 0.8 * 0.4),
      'b': 0.8 * 0.4 / (1 + 1.3 * 0.3 + 0.8 * 0.4),
      'r': 2.3 * 1.3 * 0.3 * 0.8 * 0.4 / (1 + 1.3 * 0.3 + 0.8 * 0.4)**2}),
    ('Consecutive Flow Reaction', {'A': 0.3, 'D': 0.4},
     {'b': 1.2 * 0.3 / (0.8 + 0.4),
      'c': 1.2 * 0.3 * 0.8 / ((0.8 + 0.4) * (1.6 + 0.4)),
      'r': 1.6 * 1.2 * 0.3 * 0.8 / ((0.8 + 0.4) * (1.6 + 0.4))}),
    ('Reversible Molecular Association', {'T': 0.3, 'L': 0.4},
     {'a': 0.3 / (1 + 1.8 * 0.4), 'b': 0.4, 'c': 0.3 * 1.8 * 0.4 / (1 + 1.8 * 0.4)}),
    ('Acid Base Buffer', {'A0': 20.0, 'B0': 30.0},
     {'a': 20, 'b': 30, 'h': 0.018 * 20 / 30, 'pH': -np.log10(1.8e-5 * 20 / 30)}),
])
def test_blueprints_reproduce_independently_calculated_scientific_laws(family, inputs, expected):
    task = load_task(task_path(family + ' - Original'))
    arrays = {name: np.array([value]) for name, value in inputs.items()}
    for name, value in expected.items():
        actual = evaluate_expression(task.solution[name], arrays, 1)
        np.testing.assert_allclose(actual, [value], rtol=1e-12)


@pytest.mark.parametrize('family', FAMILIES)
@pytest.mark.parametrize('suffix', ['Variant 5', 'Variant 1-2-3-4-5'])
def test_component_change_and_full_combination_change_target_and_hidden_probe(family, suffix):
    original = load_task(task_path(family + ' - Original'))
    combined = load_task(task_path(family + ' - ' + suffix))
    values = {v.name: np.array([v.sampling['ood_boundary']]) for v in original.by_role('input')}
    before = evaluate_expression(original.solution[original.target.name], values, 1)
    after = evaluate_expression(combined.solution[combined.target.name], values, 1)
    assert not np.allclose(before, after, rtol=1e-6, atol=0)
    assert any(not np.allclose(evaluate_expression(original.solution[p.probe], values, 1),
                               evaluate_expression(combined.solution[p.probe], values, 1),
                               rtol=1e-6, atol=0) for p in combined.mechanism_probes)


@pytest.mark.parametrize('family', FAMILIES)
def test_combined_models_generate_finite_id_and_ood_observations(family):
    task = load_task(task_path(family + ' - Variant 1-2-3-4-5'))
    data = generate_synthetic_data(task, seed=7, train_samples=16, id_test_samples=16, ood_test_samples=16)
    for split in ('train', 'id_test', 'ood_test'):
        assert data[split].shape == (len(task.observed), 16)
        assert np.isfinite(data[split]).all()
        assert (data[split] > 0).all()
    for row, variable in enumerate(task.observed):
        if variable.role != 'input':
            continue
        assert (data['train'][row] < variable.sampling['ood_boundary']).all()
        assert (data['ood_test'][row] > variable.sampling['ood_boundary']).all()


def test_trivial_adsorption_probe_candidates_are_archived_from_automatic_discovery():
    archived = [entry for entry in CATALOGUE if entry['status'] == 'archived']
    assert len(archived) == 7
    for entry in archived:
        assert entry['task_name'].startswith('Langmuir Surface Adsorption - Variant ')
        assert not set(entry['mutations']) & {4, 5}
        assert entry['path'].endswith('.yaml.archived')
        assert not Path(str(catalogue_path(entry)).removesuffix('.archived')).exists()
        assert 'not identifiable' in entry['archive_reason']
