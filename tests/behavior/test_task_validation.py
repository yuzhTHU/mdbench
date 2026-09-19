"""Executable examples of task-file and collection validation."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from src.validate_problem import get_parser, load_task, main

# One length-valued hidden state and one area-valued target. These toy tasks
# illustrate software contracts, not scientifically reviewed benchmark items.
TASK_YAML = """\
task_name: Response - Variant 1
task_description: Relate the input length to the output area.
mutation: Add an offset to the hidden response.
mechanism_model:
  - {formula: 'h = 2*x + 3', role: constitutive/component relations, description: Hidden response.}
  - {formula: 'y = h*x', role: conservation/balance relations, description: Output balance.}
phenomenal_model: y = (2*x + 3)*x
variables:
  - {name: y, description: Output area., unit: 'm^2', role: target}
  - name: x
    description: Input length.
    unit: m
    role: input
    sampling: {min: 1.0, max: 8.0, ood_boundary: 4.0, distribution: uniform}
  - {name: h, description: Hidden length., unit: m, role: internal}
mechanism_probes:
  - {probe: h, description: Hidden length., answer: h}
"""


def test_validate_cli_reports_success_for_a_complete_task_file(tmp_path):
    path = tmp_path / 'Response - Variant 1.yaml'
    path.write_text(TASK_YAML)

    result = subprocess.run(
        [sys.executable, '-m', 'src.validate_problem', '--tasks', str(path)],
        cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=15,
    )

    assert result.returncode == 0, result.stderr
    report, = json.loads(result.stdout)
    assert report['task_name'] == 'Response - Variant 1'
    assert report['ok'] is True
    assert report['warnings'] == []
    assert set(report['solution']) == {'h', 'y'}


def test_task_name_must_match_the_yaml_filename(tmp_path):
    path = tmp_path / 'Different Name - Variant 1.yaml'
    path.write_text(TASK_YAML)  # task_name is still Response - Variant 1.

    with pytest.raises(ValueError, match='task_name must equal filename'):
        load_task(path)


def test_validate_collection_rejects_duplicate_names_in_different_directories(tmp_path, capsys):
    for folder in ('first', 'second'):
        directory = tmp_path / folder
        directory.mkdir()
        (directory / 'Response - Variant 1.yaml').write_text(TASK_YAML)

    args = get_parser().parse_args(['--tasks', str(tmp_path / 'first'), str(tmp_path / 'second')])
    exit_code = main(args)
    reports = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert reports[0]['ok'] is True
    assert reports[1]['ok'] is False
    assert 'Duplicate task_name' in reports[1]['error']


def test_family_variants_must_have_identical_public_descriptions(tmp_path, capsys):
    (tmp_path / 'Response - Original.yaml').write_text(
        TASK_YAML.replace('Response - Variant 1', 'Response - Original')
    )
    (tmp_path / 'Response - Variant 1.yaml').write_text(
        TASK_YAML.replace('Relate the input length to the output area.', 'Reveal the hidden offset.')
    )

    exit_code = main(get_parser().parse_args(['--tasks', str(tmp_path)]))
    reports = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert sum(report['ok'] for report in reports) == 1
    assert 'family descriptions must be identical' in reports[1]['error']


@pytest.mark.parametrize(('original', 'replacement', 'error'), [
    ('task_name:', 'problem_name:', 'missing fields'),
    ('role: internal', 'role: intermediate', 'Invalid variable role'),
    ('name: h,', 'name: 1h,', 'Invalid/reserved variable name'),
    ('name: h,', 'name: x,', 'Duplicate variable names'),
    ('role: target', 'role: internal', 'Exactly one target'),
    ('role: internal', 'role: target', 'Exactly one target'),
])
def test_loading_rejects_declarations_outside_the_task_schema(tmp_path, original, replacement, error):
    path = tmp_path / 'Response - Variant 1.yaml'
    path.write_text(TASK_YAML.replace(original, replacement))

    with pytest.raises(ValueError, match=error):
        load_task(path)


@pytest.mark.parametrize('boundary', [1.0, 8.0])
def test_ood_boundary_must_be_strictly_inside_the_sampling_range(tmp_path, boundary):
    path = tmp_path / 'Response - Variant 1.yaml'
    path.write_text(TASK_YAML.replace('ood_boundary: 4.0', f'ood_boundary: {boundary}'))

    with pytest.raises(ValueError, match='min < ood_boundary < max'):
        load_task(path)


@pytest.mark.parametrize('variable', ['y', 'h'])
def test_target_and_internal_variables_cannot_request_sampling(tmp_path, variable):
    path = tmp_path / 'Response - Variant 1.yaml'
    line = next(line for line in TASK_YAML.splitlines() if f'name: {variable},' in line)
    sampled_line = line[:-1] + ', sampling: {min: 1, max: 8, ood_boundary: 4, distribution: uniform}}'
    path.write_text(TASK_YAML.replace(line, sampled_line))

    with pytest.raises(ValueError, match='Internal/target variables cannot declare sampling'):
        load_task(path)


def test_validate_directory_ignores_legacy_and_archived_tasks(tmp_path, capsys):
    (tmp_path / 'Response - Variant 1.yaml').write_text(TASK_YAML)
    (tmp_path / 'legacy').mkdir()
    (tmp_path / 'legacy' / 'broken.yaml').write_text('not a task')
    (tmp_path / 'broken.yaml.archived').write_text('not a task')

    exit_code = main(get_parser().parse_args(['--tasks', str(tmp_path)]))
    reports = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert [report['task_name'] for report in reports] == ['Response - Variant 1']
