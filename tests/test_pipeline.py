from concurrent.futures import ThreadPoolExecutor
import io
import json
from types import SimpleNamespace
import threading
import numpy as np
import pytest
import requests
from src.export_problems import export_task, public_problem
from src.synthetic_data import generate_synthetic_data
from src.scoring import accuracy_metrics, feedback
from src.feedback_server import create_server, FeedbackCache
from src.evaluate import evaluate
from src.validate_problem import task_from_dict, validate_task


def test_sampling_export_privacy_reproducibility(demo, tmp_path):
    counts = dict(seed=11, train_samples=16, id_test_samples=8, ood_test_samples=6)
    first = generate_synthetic_data(demo, **counts)
    second = generate_synthetic_data(demo, **counts)
    for split in ('train', 'id_test', 'ood_test'): np.testing.assert_array_equal(first[split], second[split])
    for row, variable in enumerate(demo.observed):
        if variable.role == 'target': continue
        assert (first['train'][row] < variable.sampling['ood_boundary']).all()
        assert (first['ood_test'][row] >= variable.sampling['ood_boundary']).all()
    paths = export_task(demo, tmp_path, **counts)
    raw = json.loads(paths['problem'].read_text())
    assert set(raw) == {'task_description', 'variables', 'data_columns', 'data_layout'}
    assert all(v['role'] in ('input', 'target') for v in raw['variables'])
    assert all(v.name not in [item['name'] for item in raw['variables']] for v in demo.by_role('internal'))
    assert 'mechanism_probes' in json.loads(paths['answer'].read_text())['task']
    with pytest.raises(FileExistsError): export_task(demo, tmp_path, **counts)


def test_feedback_uses_only_public_model_and_train(demo):
    data = generate_synthetic_data(demo, train_samples=12, id_test_samples=4, ood_test_samples=4)['train']
    submission = '\n'.join(m.formula_str for m in demo.mechanism_model)
    result = feedback(public_problem(demo), data, submission)
    assert result['ok'] and result['train']['r2'] == 1
    assert result['train']['numerically_equivalent']
    assert all(v.name in result['solution'] for v in demo.by_role('internal'))
    bad = feedback(public_problem(demo), data, 'P=V')
    assert bad['train']['r2'] < .9


def test_nonfinite_is_failure_and_constant_metrics():
    result = accuracy_metrics([1, np.nan], [1, 1])
    assert result['r2'] is None and result['finite_fraction'] == .5
    assert accuracy_metrics([1,1], [1,1])['r2'] == 1
    assert accuracy_metrics([2,2], [1,1])['r2'] == 0
    assert not accuracy_metrics([0, 0], [1e-20, 1e-20])['numerically_equivalent']


def test_concurrent_multipart_and_cache(simple_raw, tmp_path):
    task = task_from_dict(simple_raw)
    paths = export_task(task, tmp_path, train_samples=12, id_test_samples=4, ood_test_samples=4)
    contents = {'problem': paths['problem'].read_bytes(), 'train_data': paths['train'].read_bytes(),
                'submission': b'h=2*x+3\ny=h*x'}
    server = create_server(port=0, workers=2)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{server.server_port}/evaluate'
    session = requests.Session()
    session.trust_env = False
    try:
        def post(_):
            return session.post(url, files={k: (k, v) for k, v in contents.items()}, timeout=10).json()
        with ThreadPoolExecutor(max_workers=6) as pool: results = list(pool.map(post, range(6)))
        assert all(r['ok'] and r['train']['r2'] == 1 for r in results)
        assert len(server.cache.results) == 1
        missing = session.post(url, files={'submission': ('a.txt', b'y=x')}, timeout=10)
        assert missing.status_code == 400
        changed = dict(contents, submission=b'y=x')
        assert not server.cache.evaluate(changed)['train']['numerically_equivalent']
        assert len(server.cache.results) == 2
    finally:
        server.shutdown(); server.server_close(); thread.join()


def test_independent_probe_restores_and_scoring(simple_raw, tmp_path):
    task = task_from_dict(simple_raw)
    paths = export_task(task, tmp_path / 'task', train_samples=16, id_test_samples=8, ood_test_samples=8)
    submission = ['u=2*x+3', 'y=u*x']  # different internals; probe expansion must use these.
    calls = []
    def ask(question, output_dir=None):
        assert 'answer:' not in question
        calls.append((question, output_dir))
        return 'h=u'
    args = SimpleNamespace(save_path=tmp_path / 'result', algorithm='codex', probe_workers=2)
    result = evaluate(args, paths['answer'], submission, ask)
    assert len(calls) == 1
    assert result['phenomenal']['ood_test']['r2'] == 1
    assert result['mechanism_probes'][0]['scores']['id_test']['symbolically_equivalent']
    assert result['mechanism_recovery']['ood_test']['numerically_equivalent'] == 1


def test_bad_probe_does_not_use_ground_truth_to_repair(simple_raw, tmp_path):
    task = task_from_dict(simple_raw)
    paths = export_task(task, tmp_path / 'task', train_samples=4, id_test_samples=4, ood_test_samples=4)
    submission = ['y=(2*x+3)*x']
    def ask(*args): return 'h=h'
    args = SimpleNamespace(save_path=tmp_path / 'result', algorithm='codex', probe_workers=2)
    result = evaluate(args, paths['answer'], submission, ask)
    assert result['phenomenal']['train']['symbolically_equivalent']
    assert not result['mechanism_probes'][0]['ok']
    assert result['mechanism_recovery']['train']['numerically_equivalent'] == 0
