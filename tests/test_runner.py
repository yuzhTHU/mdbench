import json
from pathlib import Path
from types import SimpleNamespace
import requests
import run as runner
from src.export_problems import export_task


def arguments(paths, save):
    return SimpleNamespace(problem_file=paths['problem'], train_data_npy_file=paths['train'], answer=paths['answer'],
                           save_path=save, save_dir=save.parent, algorithm='codex', feedback_server_url=None,
                           feedback_host='127.0.0.1', feedback_port=0, feedback_workers=2, feedback_cache_size=4)


def test_runner_server_lifecycle_model_and_performance(demo, tmp_path, monkeypatch):
    paths = export_task(demo, tmp_path / 'task', train_samples=4, id_test_samples=4, ood_test_samples=4)
    seen = []
    def algorithm(args, problem, train, url):
        session = requests.Session(); session.trust_env = False
        assert session.get(url.replace('/evaluate', '/health'), timeout=5).json()['ok']
        seen.append(url)
        return {'submission': 'submission.txt', 'session': 'session.jsonl', 'extra': '\x1b[31mpreserved\x1b[0m'}
    monkeypatch.setattr(runner, 'get_algorithm', lambda name: algorithm)
    def evaluate(args, answer, submission, checkpoint, model):
        assert model['extra'] == 'preserved'
        assert checkpoint == 'session.jsonl'
        return {'phenomenal': {'ok': True}, 'mechanism_probes': []}
    monkeypatch.setattr(runner, 'evaluate', evaluate)
    args = arguments(paths, tmp_path / 'run')
    assert runner.main(args) == 0
    performance = json.loads((Path(args.save_path) / 'performance.json').read_text())
    assert performance['agent_seconds'] >= 0 and performance['evaluation_seconds'] >= 0
    session = requests.Session(); session.trust_env = False
    try: session.get(seen[0], timeout=1)
    except requests.ConnectionError: pass
    else: assert False, 'Owned server was left running'


def test_runner_records_failure_without_ansi(demo, tmp_path, monkeypatch):
    paths = export_task(demo, tmp_path / 'task', train_samples=4, id_test_samples=4, ood_test_samples=4)
    def algorithm(*args): raise ValueError('\x1b[31mfailed\x1b[0m')
    monkeypatch.setattr(runner, 'get_algorithm', lambda name: algorithm)
    args = arguments(paths, tmp_path / 'run')
    assert runner.main(args) == 1
    text = (Path(args.save_path) / 'performance.json').read_text()
    assert '\\u001b' not in text and json.loads(text)['error'] == 'failed'
