import json
from pathlib import Path
from types import SimpleNamespace
import requests
import run as runner
from src.export_problems import export_task


def test_runner_parser_matches_common_experiment_structure():
    required = ['--problem_file', 'problem.json', '--answer', 'answer.json']
    parser = runner.build_argparser(required)
    args = parser.parse_args(required + ['--exp-name', 'demo', '--codex-text-only',
                                         '--no-codex-text-only', '--verbose', '--no-verbose'])
    assert args.name == 'run' and args.exp_name == 'demo'
    assert args.save_dir == './logs/run' and args.seed == -1
    assert args.codex_text_only is False and args.verbose is False


def arguments(paths, save):
    return SimpleNamespace(problem_file=paths['problem'], answer=paths['answer'],
                           save_path=save, save_dir=save.parent, algorithm='codex', feedback_server_url=None,
                           feedback_host='127.0.0.1', feedback_port=0, feedback_workers=2, feedback_cache_size=4)


def test_runner_server_lifecycle_submission_and_performance(demo, tmp_path, monkeypatch):
    paths = export_task(demo, tmp_path / 'task', train_samples=4, id_test_samples=4, ood_test_samples=4)
    seen, restored = [], []
    checkpoint = {'nested': []}
    def algorithm(args, problem, train, url):
        session = requests.Session(); session.trust_env = False
        assert session.get(url.replace('/evaluate', '/health'), timeout=5).json()['ok']
        seen.append(url)
        return ['y=x'], checkpoint
    monkeypatch.setattr(runner, 'get_algorithm', lambda name: algorithm)
    def ask_factory(args, received):
        restored.append(received)
        received['nested'].append('probe-local-mutation')
        return lambda question, probe_name, probe_description, output_dir=None: 'h=x'
    monkeypatch.setattr(runner, 'get_ask', lambda name: ask_factory)
    def evaluate(args, answer, submission, make_ask):
        assert submission == ['y=x']
        first, second = make_ask(), make_ask()
        assert first('derive h (h) as a function', 'h', 'h') == 'h=x'
        assert second('derive h (h) as a function', 'h', 'h') == 'h=x'
        return {'phenomenal': {'ok': True}, 'mechanism_probes': []}
    monkeypatch.setattr(runner, 'evaluate', evaluate)
    args = arguments(paths, tmp_path / 'run')
    assert runner.main(args) == 0
    assert checkpoint == {'nested': []}
    assert len(restored) == 2 and restored[0] is not restored[1]
    assert restored == [
        {'nested': ['probe-local-mutation']},
        {'nested': ['probe-local-mutation']}]
    performance = json.loads((Path(args.save_path) / 'performance.json').read_text())
    assert performance['agent_seconds'] >= 0 and performance['evaluation_seconds'] >= 0
    assert (Path(args.save_path) / 'submission.txt').read_text() == 'y=x\n'
    assert not (Path(args.save_path) / 'saved_checkpoint' / 'model.json').exists()
    assert 'model' not in performance
    dataest = Path(args.save_path) / 'dataest'
    assert {p.name for p in dataest.iterdir()} == {
        'problem.json', 'answer.json', 'train.npy', 'id_test.npy', 'ood_test.npy'}
    staged_answer = json.loads((dataest / 'answer.json').read_text())
    assert staged_answer['data'] == {
        'train': 'train.npy', 'id_test': 'id_test.npy', 'ood_test': 'ood_test.npy'}
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


def test_runner_requires_explicit_submission_return(demo, tmp_path, monkeypatch):
    paths = export_task(demo, tmp_path / 'task', train_samples=4, id_test_samples=4, ood_test_samples=4)
    monkeypatch.setattr(runner, 'get_algorithm', lambda name: lambda *args: {
        'session': 'session.jsonl', 'events': 'events.jsonl'})
    args = arguments(paths, tmp_path / 'run')
    assert runner.main(args) == 1
    performance = json.loads((Path(args.save_path) / 'performance.json').read_text())
    assert performance['error_type'] == 'TypeError'
    assert '(submission, checkpoint) tuple' in performance['error']
    assert not (Path(args.save_path) / 'submission.txt').exists()


def test_runner_default_path_uses_experiment_and_task_names(demo, tmp_path, monkeypatch):
    paths = export_task(demo, tmp_path / 'task', train_samples=4, id_test_samples=4, ood_test_samples=4)
    monkeypatch.setattr(runner, 'get_algorithm', lambda name: lambda *args: (
        ['y=x'], {'frozen': True}))
    monkeypatch.setattr(runner, 'get_ask', lambda name: lambda args, checkpoint: None)
    monkeypatch.setattr(runner, 'evaluate', lambda *args, **kwargs: {
        'phenomenal': {'ok': True}, 'mechanism_probes': []})
    args = arguments(paths, tmp_path / 'unused')
    args.save_path = None
    args.save_dir = tmp_path / 'logs' / 'run'
    args.exp_name = 'codex-test-model'
    assert runner.main(args) == 0
    expected = (args.save_dir / args.exp_name / demo.task_name).resolve()
    assert Path(args.save_path) == expected
    assert (expected / 'performance.json').is_file()
