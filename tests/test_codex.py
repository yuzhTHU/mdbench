"""Codex subprocess tests, including a real CLI against an offline mock provider."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
import shutil
import sys
import threading
from types import SimpleNamespace
import pytest
import src.algorithms.codex as codex
from src.algorithms.codex import run, get_ask, resume, _invoke, clean_ansi
from src.export_problems import export_task
from src.evaluate import evaluate


def test_codex_command_accepts_machine_specific_global_options():
    args = SimpleNamespace(
        codex_command='custom-codex --profile lab -m provider/model',
        codex_text_only=False)
    command = codex._command(args)
    assert command[:7] == [
        'custom-codex', '--profile', 'lab', '-m', 'provider/model', 'exec', '--json']
    assert codex._selected_profile(args) == 'lab'
    assert codex._selected_model(args) == 'provider/model'
    with pytest.raises(ValueError, match='sandbox'):
        codex._codex_prefix(SimpleNamespace(
            codex_command='codex --sandbox danger-full-access'))


def test_agent_runtime_shadows_mdbench_and_denies_private_paths(tmp_path, caplog):
    root = tmp_path / 'temporary-runtime'; root.mkdir()
    workspace = root / 'workspace'; workspace.mkdir()
    blocked = codex._blocked_command_directory(root)
    answer = tmp_path / 'private' / 'answer.json'
    answer.parent.mkdir(); answer.write_text('{}')
    save = tmp_path / 'run-artifacts'; save.mkdir()
    args = SimpleNamespace(
        codex_command='codex', codex_text_only=False,
        save_path=save, answer=answer)
    with caplog.at_level(logging.INFO):
        command = codex._command(
            args, workspace=workspace, blocked_bin=blocked,
            allowed_network_hosts=('127.0.0.1',))
    configuration = '\n'.join(command)
    assert (blocked / 'mdbench').is_file()
    assert str(blocked) in configuration
    assert f'{save.resolve()}' in configuration
    assert f'{answer.parent.resolve()}' in configuration
    assert str(Path(codex.__file__).resolve().parents[1]) in configuration
    benchmark_root = Path(codex.__file__).resolve().parents[2]
    for private_entry in ('.github', 'LICENSE', 'proposal.md', 'README.md', 'run.py', 'tests'):
        assert str(benchmark_root / private_entry) in configuration
    assert f'"{benchmark_root / "venv"}" = "deny"' not in configuration
    assert f'"{benchmark_root / "third-party"}" = "deny"' not in configuration
    assert f'Codex project deny: {benchmark_root / "proposal.md"}' in caplog.text
    assert f'readable exceptions: {benchmark_root / "third-party"}, {benchmark_root / "venv"}' in caplog.text
    assert 'features.network_proxy=true' in configuration
    assert '127.0.0.1' in configuration
    assert 'allow_upstream_proxy' in configuration


def test_feedback_host_allowlist_accepts_only_loopback():
    assert codex._local_feedback_host('http://127.0.0.1:8123/evaluate') == '127.0.0.1'
    assert codex._local_feedback_host('http://localhost:8123/evaluate') == 'localhost'
    assert codex._local_feedback_host('http://[::1]:8123/evaluate') == '::1'
    with pytest.raises(ValueError, match='loopback'):
        codex._local_feedback_host('https://feedback.example.com/evaluate')


def test_codex_owns_openrouter_gateway_and_reads_environment(tmp_path, monkeypatch):
    secret = 'environment-only-upstream-key'
    monkeypatch.setenv('OPENROUTER_API_KEY', secret)

    class Response:
        def raise_for_status(self): pass
        def json(self):
            return {'data': [{'id': 'provider/model', 'pricing': {
                'prompt': '0.000001', 'completion': '0.000002'}}]}

    class Session:
        trust_env = True
        def get(self, *args, **kwargs): return Response()
        def close(self): pass

    class Gateway:
        server_port = 43210
        def shutdown(self): pass
        def server_close(self): pass

    captured = {}
    def start(account, key, model, parallel, **kwargs):
        captured.update(key=key, model=model, parallel=parallel, kwargs=kwargs)
        return Gateway()

    monkeypatch.setattr(codex.requests, 'Session', Session)
    monkeypatch.setattr(codex, 'start_responses_gateway', start)
    args = SimpleNamespace(
        openrouter_gateway=True, openrouter_budget_usd=1.25,
        codex_command='codex -m provider/model', probe_workers=3)
    try:
        codex._ensure_openrouter_gateway(args, tmp_path)
        assert captured['key'] == secret and captured['parallel'] == 3
        assert args._codex_copy_auth is False
        assert 'OPENROUTER_API_KEY' not in args._codex_child_env
        assert args._codex_isolated_config['model_provider'] == 'openrouter'
    finally:
        codex._shutdown_openrouter_gateways()


def test_timeout_and_ansi_artifact(tmp_path):
    status = _invoke([sys.executable, '-c', 'import time; time.sleep(30)'], '', tmp_path,
                     __import__('os').environ, tmp_path / 'events', tmp_path / 'stderr', .1)
    assert status['timed_out'] and status['elapsed_seconds'] < 10
    assert clean_ansi('\x1b[31merror\x1b[0m') == 'error'


def test_codex_defaults_to_long_run_and_unlimited_accounting():
    parser = argparse.ArgumentParser()
    codex.update_parser(parser)
    args = parser.parse_args([])
    assert args.timeout == 900
    assert args.openrouter_budget_usd is None


def test_real_cli_checkpoint_and_independent_probe_recovery(demo, tmp_path, monkeypatch):
    executable = shutil.which('codex')
    if not executable: pytest.skip('Codex CLI is not installed; other tests do not require it.')
    calls = []
    lock = threading.Lock()
    initial_text = '\n'.join(m.formula_str for m in demo.mechanism_model)
    class Provider(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            with lock: calls.append(request)
            user_texts = [item.get('content') for item in request.get('input', []) if item.get('role') == 'user']
            question = json.dumps(user_texts[-1]) if user_texts else ''
            text = initial_text
            for probe in demo.mechanism_probes:
                if f'derive {probe.probe} (' in question:
                    text = f'{probe.probe}={demo.solution[probe.probe]}'
                    break
            item = {'id': 'msg_test', 'type': 'message', 'role': 'assistant', 'status': 'completed',
                    'content': [{'type': 'output_text', 'text': text, 'annotations': []}]}
            response = {'id': 'resp_test', 'object': 'response', 'status': 'completed', 'model': 'gpt-5',
                        'output': [item], 'usage': {'input_tokens': 100, 'output_tokens': 50, 'total_tokens': 150,
                        'input_tokens_details': {'cached_tokens': 0}, 'output_tokens_details': {'reasoning_tokens': 0}}}
            messages = [
                {'type': 'response.created', 'response': dict(response, status='in_progress', output=[])},
                {'type': 'response.output_item.added', 'output_index': 0, 'item': dict(item, status='in_progress', content=[])},
                {'type': 'response.content_part.added', 'item_id': item['id'], 'output_index': 0, 'content_index': 0,
                 'part': {'type': 'output_text', 'text': '', 'annotations': []}},
                {'type': 'response.output_text.delta', 'item_id': item['id'], 'output_index': 0, 'content_index': 0, 'delta': text},
                {'type': 'response.output_item.done', 'output_index': 0, 'item': item},
                {'type': 'response.completed', 'response': response}]
            payload = ''.join('event: ' + m['type'] + '\ndata: ' + json.dumps(m) + '\n\n' for m in messages).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        def log_message(self, *args): pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Provider)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    home = tmp_path / 'source-home'
    home.mkdir()
    (home / 'config.toml').write_text(f'''model = "gpt-5"
model_provider = "offline"
[model_providers.offline]
name = "Offline test"
base_url = "http://127.0.0.1:{server.server_port}/v1"
wire_api = "responses"
requires_openai_auth = false
''')
    monkeypatch.setenv('CODEX_HOME', str(home))
    paths = export_task(demo, tmp_path / 'task', train_samples=8, id_test_samples=4, ood_test_samples=4)
    args = SimpleNamespace(save_path=tmp_path / 'run', codex_bin=executable, codex_model='gpt-5',
                           timeout=30, probe_timeout=30, probe_workers=2, algorithm='codex')
    try:
        submission, model = run(args, paths['problem'], paths['train'], 'http://127.0.0.1:1/evaluate')
        save = Path(args.save_path)
        assert {p.name for p in save.iterdir()} == {
            'audit', 'saved_checkpoint', 'prompt.txt'}
        assert {p.name for p in (save / 'audit').iterdir()} >= {
            'codex.events.jsonl', 'stderr.txt', 'stdout.txt', 'last_message.txt', 'process.json'}
        assert {p.name for p in (save / 'saved_checkpoint').iterdir()} == {
            'codex.session.jsonl', 'model.json'}
        checkpoint = (save / 'saved_checkpoint' / 'codex.session.jsonl').read_bytes()
        result = evaluate(args, paths['answer'], submission,
                          lambda: get_ask(args, deepcopy(model)))
        assert result['phenomenal']['train']['numerically_equivalent']
        assert all(p['ok'] for p in result['mechanism_probes']), result
        assert result['mechanism_recovery']['ood_test']['numerically_equivalent'] == 1
        assert (save / 'saved_checkpoint' / 'codex.session.jsonl').read_bytes() == checkpoint
        probe_dirs = list((save / 'probe').iterdir())
        assert len(probe_dirs) == len(demo.mechanism_probes)
        for probe_dir in probe_dirs:
            assert {'prompt.txt', 'reply.txt', 'audit'} <= {p.name for p in probe_dir.iterdir()}
            assert {'codex_events.jsonl', 'process.json', 'stderr.txt', 'stdout.txt', 'status.json'} <= {
                p.name for p in (probe_dir / 'audit').iterdir()}
        assert len(calls) == 3
        for call in calls[1:]:
            context = json.dumps(call['input'])
            assert sum(f'derive {p.probe} (' in context for p in demo.mechanism_probes) == 1
            names = {tool.get('name', tool.get('type')) for tool in call.get('tools', [])}
            assert not names & {'shell', 'shell_command', 'exec_command', 'web_search', 'spawn_agent'}
    finally:
        server.shutdown(); server.server_close(); thread.join()
