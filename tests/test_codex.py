"""Codex subprocess tests, including a real CLI against an offline mock provider."""
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import sys
import threading
from types import SimpleNamespace
import pytest
from src.algorithms.codex import run, resume, _invoke, clean_ansi
from src.export_problems import export_task
from src.evaluate import evaluate


def test_timeout_and_ansi_artifact(tmp_path):
    status = _invoke([sys.executable, '-c', 'import time; time.sleep(30)'], '', tmp_path,
                     __import__('os').environ, tmp_path / 'events', tmp_path / 'stderr', .1)
    assert status['timed_out'] and status['elapsed_seconds'] < 10
    assert clean_ansi('\x1b[31merror\x1b[0m') == 'error'


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
        submission, ask = run(args, paths['problem'], paths['train'], 'http://127.0.0.1:1/evaluate')
        save = Path(args.save_path)
        assert {p.name for p in save.iterdir()} == {
            'audit', 'saved_checkpoint', 'prompt.txt'}
        assert {p.name for p in (save / 'audit').iterdir()} >= {
            'codex.events.jsonl', 'stderr.txt', 'stdout.txt', 'last_message.txt', 'process.json'}
        assert {p.name for p in (save / 'saved_checkpoint').iterdir()} == {
            'codex.session.jsonl', 'model.json'}
        checkpoint = (save / 'saved_checkpoint' / 'codex.session.jsonl').read_bytes()
        result = evaluate(args, paths['answer'], submission, ask)
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
