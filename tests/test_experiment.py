"""Offline checks for .env authentication, durable accounting and profile isolation."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import shlex
import shutil
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.request import Request, build_opener, ProxyHandler
import pytest

from src.algorithms.codex import _isolated_home, run
from src.export_problems import export_task

spec = importlib.util.spec_from_file_location('benchmark_experiment', Path(__file__).resolve().parents[1] / 'run/openrouter_experiment.py')
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)
PRICING = {'prompt': '0.00000006', 'completion': '0.00000012', 'input_cache_read': '0.000000012'}


def test_repository_env_overrides_an_inherited_wrong_key(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, 'ROOT', tmp_path)
    monkeypatch.setenv('OPENROUTER_API_KEY', 'wrong-shell-key')
    (tmp_path / '.env').write_text('export OPENROUTER_API_KEY="repository-key" # local key\n')
    assert experiment.repository_key() == 'repository-key'


def test_inflight_reservation_survives_controller_restart(tmp_path):
    first = experiment.Accounting(tmp_path, 1, PRICING)
    request_id = first.begin('task-a', 5000, 16000)
    snapshot = json.loads((tmp_path / 'usage.json').read_text())
    reserved = snapshot['reserved_inflight_usd']
    assert request_id == 1 and reserved > 0
    restarted = experiment.Accounting(tmp_path, 1, PRICING)
    assert restarted.cost == reserved
    assert restarted.estimated_cost == reserved
    assert not restarted.pending
    records = [json.loads(line) for line in (tmp_path / 'usage.jsonl').read_text().splitlines()]
    assert records[-1]['error'] == 'recovered_incomplete_request'
    assert records[-1]['cost_source'] == 'incomplete_request_upper_bound'


def test_concurrent_admission_cannot_reserve_beyond_budget(tmp_path):
    account = experiment.Accounting(tmp_path, 0.01, PRICING)
    with ThreadPoolExecutor(max_workers=10) as pool:
        accepted = list(pool.map(lambda _: account.begin('task-a', 5000, 16000), range(10)))
    assert any(i is not None for i in accepted)
    assert any(i is None for i in accepted)
    snapshot = json.loads((tmp_path / 'usage.json').read_text())
    assert snapshot['charged_usd'] + snapshot['reserved_inflight_usd'] <= 0.01
    assert snapshot['stop_reason'] == 'budget_limit'


def test_gateway_persists_provider_usage_and_uses_only_its_repository_key(tmp_path, monkeypatch):
    calls = []
    response = {'id': 'gen_offline', 'usage': {'input_tokens': 100, 'output_tokens': 200,
                 'cost': 0.0001, 'input_tokens_details': {'cached_tokens': 50}}}
    class Reply:
        status_code = 200
        headers = {'Content-Type': 'text/event-stream'}
        def iter_lines(self, **kwargs):
            yield b'event: response.completed'
            yield b'data: ' + json.dumps({'type': 'response.completed', 'response': response}).encode()
            yield b''
        def close(self): pass
    class Session:
        def post(self, url, **kwargs): calls.append((url, kwargs)); return Reply()
        def close(self): pass
    monkeypatch.setattr(experiment.requests, 'Session', Session)
    monkeypatch.setenv('OPENROUTER_API_KEY', 'wrong-inherited-key')
    account = experiment.Accounting(tmp_path, 1, PRICING)
    server = experiment.gateway(account, 'repository-key', 'deepseek/test', 2)
    try:
        request = Request(f'http://127.0.0.1:{server.server_port}/task-a/api/v1/responses',
            data=json.dumps({'model': 'deepseek/test', 'stream': True}).encode(),
            headers={'Authorization': 'Bearer local-experiment-only', 'Content-Type': 'application/json'})
        with build_opener(ProxyHandler({})).open(request, timeout=10) as reply: reply.read()
        assert calls[0][1]['headers']['Authorization'] == 'Bearer repository-key'
        usage = json.loads((tmp_path / 'usage.json').read_text())
        assert usage['provider_reported_usd'] == 0.0001
        assert usage['input_tokens'] == 100 and usage['output_tokens'] == 200
        assert usage['cached_tokens'] == 50 and usage['inflight'] == 0
        assert (tmp_path / 'api/000001/response.sse').is_file()
    finally:
        server.shutdown(); server.server_close()


def test_persistent_profile_keeps_provider_but_never_copies_account_auth(tmp_path, monkeypatch):
    original = tmp_path / 'original'; original.mkdir()
    (original / 'config.toml').write_text('''model_provider="openai"
[model_providers.openrouter]
name="OpenRouter"
base_url="https://openrouter.ai/api/v1"
wire_api="responses"
[model_providers.openrouter.auth]
command="sh"
args=["-c", "echo $OPENROUTER_API_KEY"]
''')
    (original / 'openrouter.config.toml').write_text('model_provider="openrouter"\nmodel="some-model"\n')
    (original / 'auth.json').write_text('{"secret":"account-auth-must-not-be-copied"}')
    monkeypatch.setenv('CODEX_HOME', str(original))
    destination = tmp_path / 'persistent'
    _isolated_home(destination, 'openrouter')
    assert not (destination / 'auth.json').exists()
    assert 'OPENROUTER_API_KEY' in (destination / 'openrouter.config.toml').read_text()
    assert 'account-auth-must-not-be-copied' not in (destination / 'config.toml').read_text()


def test_real_codex_tools_can_use_numpy_but_cannot_read_private_task_files(demo, tmp_path, monkeypatch):
    if not shutil.which('codex') or not shutil.which('bwrap'):
        pytest.skip('Codex and its Linux filesystem helper are required.')
    calls = []
    marker = experiment.ROOT / 'playground' / ('offline-private-' + tmp_path.name + '.txt')
    marker.write_text('PRIVATE_VALUE_MUST_NOT_BE_VISIBLE')
    script = f'''import numpy
print("NUMPY_OK")
from pathlib import Path
try: print(Path({str(marker)!r}).read_text())
except (PermissionError, FileNotFoundError): print("PRIVATE_BLOCKED")
'''
    arguments = json.dumps({'cmd': shlex.quote(sys.executable) + ' -c ' + shlex.quote(script), 'yield_time_ms': 10000})
    class Provider(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length']))); calls.append(body)
            if any(i.get('type') == 'function_call_output' for i in body.get('input', [])):
                item = {'id':'msg_final', 'type':'message', 'role':'assistant', 'status':'completed',
                        'content':[{'type':'output_text', 'text':'\n'.join(m.formula_str for m in demo.mechanism_model), 'annotations':[]}]}
            else:
                item = {'id':'fc_test', 'type':'function_call', 'call_id':'call_test',
                        'name':'exec_command', 'arguments':arguments, 'status':'completed'}
            response = {'id':'resp_offline', 'status':'completed', 'model':'gpt-5', 'output':[item],
                        'usage':{'input_tokens':10, 'output_tokens':10, 'total_tokens':20}}
            events = [{'type':'response.created', 'response':dict(response,status='in_progress',output=[])},
                      {'type':'response.output_item.added','output_index':0,'item':dict(item,status='in_progress')},
                      {'type':'response.output_item.done','output_index':0,'item':item},
                      {'type':'response.completed','response':response}]
            payload = ''.join('event: '+e['type']+'\ndata: '+json.dumps(e)+'\n\n' for e in events).encode()
            self.send_response(200); self.send_header('Content-Type','text/event-stream')
            self.send_header('Content-Length',str(len(payload))); self.end_headers(); self.wfile.write(payload)
    server = ThreadingHTTPServer(('127.0.0.1',0),Provider)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    original = tmp_path/'home'; original.mkdir()
    (original/'config.toml').write_text('[model_providers.openrouter]\nname="Offline"\nwire_api="responses"\n')
    (original/'openrouter.config.toml').write_text('model_provider="openrouter"\n')
    monkeypatch.setenv('CODEX_HOME',str(original)); monkeypatch.setenv('MDBENCH_LOCAL_GATEWAY_TOKEN','offline-key')
    paths = export_task(demo,tmp_path/'data',train_samples=4,id_test_samples=4,ood_test_samples=4)
    try:
        with tempfile.TemporaryDirectory(prefix='offline-permissions-',dir=experiment.ROOT/'logs/run') as directory:
            args = SimpleNamespace(save_path=directory,codex_model='gpt-5',codex_profile='openrouter',
                codex_provider_base_url=f'http://127.0.0.1:{server.server_port}/v1',
                codex_provider_env_key='MDBENCH_LOCAL_GATEWAY_TOKEN',persist_runtime=True,
                codex_private_root=str(experiment.ROOT),timeout=30)
            try:
                run(args,paths['problem'],paths['train'],'http://127.0.0.1:1/evaluate')
            except RuntimeError as exc:
                raise AssertionError((Path(directory)/'codex.stderr.txt').read_text()) from exc
            outputs = [i['output'] for call in calls for i in call.get('input',[]) if i.get('type')=='function_call_output']
            output = '\n'.join(map(str,outputs))
            assert 'NUMPY_OK' in output, output
            assert 'PRIVATE_BLOCKED' in output, output
            assert 'PRIVATE_VALUE_MUST_NOT_BE_VISIBLE' not in output
    finally:
        server.shutdown(); server.server_close(); marker.unlink()
