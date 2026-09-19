"""Codex CLI baseline with persisted end-of-run checkpoints.

The evaluator requests a fresh callable for every probe. Each callable restores
the exact saved rollout into a fresh CODEX_HOME, so no probe turns are appended
to the original checkpoint or seen by another probe.
"""
from __future__ import annotations
import argparse
import atexit
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import shlex
import threading
import uuid
import requests
import hashlib
import ipaddress
import json
import logging
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import tomllib
from typing import Any, Callable
from urllib.parse import urlsplit

__all__ = ["update_parser", "run", "get_ask"]
ANSI = re.compile(r'\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))')
_logger = logging.getLogger(__name__)
_OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1'
_gateway_lock = threading.RLock()
_gateways = {}


def update_parser(parser):
    """ Add Algorithm-specific arguments to the parser. """
    parser.add_argument('--timeout', type=float, default=900, help='Agent run time limit in seconds.')
    parser.add_argument('--codex_bin', default='codex')
    parser.add_argument('--codex_command', default=None,
                        help='Codex executable plus global options, e.g. "codex --profile lab -m provider/model".')
    parser.add_argument('--codex_model', default=None)
    parser.add_argument('--codex_text_only', action='store_true', help='Disable tools that can add unsupported image input to text-only models.')
    # Kept as hidden compatibility hooks for older experiment manifests. New
    # users should select their local installation with --codex-command.
    parser.add_argument('--codex_provider_base_url', default=None, help=argparse.SUPPRESS)
    parser.add_argument('--codex_provider_env_key', default=None, help=argparse.SUPPRESS)
    parser.add_argument('--codex_reasoning_effort', default=None)
    parser.add_argument('--codex_blocked_tools', default=None)
    parser.add_argument('--codex_tool_path', default=None)
    parser.add_argument('--openrouter_gateway', action='store_true',
                        help='Route Codex through a local metered OpenRouter gateway.')
    parser.add_argument('--openrouter_budget_usd', type=float, default=None,
                        help='Optional positive request budget. Omit to record usage without enforcing a spending limit.')
    return parser


def _codex_prefix(args):
    configured = getattr(args, 'codex_command', None)
    command = shlex.split(configured) if configured else [getattr(args, 'codex_bin', 'codex')]
    if not command:
        raise ValueError('--codex-command must contain an executable.')
    forbidden = {'--sandbox', '-s', '--dangerously-bypass-approvals-and-sandbox',
                 '--yolo', '--full-auto'}
    if any(token in forbidden or token.startswith('--sandbox=') for token in command[1:]):
        raise ValueError('--codex-command may not override the benchmark sandbox.')
    protected = ('sandbox_mode', 'sandbox_workspace_write', 'default_permissions',
                 'permissions.', 'features.network_proxy')
    for index, token in enumerate(command[:-1]):
        if token in ('-c', '--config') and command[index + 1].lstrip().startswith(protected):
            raise ValueError('--codex-command may not override benchmark permissions.')
    return command


def _command_option(command, *names):
    for index, token in enumerate(command):
        for name in names:
            if token == name:
                if index + 1 >= len(command):
                    raise ValueError(f'{name} in --codex-command requires a value.')
                return command[index + 1]
            if name.startswith('--') and token.startswith(name + '='):
                return token.split('=', 1)[1]
    return None


def _selected_profile(args):
    return _command_option(_codex_prefix(args), '--profile', '-p')


def _user_configuration(profile=None):
    home = Path(os.environ.get('CODEX_HOME', Path.home() / '.codex'))
    config_file = home / 'config.toml'
    config = tomllib.loads(config_file.read_text()) if config_file.is_file() else {}
    selected = {}
    if profile:
        legacy = home / f'{profile}.config.toml'
        if legacy.is_file():
            selected = tomllib.loads(legacy.read_text())
        elif isinstance(config.get('profiles', {}).get(profile), dict):
            selected = dict(config['profiles'][profile])
    return home, config, selected


def _selected_model(args):
    explicit = (_command_option(_codex_prefix(args), '--model', '-m')
                or getattr(args, 'codex_model', None))
    if explicit:
        return explicit
    isolated = getattr(args, '_codex_isolated_config', None) or {}
    if isolated.get('model'):
        return isolated['model']
    _, config, selected = _user_configuration(_selected_profile(args))
    return selected.get('model') or config.get('model')


def _blocked_command_directory(root):
    """Shadow benchmark administration commands inside the agent environment."""
    directory = Path(root) / 'blocked-bin'
    directory.mkdir()
    command = directory / 'mdbench'
    command.write_text(
        '#!/bin/sh\n'
        'echo "mdbench is unavailable inside the benchmark agent" >&2\n'
        'exit 126\n')
    command.chmod(0o755)
    return directory


def _local_feedback_host(url):
    """Return one exact loopback allowlist entry or reject the endpoint."""
    parts = urlsplit(url)
    host = (parts.hostname or '').rstrip('.').lower()
    if parts.scheme not in ('http', 'https') or not host:
        raise ValueError('Feedback URL must be an HTTP(S) URL with a hostname.')
    if host == 'localhost':
        return host
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise ValueError('Codex feedback URL must use localhost or a loopback IP address.') from exc
    if not address.is_loopback:
        raise ValueError('Codex feedback URL must use localhost or a loopback IP address.')
    return host



def run(args, problem_file: Path, train_data_npy_file: Path,
        feedback_server_url) -> tuple[list[str], Any]:
    save = Path(args.save_path).resolve()
    save.mkdir(parents=True, exist_ok=True)
    _ensure_openrouter_gateway(args, save)
    audit, saved_checkpoint = save / 'audit', save / 'saved_checkpoint'
    audit.mkdir(exist_ok=True); saved_checkpoint.mkdir(exist_ok=True)
    events, errors = audit / 'codex.events.jsonl', audit / 'stderr.txt'
    stdout, process_log = audit / 'stdout.txt', audit / 'process.json'
    checkpoint = saved_checkpoint / 'codex.session.jsonl'
    feedback_host = _local_feedback_host(feedback_server_url)
    with tempfile.TemporaryDirectory(prefix='mdbench-codex-') as directory:
        root = Path(directory)
        workspace = root / 'workspace'
        workspace.mkdir(parents=True)
        blocked_bin = _blocked_command_directory(root)
        shutil.copy2(problem_file, workspace / 'problem.json')
        shutil.copy2(train_data_npy_file, workspace / 'train.npy')
        env = _isolated_home(root / 'home', _selected_profile(args),
                             base_env=getattr(args, '_codex_child_env', None),
                             copy_auth=getattr(args, '_codex_copy_auth', True),
                             isolated_config=getattr(args, '_codex_isolated_config', None))
        env['PATH'] = str(blocked_bin) + os.pathsep + env.get('PATH', '')
        prompt = f'''Discover a scientific mechanism from the observations in problem.json and train.npy.
Only observed variables are provided. The NPY array has shape (variables, samples);
row names and their scientific meanings are in problem.json data_columns/variables.
Scientific Python is available at {sys.executable}; use it for numpy/sympy analysis.
{('Use textual/numerical tables for analysis; image input is unavailable in this experiment.' if getattr(args, 'codex_text_only', False) else 'Use numerical tables for analysis.')}
Propose a set of algebraic equations involving those inputs, the target and any
scientifically meaningful unobserved internal states needed to explain the data.
No separate symbolic constants: define their numerical values as equations.
Use only ordinary algebra, ^ or ** powers, sqrt, exp, log, trigonometric functions,
and abs. No differential equations. Your equations must uniquely solve the target
and every internal state as explicit expressions in the supplied inputs.
Write submission.txt now, then improve it within {args.timeout} seconds. Keep the
file updated so it is available if the time limit interrupts you. One equality
per line, no markdown or prose. Do not change problem.json or train.npy.
You can obtain objective train accuracy feedback by POST to {feedback_server_url}.
Keep the environment-provided HTTP proxy enabled: it is the sandbox's controlled
route to this local feedback endpoint. Do not set trust_env=False or override it.
Upload three multipart file fields: problem (problem.json), train_data (train.npy),
submission (submission.txt). For example use Python session.post(url, files={{
"problem": open("problem.json", "rb"), "train_data": open("train.npy", "rb"),
"submission": open("submission.txt", "rb")}}).json().
Internal predictions may be queried later. Do not use external reference answers.
Do not run git commands. Do not read shell startup files or credentials.
The mdbench command and benchmark package are intentionally unavailable; use
only the feedback endpoint described above.
Do not inspect files outside this workspace, except the provided Python environment.
End with the same equations as your submission, so the final conversation records it.
'''
        (save / 'prompt.txt').write_text(prompt)
        final = workspace / 'last_message.txt'
        status = _invoke(_command(args, workspace=workspace, blocked_bin=blocked_bin,
                                  allowed_network_hosts=(feedback_host,))
                         + ['-o', str(final), '-'],
                         prompt, workspace, env, events, errors, args.timeout,
                         stdout=stdout, process_log=process_log)
        model = {'algorithm': 'codex', 'events': str(events),
                 'session': str(checkpoint), 'codex_model': _selected_model(args), **status}
        model['codex_profile'] = _selected_profile(args)
        model['codex_command'] = getattr(args, 'codex_command', None)
        submission = None
        if (workspace / 'submission.txt').is_file():
            candidate = clean_ansi((workspace / 'submission.txt').read_text())
            if candidate.strip(): submission = candidate
        if final.is_file():
            text = clean_ansi(final.read_text())
            (audit / 'last_message.txt').write_text(text)
            if submission is None:
                from ..scoring import submission_formulas
                try:
                    submission_formulas(text)
                except ValueError:
                    pass
                else:
                    submission = text
        try:
            thread_id = _thread_id(events)
            sessions = list((root / 'home' / 'sessions').rglob(f'*{thread_id}*.jsonl'))
            if len(sessions) != 1: raise RuntimeError('Expected exactly one persisted Codex rollout.')
            shutil.copy2(sessions[0], checkpoint)
            model['session_id'] = thread_id
            model['rollout_filename'] = sessions[0].name
        except (ValueError, RuntimeError) as exc:
            model['error'] = str(exc)
        (saved_checkpoint / 'model.json').write_text(json.dumps(model, indent=2) + '\n')
        if submission is None or not checkpoint.is_file():
            raise RuntimeError('Codex did not produce both submission and resumable checkpoint; see saved model/events/stderr.')
        # Freeze the saved checkpoint; future probe turns only ever modify copies.
        checkpoint.chmod(0o444)
        from ..scoring import submission_formulas
        return submission_formulas(submission), model



def clean_ansi(text):
    return ANSI.sub('', text)


def utc(): return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False, default=str)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temporary, path)


def load_api_key(env_file: Path, key_name: str):
    """Read one literal key assignment from an explicit file, never shell state."""
    values = []
    for line in Path(env_file).read_text().splitlines():
        line = line.strip().removeprefix('export ')
        if '=' in line and line.split('=', 1)[0].strip() == key_name:
            pieces = shlex.split(line.split('=', 1)[1], comments=True)
            if len(pieces) != 1: raise ValueError(f'Invalid {key_name} assignment in credential file.')
            values.append(pieces[0])
    if len(values) != 1 or not values[0]: raise ValueError(f'Require exactly one nonempty {key_name} assignment in credential file.')
    return values[0]


class UsageAccounting:
    """Fsync request accounting, optionally enforce a budget, and recover interrupted calls.

    Pricing values are USD per token, using prompt/completion/input_cache_read.
    One controller owns a ledger; its request threads share this instance.
    """
    def __init__(self, root, budget, pricing):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        self.root, self.budget, self.pricing = root, budget, pricing
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.reason = None
        self.counter = 0
        self.pending = {}
        self.cost = self.actual_cost = self.estimated_cost = 0.0
        self.input_tokens = self.output_tokens = self.cached_tokens = 0
        self.errors = []
        self.task_cost = {}
        self.started = time.monotonic()
        ledger = root / 'usage.jsonl'
        if ledger.exists():
            for line in ledger.read_text().splitlines():
                try: record = json.loads(line)
                except json.JSONDecodeError: continue
                self.counter = max(self.counter, record.get('request_id', 0))
                if record['event'] == 'request_started': self.pending[record['request_id']] = record
                if record['event'] == 'request_finished':
                    self.pending.pop(record['request_id'], None)
                    self._totals(record)
            # A killed controller cannot know the final charge for abandoned
            # requests. Retain their conservative upper bound, never forget them.
            for request_id, record in list(self.pending.items()):
                self.finish(request_id, {}, None, 'recovered_incomplete_request')
        self.snapshot()

    def _totals(self, record):
        charge = record['charged_usd']
        self.cost += charge
        if record['cost_source'] == 'provider': self.actual_cost += charge
        else: self.estimated_cost += charge
        self.input_tokens += record.get('input_tokens', 0)
        self.output_tokens += record.get('output_tokens', 0)
        self.cached_tokens += record.get('cached_tokens', 0)
        self.task_cost[record['task_id']] = self.task_cost.get(record['task_id'], 0) + charge

    def record(self, event, **fields):
        with self.lock:
            record = dict(time=utc(), event=event, **fields)
            with (self.root / 'usage.jsonl').open('a') as f:
                f.write(json.dumps(record, allow_nan=False) + '\n'); f.flush(); os.fsync(f.fileno())
            return record

    def begin(self, task_id, size, maximum):
        with self.lock:
            reserve = (size + 4096) * float(self.pricing['prompt']) + maximum * float(self.pricing['completion'])
            if self.stop.is_set(): return None
            if (self.budget is not None
                    and self.cost + sum(r['reserved_usd'] for r in self.pending.values()) + reserve > self.budget):
                self.trip('budget_limit')
                return None
            self.counter += 1
            record = self.record('request_started', request_id=self.counter, task_id=task_id,
                                 reserved_usd=reserve, request_bytes=size, max_output_tokens=maximum)
            self.pending[self.counter] = record
            self.snapshot()
            return self.counter

    def finish(self, request_id, usage, response_id, error=None):
        with self.lock:
            pending = self.pending.pop(request_id)
            prompt = usage.get('input_tokens', usage.get('prompt_tokens', 0)) or 0
            output = usage.get('output_tokens', usage.get('completion_tokens', 0)) or 0
            cached = (usage.get('input_tokens_details') or usage.get('prompt_tokens_details') or {}).get('cached_tokens', 0) or 0
            charge = usage.get('cost')
            source = 'provider'
            if charge is None:
                source = 'pricing_estimate' if usage else 'incomplete_request_upper_bound'
                charge = (max(0, prompt - cached) * float(self.pricing['prompt'])
                          + cached * float(self.pricing.get('input_cache_read', self.pricing['prompt']))
                          + output * float(self.pricing['completion'])) if usage else pending['reserved_usd']
            record = self.record('request_finished', request_id=request_id, task_id=pending['task_id'],
                                 response_id=response_id, input_tokens=prompt, output_tokens=output,
                                 cached_tokens=cached, charged_usd=float(charge), cost_source=source,
                                 usage=usage, error=error)
            self._totals(record); self.snapshot()

    def trip(self, reason):
        with self.lock:
            if not self.stop.is_set():
                self.record('circuit_breaker', reason=reason)
                self.reason = reason; self.stop.set()
            self.snapshot()

    def snapshot(self):
        with self.lock:
            atomic_json(self.root / 'usage.json', dict(time=utc(), charged_usd=self.cost,
                provider_reported_usd=self.actual_cost, estimated_or_uncertain_usd=self.estimated_cost,
                reserved_inflight_usd=sum(r['reserved_usd'] for r in self.pending.values()),
                budget_usd=self.budget, input_tokens=self.input_tokens, output_tokens=self.output_tokens,
                cached_tokens=self.cached_tokens, request_count=self.counter,
                inflight=len(self.pending), task_cost_usd=self.task_cost,
                controller_seconds=time.monotonic()-self.started, stopped=self.stop.is_set(), stop_reason=self.reason))


def start_responses_gateway(account, key, model, parallel, *, upstream_base_url,
                            local_token='local-experiment-only', max_output_tokens=16000,
                            retry_attempts=4):
    """Start a loopback Responses gateway; caller must shutdown/server_close it.

    Clients use /<task-id>/api/v1 as their base URL. Only the gateway holds the
    upstream credential. Requests, streams and usage persist under account.root.
    """
    if parallel < 1 or max_output_tokens < 1 or retry_attempts < 1:
        raise ValueError('Gateway concurrency, token limit and retry count must be positive.')
    upstream_base_url = upstream_base_url.rstrip('/')
    slots = threading.BoundedSemaphore(parallel)
    failures_lock = threading.Lock()
    failure_streak = 0
    def infrastructure_outcome(failed):
        nonlocal failure_streak
        with failures_lock:
            failure_streak = failure_streak + 1 if failed else 0
            if failure_streak >= 3: account.trip("three_consecutive_gateway_errors")
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass

        def failure(self, code, message):
            body = json.dumps({'error': {'message': message, 'type': 'codex_gateway'}}).encode()
            self.send_response(code); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body))); self.end_headers()
            try: self.wfile.write(body)
            except OSError: pass

        def do_POST(self):
            match = re.fullmatch(r'/([^/]+)/api/v1/(responses(?:/compact)?)', self.path)
            if not match: return self.failure(404, 'Unsupported gateway route')
            if self.headers.get('Authorization') != 'Bearer ' + local_token: return self.failure(401, 'Local credential required')
            task_id, route = match.groups()
            try: size = int(self.headers.get('Content-Length', 0))
            except ValueError: return self.failure(400, 'Invalid Content-Length')
            if not 0 < size <= 32*1024*1024: return self.failure(413, 'Request size invalid')
            try:
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict): raise ValueError()
            except (ValueError, UnicodeDecodeError):
                return self.failure(400, 'Expected a JSON object')
            if body.get('model') != model: return self.failure(400, 'Model substitution is forbidden')
            maximum = body.get('max_output_tokens', max_output_tokens)
            if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
                return self.failure(400, 'max_output_tokens must be a positive integer')
            maximum = min(maximum, max_output_tokens)
            body['max_output_tokens'] = maximum
            slots.acquire()
            request_id = account.begin(task_id, size, maximum)
            if request_id is None:
                slots.release(); return self.failure(402, 'Experiment budget or circuit breaker reached')
            directory = account.root / 'api' / f'{request_id:06d}'
            directory.mkdir(parents=True)
            atomic_json(directory / 'request.json', body)
            session = requests.Session(); session.trust_env = False
            usage = {}; response_id = None; error = None; upstream = None; headers_sent = False
            try:
                for attempt in range(retry_attempts):
                    upstream = session.post(upstream_base_url + '/' + route,
                        headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json',
                                 'X-Title': 'MechanismDiscoveryBench Codex'},
                        json=body, stream=True, timeout=(20, 180))
                    atomic_json(directory / 'http.json', dict(status=upstream.status_code, attempt=attempt,
                        headers={k:v for k,v in upstream.headers.items() if k.lower() in ('content-type', 'x-request-id', 'retry-after')}))
                    if upstream.status_code != 429: break
                    account.record('http_error', request_id=request_id, status=429, attempt=attempt)
                    upstream.close(); time.sleep(min(30, 2**(attempt+1)))
                if upstream.status_code >= 400:
                    text = upstream.text.replace(key, '[REDACTED]')
                    (directory / 'error.txt').write_text(text)
                    account.record('http_error', request_id=request_id, status=upstream.status_code, message=text[:2000])
                    error = 'upstream_http_' + str(upstream.status_code)
                    if upstream.status_code in (401, 402, 403): account.trip(error)
                    if upstream.status_code >= 500: infrastructure_outcome(True)
                    self.failure(upstream.status_code, text[:2000])
                    # Rejected requests have no inference charge.
                    usage = {'input_tokens': 0, 'output_tokens': 0, 'cost': 0}
                    return
                self.send_response(upstream.status_code)
                self.send_header('Content-Type', upstream.headers.get('Content-Type', 'text/event-stream'))
                self.send_header('Connection', 'close'); self.end_headers(); headers_sent = True
                disconnected = False
                with (directory / 'response.sse').open('wb') as raw:
                    for line in upstream.iter_lines(chunk_size=1024):
                        data = line + b'\n'; raw.write(data); raw.flush()
                        if line.startswith(b'data: '):
                            try:
                                event = json.loads(line[6:])
                                response = event.get('response', {})
                                if response.get('id') and not response_id:
                                    response_id = response['id']
                                    account.record('response_started', request_id=request_id, response_id=response_id)
                                if response.get('usage'): usage = response['usage']
                                if event.get('type') in ('error', 'response.failed'):
                                    account.record('stream_error', request_id=request_id, detail=event)
                                    error = str(event.get('error') or response.get('error'))
                            except json.JSONDecodeError: pass
                        if not disconnected:
                            try: self.wfile.write(data); self.wfile.flush()
                            except OSError: disconnected = True
                    os.fsync(raw.fileno())
                infrastructure_outcome(bool(error))
            except Exception as exc:
                error = str(exc).replace(key, '[REDACTED]')
                account.record('transport_error', request_id=request_id, error=error)
                infrastructure_outcome(True)
                if not headers_sent: self.failure(502, error)
            finally:
                if upstream is not None: upstream.close()
                session.close(); account.finish(request_id, usage, response_id, error); slots.release()
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def probe_gateway_capacity(base_url, model, parallel, *, local_token='local-experiment-only'):
    """Make short concurrent requests through the same metered gateway."""
    if parallel < 1: raise ValueError('Probe concurrency must be positive.')
    def capacity(index):
        client = requests.Session(); client.trust_env = False
        started = time.monotonic()
        try:
            reply = client.post(f'{base_url.rstrip("/")}/capacity-{index}/api/v1/responses',
                headers={'Authorization': 'Bearer ' + local_token},
                json={'model': model, 'stream': True, 'max_output_tokens': 64,
                      'input': 'Reply with OK.', 'reasoning': {'effort': 'low'}}, timeout=120)
            payload = reply.content
            return dict(index=index, status=reply.status_code,
                        seconds=time.monotonic()-started, response_bytes=len(payload))
        except Exception as exc:
            return dict(index=index, status=None, seconds=time.monotonic()-started,
                        error=clean_ansi(str(exc)).replace(local_token, '[REDACTED]'))
        finally: client.close()
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        return list(pool.map(capacity, range(parallel)))


def gateway_environment(env_key, local_token, *, path_prefix=()):
    """Supply a local credential without forwarding inherited provider secrets."""
    env = {k: v for k, v in os.environ.items()
           if not re.search(r'(API_KEY|SECRET|TOKEN|PASSWORD)', k, re.I)}
    env[env_key] = local_token
    env['PATH'] = ':'.join([*(str(p) for p in path_prefix), env.get('PATH', '')])
    env['SHELL'] = '/bin/bash'
    env.pop('BASH_ENV', None); env.pop('ENV', None)
    env['NO_PROXY'] = env['no_proxy'] = 'localhost,127.0.0.1,::1'
    return env


def _ensure_openrouter_gateway(args, save):
    """Start the Codex-owned gateway once; keep it alive through probe turns."""
    if not getattr(args, 'openrouter_gateway', False):
        return
    budget = getattr(args, 'openrouter_budget_usd', None)
    if budget is not None and budget <= 0:
        raise ValueError('--openrouter-budget-usd must be positive when provided.')
    key = os.environ.get('OPENROUTER_API_KEY') or os.environ.get('OPENROUTER_API')
    if not key:
        raise ValueError('Set OPENROUTER_API_KEY in the environment to use --openrouter-gateway.')
    model = _selected_model(args)
    if not model:
        raise ValueError(
            'OpenRouter gateway could not determine the model; select it in '
            '--codex-command, --codex-model, or the active Codex profile.')
    identity = str(Path(save).resolve())
    with _gateway_lock:
        if identity in _gateways:
            for name, value in _gateways[identity]['settings'].items():
                setattr(args, name, value)
            return
        if _gateways:
            raise RuntimeError('The Codex OpenRouter runtime supports one benchmark task per process.')
        session = requests.Session(); session.trust_env = False
        try:
            response = session.get(
                _OPENROUTER_BASE_URL + '/models',
                headers={'Authorization': 'Bearer ' + key}, timeout=30)
            response.raise_for_status()
            metadata = next((item for item in response.json().get('data', [])
                             if item.get('id') == model), None)
        finally:
            session.close()
        if metadata is None:
            raise ValueError(f'OpenRouter model metadata not found: {model}')
        pricing = metadata.get('pricing')
        if not isinstance(pricing, dict) or not all(name in pricing for name in ('prompt', 'completion')):
            raise ValueError(f'OpenRouter model has incomplete pricing metadata: {model}')
        try:
            prices = [float(pricing[name]) for name in ('prompt', 'completion')]
        except (TypeError, ValueError) as exc:
            raise ValueError(f'OpenRouter model has invalid pricing metadata: {model}') from exc
        if any(not math.isfinite(price) or price < 0 for price in prices):
            raise ValueError(f'OpenRouter model has invalid pricing metadata: {model}')
        root = Path(save) / 'openrouter'
        root.mkdir(parents=True, exist_ok=True)
        atomic_json(root / 'model_metadata.json', metadata)
        account = UsageAccounting(root, budget, pricing)
        local_key = 'MDBENCH_LOCAL_GATEWAY_TOKEN'
        local_token = secrets.token_urlsafe(32)
        parallel = max(1, int(getattr(args, 'probe_workers', 1)))
        gateway = start_responses_gateway(
            account, key, model, parallel, upstream_base_url=_OPENROUTER_BASE_URL,
            local_token=local_token, max_output_tokens=16000, retry_attempts=4)
        local_url = f'http://127.0.0.1:{gateway.server_port}/single-task/api/v1'
        settings = {
            'codex_provider_base_url': local_url,
            'codex_provider_env_key': local_key,
            '_codex_child_env': gateway_environment(local_key, local_token),
            '_codex_copy_auth': False,
            '_codex_isolated_config': {
                'model': model,
                'model_provider': 'openrouter',
                'model_providers': {'openrouter': {
                    'name': 'MDBench local OpenRouter gateway', 'base_url': local_url,
                    'env_key': local_key, 'wire_api': 'responses'}},
            },
        }
        if effort := getattr(args, 'codex_reasoning_effort', None):
            settings['_codex_isolated_config']['model_reasoning_effort'] = effort
        for name, value in settings.items():
            setattr(args, name, value)
        _gateways[identity] = {
            'server': gateway, 'account': account, 'settings': settings}
        _logger.info(
            'Codex OpenRouter gateway enabled for one task: model=%s '
            'budget_usd=%s max_in_flight=%s', model, budget, parallel)


def _shutdown_openrouter_gateways():
    with _gateway_lock:
        runtimes = list(_gateways.values())
        _gateways.clear()
    for runtime in runtimes:
        gateway, account = runtime['server'], runtime['account']
        gateway.shutdown(); gateway.server_close(); account.snapshot()
        _logger.info(
            'Codex OpenRouter usage: charged_usd=%.9f provider_reported_usd=%.9f '
            'estimated_or_uncertain_usd=%.9f requests=%s input_tokens=%s '
            'cached_tokens=%s output_tokens=%s stopped=%s stop_reason=%s',
            account.cost, account.actual_cost, account.estimated_cost,
            account.counter, account.input_tokens, account.cached_tokens,
            account.output_tokens, account.stop.is_set(), account.reason)


atexit.register(_shutdown_openrouter_gateways)


def stop_codex_processes(destination):
    """Interrupt saved process groups only when their Linux PID identity matches."""
    for file in Path(destination).rglob('process.json'):
        try:
            record = json.loads(file.read_text()); pid = record['pid']
            if record.get('start_ticks') != Path(f'/proc/{pid}/stat').read_text().split()[21]: continue
            if os.getpgid(pid) != pid: continue
            os.killpg(pid, signal.SIGINT)
        except (OSError, ValueError, KeyError, IndexError): pass


def is_infrastructure_failure(performance, destination):
    """A time-budget failure with a saved checkpoint is an algorithm failure."""
    destination = Path(destination)
    operational_error = 'error' in performance and not (destination / 'submission.txt').exists()
    model_path = destination / 'saved_checkpoint' / 'model.json'
    model = json.loads(model_path.read_text()) if model_path.exists() else {}
    normal_timeout = model.get('timed_out') and (destination / 'saved_checkpoint' / 'codex.session.jsonl').exists()
    return bool(operational_error and not normal_timeout)


def _toml(value):
    if isinstance(value, bool): return 'true' if value else 'false'
    if isinstance(value, str): return json.dumps(value)
    if isinstance(value, (int, float)): return str(value)
    if isinstance(value, list): return '[' + ', '.join(_toml(v) for v in value) + ']'
    if isinstance(value, dict): return '{' + ', '.join(f'{json.dumps(k)} = {_toml(v)}' for k, v in value.items()) + '}'
    raise TypeError('Unsupported Codex provider configuration value.')


def _isolated_home(home: Path, profile=None, *, base_env=None, copy_auth=None,
                   isolated_config=None):
    home.mkdir(parents=True, exist_ok=True)
    if copy_auth is None:
        copy_auth = not profile  # Preserve the historical helper default.
    original = Path(os.environ.get('CODEX_HOME', Path.home() / '.codex'))
    auth = original / 'auth.json'
    if auth.is_file() and copy_auth:
        shutil.copy2(auth, home / 'auth.json')
        (home / 'auth.json').chmod(0o600)
    # Preserve account/provider configuration, but not user instructions, MCP,
    # plugins, skills or hooks. Credentials are never written into run artifacts.
    config = {}
    if isolated_config is not None:
        # A gateway-provided configuration is already credential-free and
        # self-contained; do not inspect or copy the user's Codex config.
        material = dict(isolated_config)
        text = '\n'.join(f'{k} = {_toml(v)}' for k, v in material.items())
        (home / 'config.toml').write_text(text)
        if profile:
            if not re.fullmatch(r'[A-Za-z0-9_-]+', profile):
                raise ValueError('Invalid Codex profile name.')
            (home / f'{profile}.config.toml').write_text(text)
    else:
        config_file = original / 'config.toml'
        if config_file.is_file():
            config = tomllib.loads(config_file.read_text())
            keys = ('model', 'model_provider', 'model_providers', 'model_reasoning_effort',
                    'chatgpt_base_url', 'openai_base_url')
            (home / 'config.toml').write_text('\n'.join(f'{k} = {_toml(config[k])}' for k in keys if k in config))
    if profile and isolated_config is None:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', profile): raise ValueError('Invalid Codex profile name.')
        _, config, selected = _user_configuration(profile)
        if not selected:
            raise ValueError(f'Codex profile not found: {profile}')
        provider = selected.get('model_provider', config.get('model_provider'))
        profile_definition = selected.get('model_providers', {})
        selected = {k: selected[k] for k in ('model', 'model_provider', 'model_reasoning_effort') if k in selected}
        if provider:
            definition = dict(config.get('model_providers', {}).get(provider, {}))
            definition.update(config.get('profiles', {}).get(profile, {}).get(
                'model_providers', {}).get(provider, {}))
            definition.update(profile_definition.get(provider, {}))
            safe = {k: definition[k] for k in ('name', 'base_url', 'wire_api', 'env_key',
                                              'request_max_retries', 'stream_max_retries',
                                              'stream_idle_timeout_ms') if k in definition}
            if provider == 'openrouter' and 'env_key' not in safe:
                safe['env_key'] = 'OPENROUTER_API_KEY'
            if not copy_auth and 'env_key' not in safe:
                raise ValueError('Credential-free Codex profiles require an env_key provider.')
            selected.setdefault('model_provider', provider)
            if safe:
                selected['model_providers'] = {provider: safe}
        elif not copy_auth:
            raise ValueError('Credential-free Codex profiles require an environment provider.')
        text = '\n'.join(f'{k} = {_toml(v)}' for k, v in selected.items())
        (home / 'config.toml').write_text(text)
        (home / f'{profile}.config.toml').write_text(text)
    env = dict(os.environ if base_env is None else base_env, CODEX_HOME=str(home))
    no_proxy = env.get('NO_PROXY', env.get('no_proxy', ''))
    env['NO_PROXY'] = env['no_proxy'] = no_proxy + ',localhost,127.0.0.1,::1'
    return env


def _command(args, *, probe=False, workspace=None, blocked_bin=None,
             allowed_network_hosts=()):
    prefix = _codex_prefix(args)
    command = [*prefix, 'exec']
    if probe: command.append('resume')
    command += ['--json', '--skip-git-repo-check', '-c', 'approval_policy="never"',
                '-c', 'web_search="disabled"', '-c', 'features.apps=false',
                '-c', 'features.plugins=false', '-c', 'features.hooks=false',
                '-c', 'features.multi_agent=false', '-c', 'features.memories=false',
                '-c', 'features.shell_snapshot=false', '-c', 'features.network_proxy=true',
                '-c', 'allow_login_shell=false', '-c', 'analytics.enabled=false']
    if getattr(args, 'codex_text_only', False):
        command += ['-c', 'features.view_image=false', '-c', 'features.image_generation=false',
                    '-c', 'features.browser_use=false', '-c', 'features.computer_use=false']
    tool_path = getattr(args, 'codex_tool_path', None)
    if blocked_bin:
        path = str(blocked_bin) + os.pathsep + (tool_path or os.environ.get('PATH', ''))
        command += ['-c', f'shell_environment_policy.set.PATH={_toml(path)}']
    elif tool_path:
        command += ['-c', f'shell_environment_policy.set.PATH={_toml(tool_path)}']
    if (getattr(args, 'codex_model', None)
            and not _command_option(prefix, '--model', '-m')):
        command += ['--model', args.codex_model]
    # exec resume lacks --profile on current Codex; its isolated base config
    # already contains the same provider/settings as the selected profile.
    for attribute, key in [('codex_provider_base_url', 'model_providers.openrouter.base_url'),
                           ('codex_provider_env_key', 'model_providers.openrouter.env_key'),
                           ('codex_reasoning_effort', 'model_reasoning_effort')]:
        if value := getattr(args, attribute, None): command += ['-c', f'{key}={_toml(value)}']
    if probe:
        command += ['-c', 'sandbox_mode="read-only"', '-c', 'features.shell_tool=false',
                    '-c', 'features.unified_exec=false', '-c', 'features.code_mode=false',
                    '-c', 'features.code_mode_host=false', '-c', 'features.view_image=false',
                    '-c', 'features.image_generation=false', '-c', 'features.browser_use=false',
                    '-c', 'features.computer_use=false', '-c', 'features.sleep_tool=false']
    else:
        if workspace is not None:
            if not allowed_network_hosts:
                raise ValueError('Codex agent network requires an explicit feedback host allowlist.')
            workspace = Path(workspace).resolve()
            scratch = Path(workspace) / '.tmp'
            scratch.mkdir(exist_ok=True)
            for key, value in {'TMPDIR':str(scratch), 'TMPPREFIX':str(scratch/'zsh'),
                               'MPLCONFIGDIR':str(scratch/'matplotlib')}.items():
                command += ['-c', f'shell_environment_policy.set.{key}={_toml(value)}']
            permissions = {'filesystem': {':root': 'read', str(workspace): 'write',
                            str(Path.home() / '.codex/auth.json'): 'deny',
                            str(Path.home() / '.zshrc'): 'deny'},
                           'network': {
                               'enabled': True,
                               'allow_local_binding': False,
                               'allow_upstream_proxy': False,
                               'domains': {host: 'allow' for host in allowed_network_hosts},
                           }}
            # This is a mandatory benchmark boundary, not a caller-controlled
            # option. Resolve it from this module so it remains correct for any
            # process working directory.
            benchmark_root = Path(__file__).resolve().parents[2]
            denied_roots = set()
            readable_project_entries = {'venv', 'third-party'}
            for child in benchmark_root.iterdir():
                # These roots supply the scientific Python environment and
                # native/vendor resources required by some installed packages.
                # Every other project-root entry is private from the agent.
                if child.name in readable_project_entries:
                    continue
                denied = child.resolve()
                denied_roots.add(denied)
                permissions['filesystem'][str(denied)] = 'deny'
            readable = ', '.join(str(benchmark_root / name)
                                 for name in sorted(readable_project_entries))
            _logger.info('Codex project deny rules (%d; readable exceptions: %s):',
                         len(denied_roots), readable)
            for denied in sorted(denied_roots, key=str):
                _logger.info('Codex project deny: %s', denied)
            # Nested deny mounts cannot be installed after a denied parent has
            # already been hidden by bubblewrap. Avoid redundant child rules.
            def deny_unless_covered(path):
                path = Path(path).resolve()
                if not any(path == root or root in path.parents for root in denied_roots):
                    permissions['filesystem'][str(path)] = 'deny'
            deny_unless_covered(Path(__file__).resolve().parents[1])
            deny_unless_covered(args.save_path)
            if answer := getattr(args, 'answer', None):
                deny_unless_covered(Path(answer).resolve().parent)
            search_path = os.pathsep.join(filter(None, (
                tool_path, os.environ.get('PATH', ''), str(Path(sys.executable).parent))))
            for directory in search_path.split(os.pathsep):
                executable = Path(directory) / 'mdbench'
                if executable.is_file():
                    permissions['filesystem'][str(executable.resolve())] = 'deny'
            if git_binary := shutil.which('git', path='/usr/local/bin:/usr/bin:/bin'):
                permissions['filesystem'][git_binary] = 'deny'
            if blocked := getattr(args, 'codex_blocked_tools', None):
                # The shim lives outside denied directories so its read grant
                # need not reopen a deny mount on current Linux runtimes.
                permissions['filesystem'][str(Path(blocked).resolve())] = 'read'
            command += ['-c', f'permissions.benchmark={_toml(permissions)}',
                        '-c', 'default_permissions="benchmark"', '--color', 'never']
        else:
            command += ['--sandbox', 'workspace-write', '--color', 'never',
                        '-c', 'sandbox_workspace_write.network_access=true']
    return command


def _invoke(command, prompt, cwd, env, events, stderr, timeout, *, stdout=None, process_log=None):
    if timeout <= 0: raise ValueError('Timeout must be positive.')
    events, stderr = Path(events), Path(stderr)
    stdout = Path(stdout) if stdout is not None else events.with_name('stdout.txt')
    process_log = Path(process_log) if process_log is not None else events.with_name('process.json')
    events.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    try:
        with stdout.open('w') as output, stderr.open('w') as error:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=output, stderr=error,
                                       text=True, cwd=cwd, env=env, start_new_session=True)
            identity = None
            try: identity = Path(f'/proc/{process.pid}/stat').read_text().split()[21]
            except OSError: pass
            process_log.write_text(json.dumps({'pid':process.pid, 'start_ticks':identity, 'cwd':str(cwd)}))
            timed_out = False
            try:
                process.communicate(prompt, timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                # Allow the CLI to flush its rollout and events before forced shutdown.
                for sig, grace in ((signal.SIGINT, 5), (signal.SIGTERM, 3), (signal.SIGKILL, 3)):
                    try: os.killpg(process.pid, sig)
                    except ProcessLookupError: break
                    try:
                        process.communicate(timeout=grace)
                        break
                    except subprocess.TimeoutExpired: continue
                process.wait()
            except BaseException:
                try: os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                process.wait()
                raise
    finally:
        if stdout.exists(): shutil.copy2(stdout, events)
        if stderr.exists(): stderr.write_text(clean_ansi(stderr.read_text()))
    return {'returncode': process.returncode, 'timed_out': timed_out, 'elapsed_seconds': time.monotonic() - start}


def _thread_id(events: Path):
    for line in events.read_text().splitlines():
        try: event = json.loads(line)
        except json.JSONDecodeError: continue
        if event.get('type') == 'thread.started': return event['thread_id']
    raise RuntimeError('Codex events contain no thread.started session ID.')


def _session_id(checkpoint: Path):
    for line in checkpoint.read_text().splitlines():
        item = json.loads(line)
        if item.get('type') == 'session_meta': return item['payload']['id']
    raise ValueError('Checkpoint has no session_meta ID.')


class CodexConversation:
    def __init__(self, args, model):
        self.args, self.model = args, model
        self.checkpoint = Path(model['session']).resolve()
        self.session_id = _session_id(self.checkpoint)
        self.frozen_hash = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()

    def ask(self, prompt: str, probe_name: str, probe_description: str,
            *, output_dir: str | Path) -> str:
        """Ask one probe question and return exactly one parsed equation.

        ``probe_name`` and ``probe_description`` are available for custom prompt
        construction. Using the supplied standardized ``prompt`` unchanged is
        recommended for consistency across algorithms.
        """
        del probe_name, probe_description
        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        audit = output / 'audit'; audit.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='mdbench-probe-') as directory:
            root = Path(directory)
            workspace = root / 'workspace'
            workspace.mkdir(parents=True)
            blocked_bin = _blocked_command_directory(root)
            home = root / 'home'
            env = _isolated_home(home, _selected_profile(self.args),
                                 base_env=getattr(self.args, '_codex_child_env', None),
                                 copy_auth=getattr(self.args, '_codex_copy_auth', True),
                                 isolated_config=getattr(self.args, '_codex_isolated_config', None))
            env['PATH'] = str(blocked_bin) + os.pathsep + env.get('PATH', '')
            sessions = home / 'sessions'
            sessions.mkdir()
            # Rebind the now-deleted training workspace; conversation content and
            # the end-of-run rollout stay unchanged, apart from working directory.
            lines = []
            for line in self.checkpoint.read_text().splitlines():
                item = json.loads(line)
                if item.get('type') in ('session_meta', 'turn_context') and 'cwd' in item.get('payload', {}):
                    item['payload']['cwd'] = str(workspace)
                lines.append(json.dumps(item))
            meta = next(json.loads(line)['payload'] for line in lines if json.loads(line).get('type') == 'session_meta')
            timestamp = meta['timestamp'][:19].replace(':', '-')
            filename = self.model.get('rollout_filename', f'rollout-{timestamp}-{self.session_id}.jsonl')
            if Path(filename).name != filename: raise ValueError('Invalid rollout filename.')
            (sessions / filename).write_text('\n'.join(lines) + '\n')
            reply = output / 'reply.txt'
            status = _invoke(_command(self.args, probe=True, blocked_bin=blocked_bin)
                             + ['-o', str(reply), self.session_id, '-'],
                             prompt, workspace, env, audit / 'codex_events.jsonl', audit / 'stderr.txt',
                             getattr(self.args, 'probe_timeout', 120), stdout=audit / 'stdout.txt',
                             process_log=audit / 'process.json')
            (audit / 'status.json').write_text(json.dumps(status, indent=2))
            (output / 'prompt.txt').write_text(prompt)
            if status['timed_out'] or status['returncode'] != 0 or not reply.is_file():
                raise RuntimeError(f'Codex probe failed: {status}; see {output}')
            text = clean_ansi(reply.read_text()).strip()
            reply.write_text(text + '\n')
            if hashlib.sha256(self.checkpoint.read_bytes()).hexdigest() != self.frozen_hash:
                raise RuntimeError('Original Codex checkpoint was modified during probe evaluation.')
            from ..scoring import submission_formulas
            formulas = submission_formulas(text)
            if len(formulas) != 1:
                raise ValueError('Codex probe response must contain exactly one equation.')
            return formulas[0]


def get_ask(args, checkpoint: Any) -> Callable:
    """Create a fresh probe callable from one frozen Codex checkpoint."""
    conversation = resume(args, checkpoint)
    if getattr(conversation.args, 'openrouter_gateway', False):
        save = Path(checkpoint['session']).resolve().parents[1]
        _ensure_openrouter_gateway(conversation.args, save)
    return conversation.ask


def resume(args, model):
    """Backward-compatible wrapper returning a conversation for one checkpoint."""
    import copy
    args = copy.copy(args)
    if not getattr(args, 'codex_command', None): args.codex_command = model.get('codex_command')
    if not getattr(args, 'codex_model', None): args.codex_model = model.get('codex_model')
    return CodexConversation(args, model)
