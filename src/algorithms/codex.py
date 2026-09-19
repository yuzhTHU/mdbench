"""Codex CLI baseline with persisted end-of-run checkpoints.

Each probe restores the exact saved rollout into a fresh CODEX_HOME. No probe
turns are ever appended to the original checkpoint or seen by another probe.
"""
from __future__ import annotations
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import shlex
import threading
import uuid
import requests
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import tomllib

__all__ = ["update_parser", "run"]
ANSI = re.compile(r'\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))')


def update_parser(parser):
    """ Add Algorithm-specific arguments to the parser. """
    parser.add_argument('--timeout', type=float, default=600, help='Agent run time limit in seconds.')
    parser.add_argument('--codex_bin', default='codex')
    parser.add_argument('--codex_model', default=None)
    parser.add_argument('--codex_profile', default=None)
    parser.add_argument('--codex_text_only', action='store_true', help='Disable tools that can add unsupported image input to text-only models.')
    parser.add_argument('--codex_provider_base_url', default=None)
    parser.add_argument('--codex_provider_env_key', default=None)
    parser.add_argument('--codex_reasoning_effort', default=None)
    parser.add_argument('--persist_runtime', action='store_true', help='Keep live workspace and rollouts under save_path for interrupted-run recovery.')
    parser.add_argument('--codex_private_root', default=None, help='Deny agent tool access to private benchmark files, while allowing its workspace and the project venv.')
    parser.add_argument('--codex_blocked_tools', default=None)
    parser.add_argument('--codex_tool_path', default=None)
    return parser



def run(args, problem_file: Path, train_data_npy_file: Path, feedback_server_url):
    save = Path(args.save_path).resolve()
    save.mkdir(parents=True, exist_ok=True)
    audit, saved_checkpoint = save / 'audit', save / 'saved_checkpoint'
    audit.mkdir(exist_ok=True); saved_checkpoint.mkdir(exist_ok=True)
    events, errors = audit / 'codex.events.jsonl', audit / 'stderr.txt'
    stdout, process_log = audit / 'stdout.txt', audit / 'process.json'
    checkpoint = saved_checkpoint / 'codex.session.jsonl'
    if getattr(args, 'persist_runtime', False) and not getattr(args, 'codex_profile', None):
        raise ValueError('Persistent runtimes require an environment-authenticated profile.')
    runtime = audit / 'runtime'
    context = nullcontext(str(runtime)) if getattr(args, 'persist_runtime', False) else tempfile.TemporaryDirectory(prefix='mdbench-codex-')
    with context as directory:
        root = Path(directory)
        workspace = root / 'workspace'
        workspace.mkdir(parents=True)
        shutil.copy2(problem_file, workspace / 'problem.json')
        shutil.copy2(train_data_npy_file, workspace / 'train.npy')
        env = _isolated_home(root / 'home', getattr(args, 'codex_profile', None))
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
Use a requests.Session with trust_env=False for local feedback endpoints.
Upload three multipart file fields: problem (problem.json), train_data (train.npy),
submission (submission.txt). For example use Python session.post(url, files={{
"problem": open("problem.json", "rb"), "train_data": open("train.npy", "rb"),
"submission": open("submission.txt", "rb")}}).json().
Internal predictions may be queried later. Do not use external reference answers.
Do not run git commands. Do not read shell startup files or credentials.
Do not inspect files outside this workspace, except the provided Python environment.
End with the same equations as your submission, so the final conversation records it.
'''
        (save / 'prompt.txt').write_text(prompt)
        final = workspace / 'last_message.txt'
        status = _invoke(_command(args, workspace=workspace) + ['-o', str(final), '-'],
                         prompt, workspace, env, events, errors, args.timeout,
                         stdout=stdout, process_log=process_log)
        model = {'algorithm': 'codex', 'events': str(events),
                 'session': str(checkpoint), 'codex_model': getattr(args, 'codex_model', None), **status}
        model['codex_profile'] = getattr(args, 'codex_profile', None)
        if getattr(args, 'persist_runtime', False): model['runtime'] = str(runtime)
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
        return submission_formulas(submission), resume(args, model).ask



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
    """Fsync request accounting, reserve concurrent costs and recover interrupted calls.

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
            if self.cost + sum(r['reserved_usd'] for r in self.pending.values()) + reserve > self.budget:
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


def _isolated_home(home: Path, profile=None):
    home.mkdir(parents=True, exist_ok=True)
    original = Path(os.environ.get('CODEX_HOME', Path.home() / '.codex'))
    auth = original / 'auth.json'
    if auth.is_file() and not profile:
        shutil.copy2(auth, home / 'auth.json')
        (home / 'auth.json').chmod(0o600)
    # Preserve account/provider configuration, but not user instructions, MCP,
    # plugins, skills or hooks. Credentials are never written into run artifacts.
    config = {}
    config_file = original / 'config.toml'
    if config_file.is_file():
        config = tomllib.loads(config_file.read_text())
        keys = ('model', 'model_provider', 'model_providers', 'model_reasoning_effort',
                'chatgpt_base_url', 'openai_base_url')
        (home / 'config.toml').write_text('\n'.join(f'{k} = {_toml(config[k])}' for k in keys if k in config))
    if profile:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', profile): raise ValueError('Invalid Codex profile name.')
        profile_file = original / f'{profile}.config.toml'
        selected = tomllib.loads(profile_file.read_text())
        provider = selected['model_provider']
        definition = dict(config.get('model_providers', {}).get(provider, {}))
        definition.update(selected.get('model_providers', {}).get(provider, {}))
        # Profiles used for experiments authenticate through environment values;
        # never copy account credentials into their persistent runtime artifacts.
        safe = {k: definition[k] for k in ('name', 'base_url', 'wire_api', 'env_key',
                                          'request_max_retries', 'stream_max_retries',
                                          'stream_idle_timeout_ms') if k in definition}
        if provider == 'openrouter': safe['env_key'] = 'OPENROUTER_API_KEY'
        elif 'env_key' not in safe: raise ValueError('Persistent profiles require an env_key provider.')
        selected = {k: selected[k] for k in ('model', 'model_provider', 'model_reasoning_effort') if k in selected}
        selected['model_providers'] = {provider: safe}
        # The base also contains only this provider, with no inherited auth.
        text = '\n'.join(f'{k} = {_toml(v)}' for k, v in selected.items())
        (home / 'config.toml').write_text(text)
        (home / f'{profile}.config.toml').write_text(text)
    env = dict(os.environ, CODEX_HOME=str(home))
    no_proxy = env.get('NO_PROXY', env.get('no_proxy', ''))
    env['NO_PROXY'] = env['no_proxy'] = no_proxy + ',localhost,127.0.0.1,::1'
    return env


def _command(args, *, probe=False, workspace=None):
    command = [getattr(args, 'codex_bin', 'codex'), 'exec']
    if probe: command.append('resume')
    command += ['--json', '--skip-git-repo-check', '-c', 'approval_policy="never"',
                '-c', 'web_search="disabled"', '-c', 'features.apps=false',
                '-c', 'features.plugins=false', '-c', 'features.hooks=false',
                '-c', 'features.multi_agent=false', '-c', 'features.memories=false',
                '-c', 'features.shell_snapshot=false',
                '-c', 'allow_login_shell=false', '-c', 'analytics.enabled=false']
    if getattr(args, 'codex_text_only', False):
        command += ['-c', 'features.view_image=false', '-c', 'features.image_generation=false',
                    '-c', 'features.browser_use=false', '-c', 'features.computer_use=false']
    if tool_path := getattr(args, 'codex_tool_path', None):
        command += ['-c', f'shell_environment_policy.set.PATH={_toml(tool_path)}']
    if getattr(args, 'codex_model', None): command += ['--model', args.codex_model]
    # exec resume lacks --profile on current Codex; its isolated base config
    # already contains the same provider/settings as the selected profile.
    if not probe and getattr(args, 'codex_profile', None): command += ['--profile', args.codex_profile]
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
        if private := getattr(args, 'codex_private_root', None):
            private = Path(private).resolve()
            scratch = Path(workspace) / '.tmp'
            scratch.mkdir(exist_ok=True)
            for key, value in {'TMPDIR':str(scratch), 'TMPPREFIX':str(scratch/'zsh'),
                               'MPLCONFIGDIR':str(scratch/'matplotlib')}.items():
                command += ['-c', f'shell_environment_policy.set.{key}={_toml(value)}']
            permissions = {'filesystem': {':root': 'read', str(workspace): 'write',
                            str(Path.home() / '.codex/auth.json'): 'deny',
                            str(Path.home() / '.zshrc'): 'deny'},
                           'network': {'enabled': True}}
            aliases = {str(private)}
            for alias in aliases:
                for hidden in ('.env', '.git', 'problems', 'playground', 'legacy', 'docs', 'src', 'logs/run'):
                    permissions['filesystem'][str(Path(alias) / hidden)] = 'deny'
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

    def ask(self, prompt: str, *, output_dir: str | Path) -> str:
        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        audit = output / 'audit'; audit.mkdir(exist_ok=True)
        context = nullcontext(str(audit / 'runtime')) if getattr(self.args, 'persist_runtime', False) else tempfile.TemporaryDirectory(prefix='mdbench-probe-')
        with context as directory:
            root = Path(directory)
            workspace = root / 'workspace'
            workspace.mkdir(parents=True)
            home = root / 'home'
            env = _isolated_home(home, getattr(self.args, 'codex_profile', None))
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
            status = _invoke(_command(self.args, probe=True) + ['-o', str(reply), self.session_id, '-'],
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
            return text


def resume(args, model):
    """Return an ask-capable conversation; each ask restores an isolated checkpoint."""
    import copy
    args = copy.copy(args)
    if not getattr(args, 'codex_model', None): args.codex_model = model.get('codex_model')
    if not getattr(args, 'codex_profile', None): args.codex_profile = model.get('codex_profile')
    return CodexConversation(args, model)
