"""Claude Code CLI baseline with frozen transcripts and durable usage accounting.

Example: python -m src.run_experiment --algorithm claudecode --claude-model sonnet
    --problem-file <problem.json> --answer-file <answer.json> --save-path <run>

Uses an installed ``claude`` (or --claude-command), file-based login credentials or
provider environment, and no SDK dependency. Bash requires Claude's sandbox
dependencies on Linux/WSL2 or macOS. Each probe resumes a private transcript copy
with all tools disabled. CLI cost is an API-price estimate, not a billing receipt.
See https://code.claude.com/docs/en/agent-sdk/cost-tracking for result semantics.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable
import uuid

from .codex import atomic_json, clean_ansi, utc, _local_feedback_host

__all__ = ['update_parser', 'run', 'get_ask', 'UsageAccounting']
_accounting_lock = threading.RLock()
_TOKENS = ('input_tokens', 'output_tokens', 'cache_read_input_tokens',
           'cache_creation_input_tokens')
_MODEL_TOKENS = ('inputTokens', 'outputTokens', 'cacheReadInputTokens',
                 'cacheCreationInputTokens')


def _positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('Must be a finite positive number.')
    return number


def update_parser(parser):
    """Register options consumed by the ordinary experiment runner."""
    parser.add_argument('--timeout', type=_positive, default=900)
    parser.add_argument('--claude_bin', default='claude')
    parser.add_argument('--claude_command', default=None,
                        help='Executable and optional --model/--effort options; quote paths with spaces.')
    parser.add_argument('--claude_model', default=None)
    parser.add_argument('--claude_effort', default=None)
    parser.add_argument('--claude_text_only', action='store_true')
    parser.add_argument('--claude_max_budget_usd', type=_positive, default=None,
                        help='Optional CLI spending cap for EACH invocation (run or probe).')
    return parser


def _prefix(args):
    configured = getattr(args, 'claude_command', None)
    command = shlex.split(configured) if configured else [getattr(args, 'claude_bin', 'claude')]
    if not command:
        raise ValueError('--claude-command must contain an executable.')
    # Provider/model overrides are useful; protocol, tools and checkpoint flags
    # belong to the benchmark. In particular, never allow a permission bypass.
    allowed = {'--model', '--effort', '--fallback-model'}
    index = 1
    while index < len(command):
        token = command[index]
        if token.startswith('-'):
            if token.split('=', 1)[0] not in allowed:
                raise ValueError(f'Unsupported option in --claude-command: {token}')
            if '=' not in token:
                index += 1
                if index == len(command) or command[index].startswith('-'):
                    raise ValueError(f'{token} requires a value.')
        index += 1
    return command


def _option(command, name):
    for index, token in enumerate(command):
        if token == name:
            return command[index + 1]
        if token.startswith(name + '='):
            return token.split('=', 1)[1]
    return None


def _isolated_home(home):
    """Keep authentication/provider selection, excluding user prompts and hooks."""
    home.mkdir(parents=True)
    original = Path(os.environ.get('CLAUDE_CONFIG_DIR', Path.home() / '.claude'))
    env = os.environ.copy()
    settings = original / 'settings.json'
    if settings.is_file():
        config = json.loads(settings.read_text(encoding='utf-8'))
        for key, value in config.get('env', {}).items():
            if key.startswith(('ANTHROPIC_', 'AWS_', 'GOOGLE_', 'CLOUD_ML_',
                               'CLAUDE_CODE_USE_', 'CLAUDE_CODE_OAUTH_TOKEN')):
                env.setdefault(key, str(value))
        if config.get('model'):
            env.setdefault('ANTHROPIC_MODEL', config['model'])
    credentials = original / '.credentials.json'
    if credentials.is_file():
        shutil.copy2(credentials, home / credentials.name)
        (home / credentials.name).chmod(0o600)
    env['CLAUDE_CONFIG_DIR'] = str(home)
    env['CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC'] = '1'
    env['CLAUDE_CODE_DISABLE_AUTO_MEMORY'] = '1'
    env['CLAUDE_CODE_SKIP_PROMPT_HISTORY'] = '0'
    env.pop('CLAUDECODE', None)
    env.pop('CLAUDE_CODE_SESSION_ID', None)
    env.pop('PYTHONPATH', None)
    return env


def _command(args, workspace, home, *, feedback_url=None, checkpoint=None):
    prefix = _prefix(args)
    probe = checkpoint is not None
    denied = [str(p.resolve()) for p in Path(__file__).resolve().parents[2].iterdir()
              if p.name not in {'venv', 'third-party'}]
    denied += [str(Path(args.save_path).resolve()), str(home),
               str(Path.home() / '.claude'), str(Path.home() / '.codex'),
               str(Path.home() / '.claude.json'), str(Path.home() / '.ssh')]
    if answer := getattr(args, 'answer_file', None):
        denied.append(str(Path(answer).resolve().parent))
    denied.extend(str(Path.home() / name) for name in ('.bashrc', '.bash_profile', '.zshrc', '.profile'))
    settings = {
        'disableAllHooks': True,
        'autoMemoryEnabled': False,
        'permissions': {
            'defaultMode': 'dontAsk',
            'allow': [] if probe else ['Read', 'Write', 'Edit', 'Glob', 'Grep'],
            'deny': ['Bash(git *)', 'Bash(mdbench *)'] + [
                f'{tool}(/{Path(path).as_posix()}/**)' for path in denied
                for tool in ('Read', 'Edit')],
        },
        'sandbox': {
            'enabled': not probe, 'failIfUnavailable': True,
            'autoAllowBashIfSandboxed': True, 'allowUnsandboxedCommands': False,
            'excludedCommands': [],
            'filesystem': {'denyRead': denied, 'denyWrite': [
                str(workspace / 'problem.json'), str(workspace / 'train.npy')]},
            'network': {'allowedDomains': [] if probe else [_local_feedback_host(feedback_url)],
                        'allowLocalBinding': False},
        },
    }
    settings_file = home / 'benchmark-settings.json'
    atomic_json(settings_file, settings)
    command = [*prefix, '--print', '--verbose', '--output-format', 'stream-json',
               '--include-partial-messages', '--input-format', 'text',
               '--setting-sources', '', '--settings', str(settings_file),
               '--strict-mcp-config', '--mcp-config', '{"mcpServers":{}}',
               '--disable-slash-commands', '--permission-mode', 'dontAsk',
               '--tools', '' if probe else 'Bash,Read,Write,Edit,Glob,Grep']
    for attr, flag in (('claude_model', '--model'), ('claude_effort', '--effort')):
        if (value := getattr(args, attr, None)) and not _option(prefix, flag):
            command += [flag, value]
    if budget := getattr(args, 'claude_max_budget_usd', None):
        command += ['--max-budget-usd', str(_positive(budget))]
    if probe:
        command += ['--resume', str(checkpoint), '--fork-session']
    else:
        command += ['--session-id', str(uuid.uuid4())]
    return command


def _events(path):
    """Read complete JSONL events, including output left by a killed CLI."""
    if not Path(path).is_file():
        return
    for line in Path(path).read_text(encoding='utf-8', errors='replace').splitlines():
        try:
            item = json.loads(clean_ansi(line))
        except (ValueError, TypeError):
            continue
        if isinstance(item, dict):
            yield item


def _summarize(events, baseline=None):
    """Extract final totals, or deduplicated partial usage after interruption."""
    result, init, messages = None, {}, {}
    active = {}
    for item in _events(events):
        if item.get('type') == 'system' and item.get('subtype') == 'init':
            init = item
        elif item.get('type') == 'result':
            result = item
        elif item.get('type') == 'assistant':
            message = item.get('message', {})
            if message.get('id') and message.get('usage'):
                old = messages.setdefault(message['id'], {})
                for key in _TOKENS:
                    old[key] = max(old.get(key, 0), message['usage'].get(key, 0) or 0)
        elif item.get('type') == 'stream_event':
            event = item.get('event', {})
            parent = item.get('parent_tool_use_id')
            if event.get('type') == 'message_start':
                message = event.get('message', {})
                if message.get('id'):
                    active[parent] = message['id']
                    messages.setdefault(message['id'], {}).update(message.get('usage') or {})
            elif event.get('type') == 'message_delta' and parent in active:
                messages[active[parent]].update(event.get('usage') or {})
    result = result or {}
    usage = result.get('usage') or {}
    model_usage = result.get('modelUsage') or {}
    cost = result.get('total_cost_usd')
    complete = bool(usage or model_usage)
    # Crash results can zero the final counters even after billable responses.
    if result.get('subtype') == 'error_during_execution' and not cost:
        complete = False
        cost = None
    tokens = {key: sum(m.get(key, 0) or 0 for m in messages.values()) for key in _TOKENS}
    if complete:
        tokens = {key: usage.get(key, 0) or 0 for key in _TOKENS}
    version = tuple(int(v) for v in re.findall(r'\d+', init.get('claude_code_version', ''))[:3])
    restored = baseline is not None and version >= (2, 1, 277)
    if model_usage and complete:
        tokens = {key: sum(m.get(field, 0) or 0 for m in model_usage.values())
                  for key, field in zip(_TOKENS, _MODEL_TOKENS)}
        if restored:
            previous = baseline.get('model_usage') or {}
            for key, field in zip(_TOKENS, _MODEL_TOKENS):
                tokens[key] = max(0, tokens[key] - sum(m.get(field, 0) or 0 for m in previous.values()))
    if cost is not None:
        cost = float(cost)
        if not math.isfinite(cost) or cost < 0:
            cost = None
        elif restored:
            old_cost = baseline.get('session_cost_usd')
            cost = max(0.0, cost - old_cost) if old_cost is not None else None
    return {
        **tokens, 'total_tokens': sum(tokens.values()),
        'cost_usd': cost, 'cost_source': 'claude_cli_estimate' if cost is not None else 'unavailable',
        'usage_complete': complete, 'session_cost_usd': result.get('total_cost_usd'),
        'model_usage': model_usage, 'duration_api_ms': result.get('duration_api_ms'),
        'session_id': result.get('session_id') or init.get('session_id'),
        'claude_model': init.get('model'), 'claude_code_version': init.get('claude_code_version'),
        'is_error': bool(result.get('is_error')),
        'result_subtype': result.get('subtype'), 'result': result.get('result', ''),
    }


class UsageAccounting:
    """Fsync a per-invocation ledger and aggregate run/probe costs and durations.

    Instances share a process-wide lock so parallel probe factories cannot lose
    updates. Reopening a ledger preserves completed and interrupted invocations.
    Unknown costs remain null; known_cost_usd is a lower bound in that case.
    """
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _record(self, record):
        ledger = self.root / 'usage.jsonl'
        # A controller crash may leave half a JSON line. Keep it for audit, but
        # prevent it from swallowing the next successfully appended record.
        if ledger.is_file() and ledger.stat().st_size:
            with ledger.open('rb+') as stream:
                stream.seek(-1, os.SEEK_END)
                if stream.read(1) != b'\n':
                    stream.write(b'\n')
        with (self.root / 'usage.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    def begin(self, phase, events):
        request_id = uuid.uuid4().hex
        with _accounting_lock:
            self._record(dict(event='started', time=utc(), request_id=request_id,
                              phase=phase, events=str(events)))
            self.snapshot()
        return request_id

    def finish(self, request_id, phase, status, events, baseline=None):
        usage = _summarize(events, baseline)
        record = dict(event='finished', time=utc(), request_id=request_id, phase=phase,
                      events=str(events), **status, **{k: v for k, v in usage.items() if k != 'result'})
        with _accounting_lock:
            self._record(record)
            self.snapshot()
        atomic_json(Path(events).parent / 'usage.json', record)
        return usage

    def snapshot(self):
        with _accounting_lock:
            started, finished = {}, {}
            for item in _events(self.root / 'usage.jsonl'):
                target = started if item.get('event') == 'started' else finished
                target[item['request_id']] = item
            rows = list(finished.values())
            pending = [v for k, v in started.items() if k not in finished]

            def total(items):
                known = sum(r['cost_usd'] for r in items if r.get('cost_usd') is not None)
                unknown = sum(r.get('cost_usd') is None for r in items)
                return {**{key: sum(r.get(key, 0) for r in items) for key in (*_TOKENS, 'total_tokens')},
                        'invocations': len(items), 'elapsed_seconds': sum(r['elapsed_seconds'] for r in items),
                        'known_cost_usd': known, 'cost_usd': None if unknown else known,
                        'unknown_cost_invocations': unknown,
                        'usage_complete': all(r['usage_complete'] for r in items)}

            summary = dict(time=utc(), **total(rows), pending=pending,
                           by_phase={phase: total([r for r in rows if r['phase'] == phase])
                                     for phase in sorted({r['phase'] for r in rows})},
                           cost_source='claude_cli_estimate',
                           duration_semantics='Sum of invocation wall times; parallel probes overlap.')
            if pending:
                summary.update(cost_usd=None, usage_complete=False)
            timestamps = [datetime.fromisoformat(r['time']) for r in started.values()]
            end = utc() if pending else max((r['time'] for r in rows), default=utc())
            summary['wall_seconds'] = max(0.0, (datetime.fromisoformat(end) - min(timestamps)).total_seconds()) if timestamps else 0.0
            atomic_json(self.root / 'usage.json', summary)
            return summary


def _stop(process):
    if process.poll() is not None:
        return
    if os.name == 'nt':
        # taskkill /T includes tool descendants, unlike Popen.kill on Windows.
        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        if process.poll() is None:
            process.kill()
        process.wait()
    else:
        for sig, grace in ((signal.SIGINT, 3), (signal.SIGTERM, 2), (signal.SIGKILL, 2)):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                break
            try:
                process.wait(timeout=grace)
                break
            except subprocess.TimeoutExpired:
                continue


def _invoke(command, prompt, cwd, env, audit, timeout, account, phase, baseline=None):
    _positive(timeout)
    audit.mkdir(parents=True, exist_ok=True)
    events, stderr = audit / 'claude.events.jsonl', audit / 'stderr.txt'
    request_id = account.begin(phase, events)
    start, process = time.monotonic(), None
    status = {'returncode': None, 'timed_out': False}
    try:
        with events.open('w', encoding='utf-8') as output, stderr.open('w', encoding='utf-8') as error:
            kwargs = {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == 'nt' else {'start_new_session': True}
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=output, stderr=error,
                                       text=True, encoding='utf-8', cwd=cwd, env=env, **kwargs)
            atomic_json(audit / 'process.json', {'pid': process.pid, 'cwd': str(cwd)})
            try:
                process.communicate(prompt, timeout=timeout)
            except subprocess.TimeoutExpired:
                status['timed_out'] = True
                _stop(process)
    except BaseException as exc:
        status['error'] = f'{type(exc).__name__}: {exc}'
        if process is not None:
            _stop(process)
        raise
    finally:
        if process is not None:
            status['returncode'] = process.returncode
        status['elapsed_seconds'] = time.monotonic() - start
        if events.exists():
            shutil.copy2(events, audit / 'stdout.txt')
        if stderr.exists():
            stderr.write_text(clean_ansi(stderr.read_text(encoding='utf-8', errors='replace')), encoding='utf-8')
        usage = account.finish(request_id, phase, status, events, baseline)
        atomic_json(audit / 'status.json', status)
    return status, usage


# _prompt is kept verbatim in sync with codex.run; the regression test compares
# rendered text for both text-only modes (only the CLI option name differs).


def _prompt(args, feedback_server_url):
    return f'''# Objective
Reconstruct the scientific mechanism that generated the observations in
problem.json and train.npy.

The primary goal is a mechanism model, not merely a phenomenological model. A
phenomenological model can fit the observable input-target relation directly;
while your model must instead describe scientifically meaningful unobserved 
internal components, states, or activities and how they are organized so that 
their equations generate the observed relation; otherwise, even a perfectly 
accurate direct fit is insufficient.

# Provided data
- Only observed variables are provided.
- train.npy has shape (variables, samples).
- problem.json data_columns gives the row order; problem.json variables gives
  each observed variable's role, scientific meaning, and unit when available.
- Scientific Python is available at {sys.executable} for numpy/sympy analysis.
{
    '- Use textual or numerical tables for analysis; image input is unavailable in this experiment.' 
    if getattr(args, 'claude_text_only', False) else ''
}

# Model requirements
- Submit a self-contained system of algebraic equations involving the supplied
  inputs, the target, and scientifically meaningful unobserved internal variables.
- The equations must jointly and uniquely solve the target and every introduced
  internal variables as explicit expressions of the supplied inputs.
- A direct target-versus-input equation may follow from your mechanism cannot
  replace the internal mechanistic equations.
- Do not leave free symbolic parameters. Estimate necessary constants and define
  each one numerically in an equation (for example, k = 1.2345).
- Use only ordinary algebra, ^ or ** powers, sqrt, exp, log, trigonometric
  functions, and abs. Do not use differential equations.
- Prefer the smallest scientifically coherent mechanism that explains the data.
  Use variable meanings, units, scaling, and numerical behavior to distinguish
  causal/mechanistic hypotheses from curve fits.

# Investigation and feedback
Inspect the metadata and data, formulate candidate mechanisms, test their
observable consequences, and refine the best mechanism. Write a valid initial
submission.txt immediately, then keep it updated while improving it within
{args.timeout} seconds so it survives a time-limit interruption.

You can obtain objective accuracy feedback by POST to {feedback_server_url}. 
This feedback checks observable fit only; it does not establish that you recovered 
the mechanism. After this run, you may be asked to derive several unobserved 
internal quantities using your frozen submitted model, and you will not be allowed 
to revise that model then.

Keep the environment-provided HTTP proxy enabled: it is the sandbox's controlled
route to this local feedback endpoint. Do not set trust_env=False or override it.
Upload three multipart file fields: problem (problem.json), train_data (train.npy),
and submission (submission.txt). For example, use Python:
session.post(url, files={{
    "problem": open("problem.json", "rb"),
    "train_data": open("train.npy", "rb"),
    "submission": open("submission.txt", "rb")
}}).json()

# Submission format
submission.txt must contain only equations, with one equality per line and no
Markdown or prose. End your final response with exactly the same equations as
submission.txt so the conversation records the submitted model.

# Boundaries
Do not modify problem.json or train.npy. Do not use external reference answers,
run git commands, or read shell startup files or credentials. The mdbench command
and benchmark package are intentionally unavailable; use only the feedback
endpoint described above. Do not inspect files outside this workspace except the
provided Python environment.
'''


def run(args, problem_file: Path, train_data_npy_file: Path, feedback_server_url) -> tuple[list[str], Any]:
    """Run discovery, save a frozen transcript, and return equations/checkpoint."""
    from ..scoring import submission_formulas

    save = Path(args.save_path).resolve()
    save.mkdir(parents=True, exist_ok=True)
    saved = save / 'saved_checkpoint'
    saved.mkdir(exist_ok=True)
    checkpoint = saved / 'claude.session.jsonl'
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    account = UsageAccounting(save)
    prompt = _prompt(args, feedback_server_url)
    (save / 'prompt.txt').write_text(prompt, encoding='utf-8')
    with tempfile.TemporaryDirectory(prefix='mdbench-claude-') as directory:
        root = Path(directory)
        workspace, home = root / 'workspace', root / 'home'
        workspace.mkdir()
        shutil.copy2(problem_file, workspace / 'problem.json')
        shutil.copy2(train_data_npy_file, workspace / 'train.npy')
        env = _isolated_home(home)
        command = _command(args, workspace, home, feedback_url=feedback_server_url)
        status, usage = _invoke(command, prompt, workspace, env, save / 'audit',
                                args.timeout, account, 'run')
        session_id = usage['session_id'] or _option(command, '--session-id')
        model = dict(algorithm='claudecode', session=str(checkpoint),
                     claude_command=getattr(args, 'claude_command', None),
                     claude_bin=getattr(args, 'claude_bin', 'claude'),
                     claude_model=usage['claude_model'] or getattr(args, 'claude_model', None),
                     claude_effort=getattr(args, 'claude_effort', None),
                     session_id=session_id, usage={k: v for k, v in usage.items() if k != 'result'},
                     **status)
        sessions = list((home / 'projects').rglob(f'{session_id}.jsonl'))
        if len(sessions) == 1:
            shutil.copy2(sessions[0], checkpoint)
            model['checkpoint_sha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            checkpoint.chmod(0o444)
        else:
            model['error'] = 'Expected exactly one persisted Claude Code transcript.'
        atomic_json(saved / 'model.json', model)
        reply = clean_ansi(usage['result'])
        (save / 'audit' / 'last_message.txt').write_text(reply, encoding='utf-8')
        submission = workspace / 'submission.txt'
        text = clean_ansi(submission.read_text(encoding='utf-8')) if submission.is_file() else ''
        # As with Codex, a deadline may leave a valid initial submission/checkpoint.
        if not checkpoint.is_file() or not (text.strip() or reply.strip()):
            raise RuntimeError(f'Claude Code did not produce both submission and resumable checkpoint; see {save / "audit"}.')
        return submission_formulas(text if text.strip() else reply), model


class ClaudeCodeConversation:
    def __init__(self, args, model):
        self.args, self.model = args, model
        self.checkpoint = Path(model['session']).resolve()
        self.frozen_hash = model.get('checkpoint_sha256') or hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()

    def ask(self, prompt: str, probe_name: str, probe_description: str, *, output_dir: str | Path) -> str:
        """Answer the evaluator's unchanged prompt from an independent frozen copy."""
        from ..scoring import submission_formulas

        del probe_name, probe_description
        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        (output / 'prompt.txt').write_text(prompt, encoding='utf-8')
        if hashlib.sha256(self.checkpoint.read_bytes()).hexdigest() != self.frozen_hash:
            raise RuntimeError('Original Claude Code checkpoint was modified.')
        with tempfile.TemporaryDirectory(prefix='mdbench-claude-probe-') as directory:
            root = Path(directory)
            workspace, home = root / 'workspace', root / 'home'
            workspace.mkdir()
            env = _isolated_home(home)
            restored = root / 'session.jsonl'
            # Explicit transcript paths avoid depending on Claude's internal
            # project-directory name encoding. Never resume the original file.
            restored.write_bytes(self.checkpoint.read_bytes())
            status, usage = _invoke(_command(self.args, workspace, home, checkpoint=restored),
                                    prompt, workspace, env, output / 'audit',
                                    getattr(self.args, 'probe_timeout', 120),
                                    UsageAccounting(self.checkpoint.parents[1]), 'probe',
                                    baseline=self.model.get('usage'))
            text = clean_ansi(usage['result']).strip()
            (output / 'reply.txt').write_text(text + '\n', encoding='utf-8')
            if status['timed_out'] or status['returncode'] != 0 or usage['is_error'] or not text:
                raise RuntimeError(f'Claude Code probe failed: {status}; see {output}.')
            if hashlib.sha256(self.checkpoint.read_bytes()).hexdigest() != self.frozen_hash:
                raise RuntimeError('Original Claude Code checkpoint was modified during evaluation.')
            formulas = submission_formulas(text)
            if len(formulas) != 1:
                raise ValueError('Claude Code probe response must contain exactly one equation.')
            return formulas[0]


def resume(args, model):
    """Restore model/CLI selection without mutating the evaluator's arguments."""
    args = copy.copy(args)
    for name in ('claude_command', 'claude_model', 'claude_effort', 'claude_bin'):
        if not getattr(args, name, None):
            setattr(args, name, model.get(name))
    return ClaudeCodeConversation(args, model)


def get_ask(args, checkpoint: Any) -> Callable:
    """Return a fresh probe callable, compatible with src.run_experiment."""
    return resume(args, checkpoint).ask
