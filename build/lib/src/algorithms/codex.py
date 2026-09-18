"""Codex CLI baseline with persisted end-of-run checkpoints.

Each probe restores the exact saved rollout into a fresh CODEX_HOME. No probe
turns are ever appended to the original checkpoint or seen by another probe.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import time
import tomllib

ANSI = re.compile(r'\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))')


def clean_ansi(text):
    return ANSI.sub('', text)


def _toml(value):
    if isinstance(value, bool): return 'true' if value else 'false'
    if isinstance(value, str): return json.dumps(value)
    if isinstance(value, (int, float)): return str(value)
    if isinstance(value, list): return '[' + ', '.join(_toml(v) for v in value) + ']'
    if isinstance(value, dict): return '{' + ', '.join(f'{json.dumps(k)} = {_toml(v)}' for k, v in value.items()) + '}'
    raise TypeError('Unsupported Codex provider configuration value.')


def _isolated_home(home: Path):
    home.mkdir(parents=True, exist_ok=True)
    original = Path(os.environ.get('CODEX_HOME', Path.home() / '.codex'))
    auth = original / 'auth.json'
    if auth.is_file():
        shutil.copy2(auth, home / 'auth.json')
        (home / 'auth.json').chmod(0o600)
    # Preserve account/provider configuration, but not user instructions, MCP,
    # plugins, skills or hooks. Credentials are never written into run artifacts.
    config_file = original / 'config.toml'
    if config_file.is_file():
        config = tomllib.loads(config_file.read_text())
        keys = ('model', 'model_provider', 'model_providers', 'model_reasoning_effort',
                'chatgpt_base_url', 'openai_base_url')
        (home / 'config.toml').write_text('\n'.join(f'{k} = {_toml(config[k])}' for k in keys if k in config))
    env = dict(os.environ, CODEX_HOME=str(home))
    no_proxy = env.get('NO_PROXY', env.get('no_proxy', ''))
    env['NO_PROXY'] = env['no_proxy'] = no_proxy + ',localhost,127.0.0.1,::1'
    return env


def _command(args, *, probe=False):
    command = [getattr(args, 'codex_bin', 'codex'), 'exec']
    if probe: command.append('resume')
    command += ['--json', '--skip-git-repo-check', '-c', 'approval_policy="never"',
                '-c', 'web_search="disabled"', '-c', 'features.apps=false',
                '-c', 'features.plugins=false', '-c', 'features.hooks=false',
                '-c', 'features.multi_agent=false', '-c', 'features.memories=false']
    if getattr(args, 'codex_model', None): command += ['--model', args.codex_model]
    if probe:
        command += ['-c', 'sandbox_mode="read-only"', '-c', 'features.shell_tool=false',
                    '-c', 'features.unified_exec=false', '-c', 'features.code_mode=false',
                    '-c', 'features.code_mode_host=false', '-c', 'features.view_image=false',
                    '-c', 'features.image_generation=false', '-c', 'features.browser_use=false',
                    '-c', 'features.computer_use=false', '-c', 'features.sleep_tool=false']
    else:
        command += ['--sandbox', 'workspace-write', '--color', 'never',
                    '-c', 'sandbox_workspace_write.network_access=true']
    return command


def _invoke(command, prompt, cwd, env, events, stderr, timeout):
    if timeout <= 0: raise ValueError('Timeout must be positive.')
    start = time.monotonic()
    with events.open('w') as stdout, stderr.open('w') as error:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=stdout, stderr=error,
                                   text=True, cwd=cwd, env=env, start_new_session=True)
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
    stderr.write_text(clean_ansi(stderr.read_text()))
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


def run(args, problem_file: Path, train_data_npy_file: Path, feedback_server_url):
    save = Path(args.save_path).resolve()
    save.mkdir(parents=True, exist_ok=True)
    events, errors, checkpoint = save / 'codex.events.jsonl', save / 'codex.stderr.txt', save / 'codex.session.jsonl'
    submission = save / 'submission.txt'
    # Submission is deliberately saved only from this run, never a stale artifact.
    if submission.exists(): raise FileExistsError(f'Run artifact already exists: {submission}')
    with tempfile.TemporaryDirectory(prefix='mdbench-codex-') as directory:
        root = Path(directory)
        workspace = root / 'workspace'
        workspace.mkdir()
        shutil.copy2(problem_file, workspace / 'problem.json')
        shutil.copy2(train_data_npy_file, workspace / 'train.npy')
        env = _isolated_home(root / 'home')
        prompt = f'''Discover a scientific mechanism from the observations in problem.json and train.npy.
Only observed variables are provided. The NPY array has shape (variables, samples);
row names and their scientific meanings are in problem.json data_columns/variables.
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
End with the same equations as your submission, so the final conversation records it.
'''
        (save / 'prompt.txt').write_text(prompt)
        final = workspace / 'last_message.txt'
        status = _invoke(_command(args) + ['-o', str(final), '-'], prompt, workspace, env, events, errors, args.timeout)
        model = {'algorithm': 'codex', 'submission': str(submission), 'events': str(events),
                 'session': str(checkpoint), 'codex_model': getattr(args, 'codex_model', None), **status}
        if (workspace / 'submission.txt').is_file():
            submission.write_text(clean_ansi((workspace / 'submission.txt').read_text()))
        if final.is_file():
            text = clean_ansi(final.read_text())
            (save / 'last_message.txt').write_text(text)
            if not submission.is_file():
                from ..scoring import submission_formulas
                try:
                    submission_formulas(text)
                except ValueError:
                    pass
                else:
                    submission.write_text(text)
        try:
            thread_id = _thread_id(events)
            sessions = list((root / 'home' / 'sessions').rglob(f'*{thread_id}*.jsonl'))
            if len(sessions) != 1: raise RuntimeError('Expected exactly one persisted Codex rollout.')
            shutil.copy2(sessions[0], checkpoint)
            model['session_id'] = thread_id
            model['rollout_filename'] = sessions[0].name
        except (ValueError, RuntimeError) as exc:
            model['error'] = str(exc)
        (save / 'model.json').write_text(json.dumps(model, indent=2) + '\n')
        if not submission.is_file() or not checkpoint.is_file():
            raise RuntimeError('Codex did not produce both submission and resumable checkpoint; see saved model/events/stderr.')
        # Freeze the saved checkpoint; future probe turns only ever modify copies.
        checkpoint.chmod(0o444)
        return model


class CodexConversation:
    def __init__(self, args, model):
        self.args, self.model = args, model
        self.checkpoint = Path(model['session']).resolve()
        self.session_id = _session_id(self.checkpoint)

    def ask(self, prompt: str, *, output_dir: str | Path) -> str:
        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='mdbench-probe-') as directory:
            root = Path(directory)
            workspace = root / 'workspace'
            workspace.mkdir()
            home = root / 'home'
            env = _isolated_home(home)
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
                             prompt, workspace, env, output / 'events.jsonl', output / 'stderr.txt',
                             getattr(self.args, 'probe_timeout', 120))
            (output / 'status.json').write_text(json.dumps(status, indent=2))
            (output / 'prompt.txt').write_text(prompt)
            if status['timed_out'] or status['returncode'] != 0 or not reply.is_file():
                raise RuntimeError(f'Codex probe failed: {status}; see {output}')
            text = clean_ansi(reply.read_text()).strip()
            reply.write_text(text + '\n')
            return text


def resume(args, model):
    """Return an ask-capable conversation; each ask restores an isolated checkpoint."""
    import copy
    args = copy.copy(args)
    if not getattr(args, 'codex_model', None): args.codex_model = model.get('codex_model')
    return CodexConversation(args, model)


def update_parser(parser):
    parser.add_argument('--timeout', type=float, default=600, help='Agent run time limit in seconds.')
    parser.add_argument('--codex-bin', default='codex')
    parser.add_argument('--codex-model', default=None)
    return parser
