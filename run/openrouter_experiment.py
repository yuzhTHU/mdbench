"""Auditable, budget-limited experiments through the existing ./run.py runner.

The gateway owns the real repository-.env key. Codex receives only a local
gateway credential. Every request and stream is saved before forwarding.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import hashlib
import importlib.util
import importlib.metadata
import platform
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.algorithms.codex import (
    clean_ansi, utc, atomic_json, load_api_key, UsageAccounting,
    start_responses_gateway, probe_gateway_capacity, gateway_environment,
    stop_codex_processes, is_infrastructure_failure,
)
from src.export_problems import export_task
from src.validate_problem import load_task


def legacy_module(name):
    spec = importlib.util.spec_from_file_location('experiment_' + name, ROOT / 'legacy/src/utils' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


logging = legacy_module('logger')
tag2ansi = legacy_module('tag2ansi').tag2ansi



def summarize(root, metadata, accounting, states):
    grouped = {}
    for entry in metadata['tasks']:
        state = states.get(entry['id'], {})
        result_path = root / entry['task_name'] / 'performance.json'
        if not state: continue
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        evaluation = result.get('evaluation', {})
        family = entry['task_name'].rsplit(' - ', 1)[0]
        key = family + (' / Original control' if not entry['mutations'] else ' / Variants')
        bucket = grouped.setdefault(key, dict(attempted=0, evaluated=0, errors=0,
            phenomenal_id_correct=0, phenomenal_ood_correct=0, probe_count=0,
            probes_id_correct=0, probes_ood_correct=0, seconds=0.0))
        bucket['attempted'] += 1; bucket['seconds'] += result.get('total_seconds', 0)
        bucket['errors'] += int('error' in result or state.get('status') == 'failed')
        bucket['evaluated'] += int(bool(evaluation))
        for split, label in [('id_test', 'id'), ('ood_test', 'ood')]:
            bucket['phenomenal_' + label + '_correct'] += int(evaluation.get('phenomenal', {}).get(split, {}).get('numerically_equivalent', False))
            bucket['probes_' + label + '_correct'] += sum(bool(p.get('scores', {}).get(split, {}).get('numerically_equivalent', False)) for p in evaluation.get('mechanism_probes', []))
        bucket['probe_count'] += state.get('expected_probe_count', len(evaluation.get('mechanism_probes', [])))
    for bucket in grouped.values():
        for label in ('id', 'ood'):
            bucket['phenomenal_' + label + '_accuracy'] = bucket['phenomenal_' + label + '_correct'] / bucket['attempted']
            bucket['probe_' + label + '_accuracy'] = bucket['probes_' + label + '_correct'] / bucket['probe_count'] if bucket['probe_count'] else None
    atomic_json(root / 'summary.json', dict(time=utc(), groups=grouped,
        completed=sum(s.get('status') in ('completed', 'failed') for s in states.values()),
        task_count=len(metadata['tasks']), charged_usd=accounting.cost, stop_reason=accounting.reason))


def experiment(args):
    root = args.root.resolve(); metadata = json.loads((root / 'experiment.json').read_text())
    logging.config_logger(save_path=root / 'controller.log', colorful=False, exp_name=args.phase)
    key = load_api_key(ROOT / '.env', 'OPENROUTER_API_KEY')
    session = requests.Session(); session.trust_env = False
    response = session.get('https://openrouter.ai/api/v1/models', timeout=30); response.raise_for_status()
    model = next(m for m in response.json()['data'] if m['id'] == metadata['model'])
    atomic_json(root / 'model_metadata.json', model)
    accounting = UsageAccounting(root, metadata['budget_usd'], model['pricing'])
    previous_signals = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in previous_signals:
        signal.signal(sig, lambda number, frame: accounting.trip(f'operator_signal_{number}'))
    server = start_responses_gateway(accounting, key, metadata['model'], args.concurrency,
                                     upstream_base_url='https://openrouter.ai/api/v1')
    states_path = root / 'tasks.json'
    states = json.loads(states_path.read_text()) if states_path.exists() else {}
    state_lock = threading.RLock(); processes = {}
    tools = ROOT / 'run/blocked_tools'; tools.mkdir(parents=True, exist_ok=True)
    (tools / 'git').write_text('#!/bin/sh\necho "git is forbidden in this experiment" >&2\nexit 125\n')
    (tools / 'git').chmod(0o755)
    selected = [e for e in metadata['tasks'] if states.get(e['id'], {}).get('status') not in ('completed', 'failed')]
    if args.phase == 'smoke':
        selected = [next(e for e in selected if e['task_name'].startswith(metadata['families'][0]+' - ') and len(e['mutations']) == 5),
                    next(e for e in selected if e['task_name'].startswith(metadata['families'][1]+' - ') and not e['mutations'])]
    elif args.phase == 'repair':
        selected = [e for e in metadata['tasks'] if states.get(e['id'],{}).get('status')=='failed'
                    and (root/e['task_name']/'saved_checkpoint'/'codex.session.jsonl').exists()
                    and (root/e['task_name']/'submission.txt').exists()]
    elif args.phase == 'batch':
        plan = json.loads((root / 'budget_plan.json').read_text())
        allowed = set(plan['selected_task_ids']); selected = [e for e in selected if e['id'] in allowed]
    if args.phase == 'batch' and not (root / 'capacity_probe.json').exists():
        capacity_results = probe_gateway_capacity(
            f'http://127.0.0.1:{server.server_port}', metadata['model'], args.concurrency)
        atomic_json(root / 'capacity_probe.json', {'time':utc(), 'parallel':args.concurrency, 'results':capacity_results})
        if any(result['status'] != 200 for result in capacity_results):
            accounting.trip('capacity_probe_failed')
    def persist():
        with state_lock:
            now = datetime.now(timezone.utc)
            for state in states.values():
                if state.get('status') in ('running', 'exporting'):
                    state['elapsed_seconds'] = (now - datetime.fromisoformat(state['started'])).total_seconds()
            atomic_json(states_path, states)
            accounting.snapshot(); summarize(root, metadata, accounting, states)
    def repair(entry):
        from src.evaluate import evaluate
        runner_spec = importlib.util.spec_from_file_location('experiment_runner', ROOT/'run.py')
        runner = importlib.util.module_from_spec(runner_spec); runner_spec.loader.exec_module(runner)
        task_id = entry['id']; destination = root/entry['task_name']
        command = json.loads((destination/'audit'/'command.json').read_text())
        run_args = runner.build_argparser().parse_args(command[2:])
        run_args.codex_provider_base_url = f'http://127.0.0.1:{server.server_port}/{task_id}/api/v1'
        os.environ['MDBENCH_LOCAL_GATEWAY_TOKEN'] = 'local-experiment-only'
        # Profile creation strips inherited provider auth; gateway always owns .env key.
        previous = destination/'probe'
        if previous.exists(): shutil.move(str(previous),str(destination/('probe-before-repair-'+str(time.time_ns()))))
        performance = json.loads((destination/'performance.json').read_text())
        model = json.loads((destination/'saved_checkpoint'/'model.json').read_text())
        started = time.monotonic()
        performance['evaluation'] = evaluate(run_args,run_args.answer,destination/'submission.txt',model)
        performance['evaluation_seconds'] = time.monotonic()-started
        performance['total_seconds'] = performance['agent_seconds']+performance['evaluation_seconds']
        valid = performance['evaluation']['phenomenal']['ok'] and all(p['ok'] for p in performance['evaluation']['mechanism_probes'])
        performance['ok'] = valid
        performance.pop('error',None);performance.pop('error_type',None)
        atomic_json(destination/'performance.json',performance)
        with state_lock:
            states[task_id].update(status='completed' if valid else 'failed',returncode=0 if valid else 1,
                                   ended=utc(),repaired=True,wall_seconds=performance['total_seconds'])
            persist()
        return {'task_id':task_id,'infrastructure_error':False}
    def work(entry):
        task_id = entry['id']; destination = root / entry['task_name']
        started = time.monotonic()
        if args.phase == 'repair':
            return repair(entry)
        with state_lock:
            states[task_id] = dict(task_name=entry['task_name'], status='exporting', started=utc(), save_path=str(destination))
            persist()
        try:
            source = ROOT / entry['path']
            task = load_task(source, validate=False)
            with state_lock: states[task_id]['expected_probe_count'] = len(task.mechanism_probes)
            audit = destination / 'audit'; audit.mkdir(parents=True, exist_ok=True)
            paths = export_task(task, audit / 'source_data',
                                force=True, seed=metadata['seed'] % (2**32),
                                train_samples=512, id_test_samples=512, ood_test_samples=512)
            if accounting.stop.is_set(): raise RuntimeError('Circuit breaker stopped dispatch')
            destination.mkdir(parents=True, exist_ok=True)
            atomic_json(audit / 'task_metadata.json', dict(**entry, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
            command = [sys.executable, str(ROOT / 'run.py'), '--problem-file', str(paths['problem']),
                '--answer', str(paths['answer']),
                '--save-path', str(destination), '--timeout', str(metadata['timeout_seconds']),
                '--probe-timeout', '120', '--probe-workers', '2', '--feedback-workers', '2',
                '--codex-profile', 'openrouter', '--codex-text-only', '--codex-model', metadata['model'],
                '--codex-reasoning-effort', 'high', '--codex-provider-env-key', 'MDBENCH_LOCAL_GATEWAY_TOKEN',
                '--codex-provider-base-url', f'http://127.0.0.1:{server.server_port}/{task_id}/api/v1',
                '--codex-private-root', str(ROOT), '--codex-blocked-tools', str(tools),
                '--codex-tool-path', str(tools)+':'+str(ROOT/'venv/bin')+':/usr/local/bin:/usr/bin:/bin',
                '--persist-runtime']
            atomic_json(audit / 'command.json', command)
            env = gateway_environment('MDBENCH_LOCAL_GATEWAY_TOKEN', 'local-experiment-only',
                                      path_prefix=(tools, ROOT / 'venv/bin'))
            with (audit / 'runner.stdout.txt').open('w') as stdout, (audit / 'runner.stderr.txt').open('w') as stderr:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
                with state_lock:
                    processes[task_id] = process
                    states[task_id].update(status='running', pid=process.pid); persist()
                code = process.wait(timeout=metadata['timeout_seconds'] + 360)
            performance = json.loads((destination / 'performance.json').read_text())
            with state_lock:
                states[task_id].update(status='completed' if code == 0 else 'failed', returncode=code,
                    ended=utc(), wall_seconds=time.monotonic()-started, error=performance.get('error'))
            logging.logger.info(tag2ansi(f'[green] {task_id}: exit={code}, {time.monotonic()-started:.1f}s'))
            return dict(task_id=task_id, infrastructure_error=is_infrastructure_failure(performance, destination))
        except Exception as exc:
            stop_codex_processes(destination)
            if (active := processes.get(task_id)) is not None and active.poll() is None:
                try:
                    os.killpg(active.pid, signal.SIGINT)
                    active.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(active.pid, signal.SIGKILL); active.wait()
                except ProcessLookupError: pass
            with state_lock: states[task_id].update(status='failed', ended=utc(), error=clean_ansi(str(exc)), wall_seconds=time.monotonic()-started)
            logging.logger.error('%s failed: %s', task_id, exc)
            return dict(task_id=task_id, infrastructure_error=True)
        finally:
            with state_lock: processes.pop(task_id, None); persist()
    try:
        logging.logger.info('Selected families %s; phase=%s jobs=%s concurrency=%s', metadata['families'], args.phase, len(selected), args.concurrency)
        snapshot = root / 'source_snapshots' / (utc().replace(':', '-') + '-' + args.phase)
        snapshot.mkdir(parents=True, exist_ok=True)
        relative_sources = ['run.py', 'run/openrouter_experiment.py'] + [str(p.relative_to(ROOT)) for p in (ROOT / 'src').rglob('*.py')]
        for relative in relative_sources:
            shutil.copy2(ROOT / relative, snapshot / relative.replace('/', '_'))
        atomic_json(snapshot / 'versions.json', {'python':sys.version, 'platform':platform.platform(),
                    'dependencies':{name:importlib.metadata.version(name) for name in ('numpy','sympy','scipy','requests')},
                    'codex':subprocess.check_output(['codex','--version'],text=True).strip(),
                    'sha256':{relative:hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() for relative in relative_sources}})
        atomic_json(root / 'launch.json', dict(time=utc(), phase=args.phase, concurrency=args.concurrency, controller_pid=os.getpid(), key_source=str(ROOT / '.env')))
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            iterator = iter(selected); futures = set(); consecutive_errors = 0
            def fill():
                while len(futures) < args.concurrency and not accounting.stop.is_set():
                    entry = next(iterator, None)
                    if entry is None: break
                    futures.add(pool.submit(work, entry))
            fill()
            while futures:
                finished, _ = wait(futures, timeout=10, return_when=FIRST_COMPLETED)
                for future in finished:
                    futures.remove(future); result = future.result()
                    consecutive_errors = consecutive_errors + 1 if result['infrastructure_error'] else 0
                    if consecutive_errors >= 3: accounting.trip('three_consecutive_infrastructure_errors')
                if accounting.stop.is_set():
                    with state_lock:
                        for task_id, process in processes.items():
                            stop_codex_processes(root/states[task_id]['task_name'])
                            try: os.killpg(process.pid, signal.SIGINT)
                            except ProcessLookupError: pass
                persist(); fill()
        if args.phase in ('smoke', 'repair'):
            completed = [states[e['id']] for e in selected]
            if accounting.stop.is_set() or any(not (root/e['task_name']/'saved_checkpoint'/'codex.session.jsonl').exists()
                                             or not (root/e['task_name']/'submission.txt').exists()
                                             or not list((root/e['task_name']/'probe').glob('*/reply.txt'))
                                             or 'evaluation' not in json.loads((root/e['task_name']/'performance.json').read_text())
                                             for e in selected):
                raise RuntimeError('Smoke failed to create resumable model artifacts; do not launch batch.')
            if args.phase == 'repair' and len(states)>2 and (root/'budget_plan.json').exists():
                atomic_json(root/'repair.done.json',dict(time=utc(),stopped=False,reason=None))
                return
            mean = sum(accounting.task_cost.get(e['id'], 0) for e in selected)/len(selected)
            # Three times observed cost, plus a floor for harder problems.
            conservative = max(0.03, 3 * max(accounting.task_cost.get(e['id'], 0) for e in selected))
            remaining = [e for e in metadata['tasks'] if states.get(e['id'], {}).get('status') not in ('completed', 'failed')]
            count = min(len(remaining), max(0, int((metadata['budget_usd']-accounting.cost)/conservative)))
            # Round-robin shuffled families, preserving balance if budget reduces N.
            queues = [[e for e in remaining if e['task_name'].startswith(f+' - ')] for f in metadata['families']]
            balanced = []
            while any(queues):
                for queue in queues:
                    if queue: balanced.append(queue.pop(0))
            plan = dict(time=utc(), smoke_mean_usd=mean, conservative_per_task_usd=conservative,
                expected_total_usd=accounting.cost+len(remaining)*mean,
                conservative_total_usd=accounting.cost+count*conservative,
                selected_task_ids=[e['id'] for e in balanced[:count]],
                task_count_including_smoke=count+len(selected), budget_usd=metadata['budget_usd'],
                observed_wall_seconds=[s['wall_seconds'] for s in completed])
            atomic_json(root/'budget_plan.json', plan); logging.logger.info('Budget plan %s', plan)
        atomic_json(root/(args.phase+'.done.json'), dict(time=utc(), stopped=accounting.stop.is_set(), reason=accounting.reason))
    finally:
        persist(); server.shutdown(); server.server_close(); session.close()
        for sig, previous in previous_signals.items(): signal.signal(sig, previous)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--phase', choices=('smoke', 'repair', 'batch'), required=True)
    parser.add_argument('--concurrency', type=int, default=10)
    experiment(parser.parse_args())
