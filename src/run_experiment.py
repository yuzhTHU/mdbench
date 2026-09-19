"""Run an algorithm against an exported task and evaluate its frozen checkpoint."""
from __future__ import annotations
import argparse
from copy import deepcopy
from contextlib import contextmanager
from datetime import datetime
import json
import logging
import random
from pathlib import Path
import re
import shlex
import shutil
from socket import gethostname
import sys
import threading
import time
import numpy as np
from urllib.parse import urlsplit
from urllib.request import build_opener, ProxyHandler
from src.algorithms import get_algorithm, get_ask, get_update_parser, list_algorithms
from src.algorithms.codex import clean_ansi
from src.cli.flags import add_minus_flags, add_negation_flags
from src.evaluate import evaluate
from src.feedback_server import PROTOCOL, create_server
from src.validate_problem import parse_equation

SCRIPT_NAME = 'run'
_logger = logging.getLogger(f'mdbench.{SCRIPT_NAME}')


def get_parser(parser=None, argv=None):
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument('--algorithm', default='codex', choices=list_algorithms())
    args, _ = preliminary.parse_known_args(argv)
    parser = parser or argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--algorithm', default=args.algorithm, choices=list_algorithms(), help='Algorithm name.')
    parser.add_argument('--name', default=SCRIPT_NAME, help='Experiment task name used when auto-generating exp_name.')
    parser.add_argument('--exp_name', default=None, help='Experiment name. Defaults to a timestamped name.')
    parser.add_argument('--save_dir', default='./logs/run', help='Root directory for logs and run artifacts.')
    parser.add_argument('--seed', type=int, default=-1, help='Random seed. Default -1 means using current system time.')
    parser.add_argument('--save_path', default=None, help='Exact task artifact path. Normally generated from save_dir, exp_name and task_name.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose runner logging.')
    parser.add_argument('--debug', action='store_true', default=False, help='Enable verbose logging and raise caught exceptions.')
    parser.add_argument('--problem_file', type=Path, required=True)
    parser.add_argument('--answer_file', type=Path, required=True)
    parser.add_argument('--feedback_host', default='127.0.0.1')
    parser.add_argument('--feedback_port', type=int, default=0, help='0 selects a free local port.')
    parser.add_argument('--feedback_server_url', default=None, help='Reuse a running server at its /evaluate URL.')
    parser.add_argument('--feedback_workers', type=int, default=4)
    parser.add_argument('--feedback_cache_size', type=int, default=128)
    parser.add_argument('--probe_timeout', type=float, default=120)
    parser.add_argument('--probe_workers', type=int, default=1)
    if (update_parser_fn := get_update_parser(args.algorithm)):
        parser = update_parser_fn(parser)
    add_minus_flags(parser)
    add_negation_flags(parser)
    return parser


def build_argparser(argv=None):
    return get_parser(argv=argv)


def sanitize_filename(value):
    value = re.compile(r'[ <>:"/\\|?*\x00-\x1f]').sub('_', str(value).strip())
    return (value or 'unnamed')[:255]


def _task_name(answer_file):
    answer = json.loads(Path(answer_file).read_text())
    task_name = (answer.get('task') or {}).get('task_name')
    if not isinstance(task_name, str) or not task_name.strip() or Path(task_name).name != task_name or task_name in ('.', '..'):
        raise ValueError('task_name must be one nonempty directory name.')
    return task_name


def _stage_dataest(args, save):
    """Create a self-contained copy of all public and private task data."""
    root = save / 'dataest'
    root.mkdir(exist_ok=True)
    answer = json.loads(Path(args.answer_file).read_text())
    sources = {'problem': Path(args.problem_file)}
    arrays = answer.get('data') or {}
    for split in ('train', 'id_test', 'ood_test'):
        relative = arrays.get(split)
        if not isinstance(relative, str): raise ValueError(f'answer data has no {split} NPY path.')
        sources[split] = Path(args.answer_file).parent / relative
    paths = {'problem': root / 'problem.json', 'answer': root / 'answer.json',
             'train': root / 'train.npy', 'id_test': root / 'id_test.npy',
             'ood_test': root / 'ood_test.npy'}
    for name, source in sources.items():
        if not source.is_file(): raise FileNotFoundError(source)
        if source.resolve() != paths[name].resolve(): shutil.copy2(source, paths[name])
    answer['data'] = {split: f'{split}.npy' for split in ('train', 'id_test', 'ood_test')}
    paths['answer'].write_text(json.dumps(answer, indent=2) + '\n')
    return paths


def _healthy(url):
    parts = urlsplit(url)
    health = f'{parts.scheme}://{parts.netloc}/health'
    opener = build_opener(ProxyHandler({})) if parts.hostname in ('localhost', '127.0.0.1', '::1') else build_opener()
    with opener.open(health, timeout=5) as reply:
        data = json.load(reply)
    if data.get('protocol') != PROTOCOL or not data.get('ok'):
        raise RuntimeError('Feedback endpoint is not a proposal benchmark server.')


@contextmanager
def feedback_service(args):
    if args.feedback_server_url:
        _healthy(args.feedback_server_url)
        yield args.feedback_server_url
        return
    host, port = args.feedback_host, args.feedback_port
    url = f'http://{host}:{port}/evaluate'
    if port:
        try: _healthy(url)
        except OSError: pass
        else:
            yield url
            return
    server = create_server(host, port, cache_size=args.feedback_cache_size, workers=args.feedback_workers)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f'http://{host}:{server.server_port}/evaluate'
    try:
        _healthy(url)
        yield url
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def run_experiment(args):
    for path in (args.problem_file, args.answer_file):
        if not path.is_file(): raise FileNotFoundError(path)
    if args.save_path:
        save = Path(args.save_path)
    else:
        if not getattr(args, 'exp_name', None):
            raise ValueError('--exp_name is required unless --save-path is provided.')
        save = Path(args.save_dir) / args.exp_name / _task_name(args.answer_file)
    args.save_path = str(save.resolve())
    save = Path(args.save_path)
    save.mkdir(parents=True, exist_ok=True)
    submission_file = save / 'submission.txt'
    if submission_file.exists():
        raise FileExistsError(f'Run artifact already exists: {submission_file}')
    dataest = _stage_dataest(args, save)
    performance = {'algorithm': args.algorithm}
    start = time.monotonic()
    exit_code = 0
    try:
        with feedback_service(args) as url:
            run_start = time.monotonic()
            try:
                output = get_algorithm(args.algorithm)(args, dataest['problem'], dataest['train'], url)
                if not isinstance(output, tuple) or len(output) != 2:
                    raise TypeError('Algorithm run() must return a (submission, checkpoint) tuple.')
                submission, checkpoint = output
                if not isinstance(submission, list) or not submission or not all(isinstance(f, str) for f in submission):
                    raise TypeError('Algorithm submission must be a nonempty list of formula strings.')
                for formula in submission: parse_equation(formula)
                submission_file.write_text('\n'.join(submission) + '\n')
            finally:
                performance['agent_seconds'] = time.monotonic() - run_start
            evaluation_start = time.monotonic()
            algorithm_get_ask = get_ask(args.algorithm)
            result = evaluate(
                args, dataest['answer'], submission,
                lambda: algorithm_get_ask(args, deepcopy(checkpoint)))
            performance['evaluation_seconds'] = time.monotonic() - evaluation_start
            performance['evaluation'] = result
            exit_code = int(not result['phenomenal']['ok'] or any(not p['ok'] for p in result['mechanism_probes']))
    except KeyboardInterrupt:
        performance['error'] = 'Interrupted; live artifacts remain in the persistent runtime.'
        performance['error_type'] = 'KeyboardInterrupt'
        exit_code = 130
    except Exception as exc:
        performance['error'] = clean_ansi(str(exc))
        performance['error_type'] = type(exc).__name__
        exit_code = 1
        if getattr(args, 'debug', False): raise
    finally:
        performance['total_seconds'] = time.monotonic() - start
        performance['ok'] = exit_code == 0
        text = json.dumps(performance, indent=2, allow_nan=False, default=str)
        (save / 'performance.json').write_text(text + '\n')
        print(text)
    return exit_code


def main(args):
    explicit_save_path = args.save_path is not None
    if args.exp_name is None:
        if explicit_save_path:
            args.exp_name = sanitize_filename(Path(args.save_path).parent.name)
        else:
            now = datetime.now()
            args.exp_name = sanitize_filename(
                f'{now:%Y%m%d}_{args.name}_{now:%H%M%S}_{gethostname()}'
            )
    else:
        args.exp_name = sanitize_filename(args.exp_name)
    if args.debug:
        args.verbose = True
    if args.seed == -1:
        args.seed = int(datetime.now().timestamp() * 1000) % (2**32 - 1)
    random.seed(args.seed)
    np.random.seed(args.seed)

    exp_path = Path(args.save_path).parent if explicit_save_path else Path(args.save_dir) / args.exp_name
    exp_path.mkdir(parents=True, exist_ok=True)
    if args.save_path is None:
        args.save_path = str(exp_path / _task_name(args.answer_file))
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    args.invocation = ' '.join(map(shlex.quote, [sys.executable, *sys.argv]))

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[logging.FileHandler(save_path / 'info.log'), logging.StreamHandler()],
        force=True,
    )
    _logger.info('Args: %s', args)
    (save_path / 'args.json').write_text(json.dumps(vars(args), indent=2, default=str) + '\n')

    exit_code = run_experiment(args)
    _logger.info('Experiment completed. Re-run the command with %s', args.invocation)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main(get_parser(argv=sys.argv[1:]).parse_args()))
