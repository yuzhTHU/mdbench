"""Run an algorithm against an exported task and evaluate its frozen checkpoint."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit
from urllib.request import build_opener, ProxyHandler
from src.algorithms import get_algorithm, get_update_parser, list_algorithms
from src.algorithms.codex import clean_ansi
from src.evaluate import evaluate
from src.feedback_server import PROTOCOL, create_server


def _clean(value):
    if isinstance(value, str): return clean_ansi(value)
    if isinstance(value, dict): return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_clean(v) for v in value]
    return value


def build_argparser(argv=None):
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument('--algorithm', default='codex', choices=list_algorithms())
    known, _ = preliminary.parse_known_args(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--algorithm', default=known.algorithm, choices=list_algorithms())
    parser.add_argument('--problem-file', type=Path, required=True)
    parser.add_argument('--train-data-npy-file', type=Path, required=True)
    parser.add_argument('--answer', type=Path, required=True)
    parser.add_argument('--save-path', default=None)
    parser.add_argument('--save-dir', default='logs/run')
    parser.add_argument('--feedback-host', default='127.0.0.1')
    parser.add_argument('--feedback-port', type=int, default=0, help='0 selects a free local port.')
    parser.add_argument('--feedback-server-url', default=None, help='Reuse a running server at its /evaluate URL.')
    parser.add_argument('--feedback-workers', type=int, default=4)
    parser.add_argument('--feedback-cache-size', type=int, default=128)
    parser.add_argument('--probe-timeout', type=float, default=120)
    parser.add_argument('--probe-workers', type=int, default=4)
    update = get_update_parser(known.algorithm)
    if update: parser = update(parser)
    return parser


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


def main(args):
    for path in (args.problem_file, args.train_data_npy_file, args.answer):
        if not path.is_file(): raise FileNotFoundError(path)
    args.save_path = str(Path(args.save_path or (Path(args.save_dir) / datetime.now().strftime('%Y%m%d-%H%M%S-%f'))).resolve())
    save = Path(args.save_path)
    save.mkdir(parents=True, exist_ok=True)
    performance = {'algorithm': args.algorithm}
    start = time.monotonic()
    exit_code = 0
    try:
        with feedback_service(args) as url:
            run_start = time.monotonic()
            try:
                model = get_algorithm(args.algorithm)(args, args.problem_file, args.train_data_npy_file, url)
            finally:
                performance['agent_seconds'] = time.monotonic() - run_start
            model = _clean(model)
            performance['model'] = model
            (save / 'model.json').write_text(json.dumps(model, indent=2) + '\n')
            evaluation_start = time.monotonic()
            result = evaluate(args, args.answer, model['submission'], model['session'], model=model)
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
    finally:
        performance['total_seconds'] = time.monotonic() - start
        performance['ok'] = exit_code == 0
        text = json.dumps(_clean(performance), indent=2, allow_nan=False, default=str)
        (save / 'performance.json').write_text(text + '\n')
        print(text)
    return exit_code

if __name__ == '__main__':
    raise SystemExit(main(build_argparser().parse_args()))
