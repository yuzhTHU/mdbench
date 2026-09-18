"""Concurrent file-upload feedback service with bounded content-addressed cache.

POST /evaluate multipart/form-data with problem, train_data, submission files.
GET /health returns a protocol marker for runner startup/reuse checks.
"""
from __future__ import annotations
import argparse
from collections import OrderedDict
from concurrent.futures import Future
from email.parser import BytesParser
from email.policy import default
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import threading
import numpy as np
from .scoring import feedback

PROTOCOL = 'mdbench-proposal-v1'
MAX_UPLOAD = 32 * 1024 * 1024

class FeedbackCache:
    """LRU completed results and single-flight coalescing of duplicate requests."""
    def __init__(self, max_entries=128, workers=4):
        if max_entries < 1 or workers < 1: raise ValueError('Cache size and workers must be positive.')
        self.max_entries = max_entries
        self.results = OrderedDict()
        self.pending = {}
        self.lock = threading.Lock()
        self.slots = threading.BoundedSemaphore(workers)

    def evaluate(self, files):
        digest = hashlib.sha256()
        for key in ('problem', 'train_data', 'submission'):
            data = files[key]
            digest.update(len(data).to_bytes(8, 'big'))
            digest.update(data)
        key = digest.hexdigest()
        with self.lock:
            if key in self.results:
                self.results.move_to_end(key)
                return self.results[key]
            owner = key not in self.pending
            future = self.pending.setdefault(key, Future())
        if not owner: return future.result()
        try:
            with self.slots:
                problem = json.loads(files['problem'])
                train = np.load(io.BytesIO(files['train_data']), allow_pickle=False)
                result = feedback(problem, train, files['submission'].decode('utf-8'))
        except Exception as exc:
            result = {'ok': False, 'error': str(exc), 'error_type': type(exc).__name__}
        with self.lock:
            self.results[key] = result
            while len(self.results) > self.max_entries: self.results.popitem(last=False)
            self.pending.pop(key)
            future.set_result(result)
        return result


def parse_upload(body: bytes, content_type: str) -> dict[str, bytes]:
    if not content_type.startswith('multipart/form-data') or '\r' in content_type or '\n' in content_type:
        raise ValueError('Use multipart/form-data with three file fields.')
    message = BytesParser(policy=default).parsebytes(
        f'Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n'.encode() + body)
    if not message.is_multipart(): raise ValueError('Missing multipart boundary.')
    files = {}
    for part in message.iter_parts():
        name = part.get_param('name', header='content-disposition')
        if name not in ('problem', 'train_data', 'submission') or name in files:
            raise ValueError('Expected unique fields: problem, train_data, submission.')
        data = part.get_payload(decode=True)
        if data is None: raise ValueError('Invalid file field.')
        files[name] = data
    if set(files) != {'problem', 'train_data', 'submission'}:
        raise ValueError('Upload problem JSON, train NPY and submission TXT.')
    return files


def create_server(host='127.0.0.1', port=8000, *, cache_size=128, workers=4):
    cache = FeedbackCache(cache_size, workers)
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, code, value):
            payload = json.dumps(value, allow_nan=False).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            self.send_json(200, {'ok': True, 'protocol': PROTOCOL}) if self.path == '/health' else self.send_json(404, {'error': 'Not found'})

        def do_POST(self):
            if self.path != '/evaluate':
                self.send_json(404, {'error': 'Not found'})
                return
            try:
                self.connection.settimeout(30)
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= MAX_UPLOAD: raise ValueError(f'Upload must be 1..{MAX_UPLOAD} bytes.')
                body = self.rfile.read(length)
                if len(body) != length: raise ValueError('Truncated upload.')
                files = parse_upload(body, self.headers.get('Content-Type', ''))
                result = cache.evaluate(files)
                self.send_json(200, result)
            except (ValueError, TimeoutError) as exc:
                self.send_json(400, {'ok': False, 'error': str(exc)})

        def log_message(self, *args): pass
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    server.cache = cache
    return server


def get_parser(parser=None):
    parser = parser or argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--cache-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=4)
    return parser


def main(args):
    server = create_server(args.host, args.port, cache_size=args.cache_size, workers=args.workers)
    print(json.dumps({'url': f'http://{args.host}:{server.server_port}/evaluate', 'protocol': PROTOCOL}), flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()
    return 0

if __name__ == '__main__':
    raise SystemExit(main(get_parser().parse_args()))
