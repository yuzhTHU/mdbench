"""DeepSeek Harness baseline with isolated profiles and metered API access.

The installed Harness headless profile is a one-shot interface.  The discovery
turn and every mechanism probe therefore run in separate isolated DSH homes;
probe prompts already contain the frozen submitted equations, so probes neither
mutate nor share agent state.
"""
from __future__ import annotations

import argparse
import atexit
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import math
import os
from pathlib import Path
import re
import secrets
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable

import requests
import yaml

from .codex import UsageAccounting, atomic_json, clean_ansi, load_api_key

__all__ = ["update_parser", "run", "get_ask"]

_logger = logging.getLogger(__name__)
_OPENROUTER_URL = "https://openrouter.ai/api/v1"
_DEEPSEEK_URL = "https://api.deepseek.com"
_DEFAULT_OPENROUTER_MODEL = "deepseek/deepseek-v4-flash-0731"
_DEFAULT_DEEPSEEK_MODEL = "deepseek-flash"
_gateway_lock = threading.RLock()
_gateways: dict[str, dict[str, Any]] = {}


def update_parser(parser):
    """Add DeepSeek Harness-specific command-line arguments."""
    parser.add_argument("--timeout", type=float, default=900,
                        help="Harness discovery run time limit in seconds.")
    parser.add_argument("--dsh-bin", default="dsh")
    parser.add_argument("--dsh-command", default=None,
                        help="DeepSeek Harness executable plus launcher-global options.")
    parser.add_argument("--dsh-provider", choices=("openrouter", "deepseek"),
                        default="openrouter",
                        help="API backend. OpenRouter failures never fall back automatically.")
    parser.add_argument("--dsh-model", default=_DEFAULT_OPENROUTER_MODEL,
                        help="OpenRouter model id.")
    parser.add_argument("--dsh-deepseek-model", default=_DEFAULT_DEEPSEEK_MODEL,
                        help="Official DeepSeek model id when explicitly selected.")
    parser.add_argument("--dsh-env-file", type=Path, default=None,
                        help="Credential file (default: repository .env).")
    parser.add_argument("--dsh-budget-usd", type=float, default=None,
                        help="Optional positive hard budget. Default: accounting only, no limit.")
    parser.add_argument("--dsh-max-output-tokens", type=int, default=16000)
    return parser


def _dsh_prefix(args) -> list[str]:
    configured = getattr(args, "dsh_command", None)
    command = shlex.split(configured) if configured else [getattr(args, "dsh_bin", "dsh")]
    if not command:
        raise ValueError("--dsh-command must contain an executable.")
    protected = {"--profile", "--patch", "--from-default-profile",
                 "--dump-config", "--dump-default-config"}
    if any(token in protected or any(token.startswith(name + "=") for name in protected)
           for token in command[1:]):
        raise ValueError("--dsh-command may not override the benchmark profile or patches.")
    return command


def _credential(name: str, env_file: Path | None) -> str | None:
    if value := os.environ.get(name):
        return value
    path = Path(env_file) if env_file else Path(__file__).resolve().parents[2] / ".env"
    if not path.is_file():
        return None
    assignments = [line for line in path.read_text().splitlines()
                   if re.match(rf"^\s*(?:export\s+)?{re.escape(name)}\s*=", line)]
    if not assignments:
        return None
    return load_api_key(path, name)


def _safe_environment(key_name: str, local_token: str, home: Path,
                      blocked_bin: Path) -> dict[str, str]:
    """Give DSH only a loopback token, never an upstream credential."""
    env = {key: value for key, value in os.environ.items()
           if not re.search(r"(API_KEY|SECRET|TOKEN|PASSWORD)", key, re.I)}
    env.update({key_name: local_token, "DSH_HOME": str(home),
                "DSH_PERMISSION_MODE": "workspace-write",
                "DSH_TELEMETRY_MODE": "DISABLED", "SHELL": "/bin/bash"})
    env["PATH"] = str(blocked_bin) + os.pathsep + env.get("PATH", "")
    env["NO_PROXY"] = env["no_proxy"] = "localhost,127.0.0.1,::1"
    env.pop("BASH_ENV", None)
    env.pop("ENV", None)
    return env


def _blocked_command_directory(root: Path) -> Path:
    directory = root / "blocked-bin"
    directory.mkdir()
    for name in ("mdbench", "git"):
        command = directory / name
        command.write_text(f'#!/bin/sh\necho "{name} is unavailable inside the benchmark agent" >&2\nexit 126\n')
        command.chmod(0o755)
    return directory


def _write_dsh_home(home: Path, runtime: dict[str, Any], *, probe: bool) -> Path:
    """Create a self-contained shipped-headless-compatible profile."""
    profile = home / "profiles" / "headless"
    profile.mkdir(parents=True)
    (profile / "package.json").write_text(json.dumps({
        "name": "mdbench-dsh-headless", "private": True, "dependencies": {},
        "dsh": {"profile": {"bundles": ["@deepseek-ai/dsh-base",
                                          "@deepseek-ai/dsh-headless"],
                              "patchReload": "startup"}}}, indent=2) + "\n")
    (profile / "cordis.yml").write_text("[]\n")
    (profile / "pnpm-workspace.yaml").write_text("packages:\n  - .\nnodeLinker: hoisted\nautoInstallPeers: false\n")

    disabled = ["web", "web-search-deepseek", "web-fetch-http", "tool-web", "tool-subagent",
                "tool-subagent-fork", "tool-subagent-control", "tool-workflow", "tool-ralph"]
    if probe:
        disabled += ["tool-bash", "tool-pwsh", "tool-jobs", "tool-fs",
                     "tool-fs-search", "tool-skill", "tool-todo", "tool-goal"]
    patch: list[dict[str, Any]] = [
        {"id": "agent-default-model", "config": {
            "provider": runtime["provider"], "model": runtime["model"]}},
    ] + [{"id": item, "disabled": True} for item in disabled]
    (profile / "cordis.patch.yml").write_text(yaml.safe_dump(patch, sort_keys=False))

    model = {
        "id": runtime["model"], "name": runtime["model"],
        "contextWindow": runtime["context_window"],
        "maxTokens": runtime["max_output_tokens"], "reasoning": True,
        "cost": {"input": runtime["pricing"]["prompt"] * 1_000_000,
                 "output": runtime["pricing"]["completion"] * 1_000_000,
                 "cacheRead": runtime["pricing"].get("input_cache_read",
                                                      runtime["pricing"]["prompt"]) * 1_000_000,
                 "cacheWrite": 0},
        "compat": {"thinkingFormat": "deepseek",
                   "supportsDeveloperRole": False,
                   "maxTokensField": "max_tokens"},
    }
    settings = {"llm-pi-ai": {"providers": {runtime["provider"]: {
        "displayName": "MDBench metered gateway",
        "apiKeyEnv": runtime["local_key"], "api": "openai-completions",
        "baseURL": runtime["base_url"], "models": [model]}}}}
    (home / "settings.yaml").write_text(yaml.safe_dump(settings, sort_keys=False))
    return profile / "cordis.patch.yml"


def _normalise_usage(usage: dict[str, Any]) -> dict[str, Any]:
    usage = dict(usage or {})
    if "prompt_tokens_details" not in usage:
        cached = usage.get("prompt_cache_hit_tokens")
        if cached is not None:
            usage["prompt_tokens_details"] = {"cached_tokens": cached}
    return usage


def start_chat_gateway(account: UsageAccounting, key: str, model: str, parallel: int,
                       *, upstream_base_url: str, local_token: str,
                       max_output_tokens: int, openrouter: bool,
                       retry_attempts: int = 4):
    """Proxy OpenAI Chat Completions while persisting tokens and charges."""
    slots = threading.BoundedSemaphore(parallel)
    upstream_base_url = upstream_base_url.rstrip("/")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def failure(self, code: int, message: str):
            body = json.dumps({"error": {"message": message, "type": "dsh_gateway"}}).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except OSError:
                pass

        def do_POST(self):
            if self.path.rstrip("/") != "/api/v1/chat/completions":
                return self.failure(404, "Unsupported gateway route")
            if self.headers.get("Authorization") != "Bearer " + local_token:
                return self.failure(401, "Local credential required")
            try:
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= 32 * 1024 * 1024:
                    raise ValueError("request size invalid")
                body = json.loads(self.rfile.read(size))
            except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
                return self.failure(400, str(exc))
            if body.get("model") != model:
                return self.failure(400, "Model substitution is forbidden")
            maximum = body.get("max_tokens", body.get("max_completion_tokens", max_output_tokens))
            if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
                return self.failure(400, "max_tokens must be a positive integer")
            body.pop("max_completion_tokens", None)
            body["max_tokens"] = min(maximum, max_output_tokens)
            if body.get("stream"):
                body["stream_options"] = {"include_usage": True}
            if openrouter:
                body["usage"] = {"include": True}

            slots.acquire()
            request_id = account.begin("dsh", size, body["max_tokens"])
            if request_id is None:
                slots.release()
                return self.failure(402, "Experiment budget or circuit breaker reached")
            directory = account.root / "api" / f"{request_id:06d}"
            directory.mkdir(parents=True)
            atomic_json(directory / "request.json", body)
            client = requests.Session()
            client.trust_env = False
            upstream = None
            usage: dict[str, Any] = {}
            response_id = None
            error = None
            headers_sent = False
            try:
                for attempt in range(retry_attempts):
                    upstream = client.post(upstream_base_url + "/chat/completions",
                        headers={"Authorization": "Bearer " + key,
                                 "Content-Type": "application/json",
                                 "HTTP-Referer": "https://github.com/yuzhthu/mdbench",
                                 "X-Title": "MechanismDiscoveryBench DSH"},
                        json=body, stream=True, timeout=(20, 300))
                    if upstream.status_code != 429:
                        break
                    upstream.close()
                    time.sleep(min(30, 2 ** (attempt + 1)))
                if upstream.status_code >= 400:
                    message = upstream.text.replace(key, "[REDACTED]")
                    (directory / "error.txt").write_text(message)
                    error = "upstream_http_" + str(upstream.status_code)
                    if upstream.status_code in (401, 402, 403):
                        account.trip(error)
                    usage = {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0}
                    return self.failure(upstream.status_code, message[:2000])
                self.send_response(upstream.status_code)
                self.send_header("Content-Type", upstream.headers.get("Content-Type", "text/event-stream"))
                self.send_header("Connection", "close")
                self.end_headers()
                headers_sent = True
                parse_buffer = b""
                with (directory / "response.bin").open("wb") as raw:
                    for chunk in upstream.iter_content(chunk_size=None):
                        if not chunk:
                            continue
                        raw.write(chunk)
                        parse_buffer += chunk
                        lines = parse_buffer.split(b"\n")
                        parse_buffer = lines.pop()
                        for line in lines:
                            if line.startswith(b"data: ") and line[6:] != b"[DONE]":
                                try:
                                    event = json.loads(line[6:])
                                    response_id = event.get("id", response_id)
                                    if event.get("usage"):
                                        usage = _normalise_usage(event["usage"])
                                except json.JSONDecodeError:
                                    pass
                        try:
                            self.wfile.write(chunk)
                            self.wfile.flush()
                        except OSError:
                            pass
                    os.fsync(raw.fileno())
                if parse_buffer.startswith(b"data: ") and parse_buffer[6:] != b"[DONE]":
                    try:
                        event = json.loads(parse_buffer[6:])
                        response_id = event.get("id", response_id)
                        if event.get("usage"):
                            usage = _normalise_usage(event["usage"])
                    except json.JSONDecodeError:
                        pass
                if not body.get("stream"):
                    try:
                        payload = json.loads((directory / "response.bin").read_bytes())
                        response_id = payload.get("id")
                        usage = _normalise_usage(payload.get("usage", {}))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass
            except Exception as exc:
                error = clean_ansi(str(exc)).replace(key, "[REDACTED]")
                account.record("transport_error", request_id=request_id, error=error)
                if not headers_sent:
                    self.failure(502, error)
            finally:
                if upstream is not None:
                    upstream.close()
                client.close()
                # Shutdown may already have conservatively finalized an
                # interrupted in-flight request. Keep completion idempotent.
                with account.lock:
                    if request_id in account.pending:
                        account.finish(request_id, usage, response_id, error)
                slots.release()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _openrouter_metadata(key: str, model: str) -> dict[str, Any]:
    client = requests.Session()
    client.trust_env = False
    try:
        response = client.get(_OPENROUTER_URL + "/models",
                              headers={"Authorization": "Bearer " + key}, timeout=30)
        response.raise_for_status()
        metadata = next((item for item in response.json().get("data", [])
                         if item.get("id") == model), None)
    finally:
        client.close()
    if metadata is None:
        raise ValueError(f"OpenRouter model metadata not found: {model}")
    pricing = metadata.get("pricing", {})
    for field in ("prompt", "completion"):
        try:
            value = float(pricing[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"OpenRouter has invalid {field} pricing for {model}") from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"OpenRouter has invalid {field} pricing for {model}")
    return metadata


def _deepseek_pricing() -> dict[str, float]:
    """Current V4 Flash peak/off-peak USD/token rates (conservative at boundaries)."""
    now = datetime.now(timezone.utc)
    peak = now.weekday() < 5 and (1 <= now.hour < 4 or 6 <= now.hour < 10)
    factor = 2 if peak else 1
    return {"prompt": 0.15e-6 * factor, "completion": 0.60e-6 * factor,
            "input_cache_read": 0.003e-6 * factor}


def _ensure_gateway(args, save: Path) -> dict[str, Any]:
    budget = getattr(args, "dsh_budget_usd", None)
    if budget is not None and budget <= 0:
        raise ValueError("--dsh-budget-usd must be positive when provided.")
    maximum = int(getattr(args, "dsh_max_output_tokens", 16000))
    if maximum < 1:
        raise ValueError("--dsh-max-output-tokens must be positive.")
    identity = str(Path(save).resolve())
    with _gateway_lock:
        if identity in _gateways:
            return _gateways[identity]["runtime"]
        if _gateways:
            raise RuntimeError("The DSH runtime supports one benchmark task per process.")

        requested = getattr(args, "dsh_provider", "openrouter")
        env_file = getattr(args, "dsh_env_file", None)
        if requested == "openrouter":
            key = _credential("OPENROUTER_API_KEY", env_file)
            if not key:
                raise RuntimeError("OPENROUTER_API_KEY is unavailable")
            metadata = _openrouter_metadata(
                key, getattr(args, "dsh_model", _DEFAULT_OPENROUTER_MODEL))
            model = metadata["id"]
            source = "openrouter"
            upstream = _OPENROUTER_URL
            raw_pricing = metadata["pricing"]
            pricing = {"prompt": float(raw_pricing["prompt"]),
                       "completion": float(raw_pricing["completion"]),
                       "input_cache_read": float(raw_pricing.get("input_cache_read",
                                                                  raw_pricing["prompt"]))}
            context = int(metadata.get("context_length") or 262144)
        else:
            key = _credential("DEEPSEEK_API_KEY", env_file)
            if not key:
                raise RuntimeError("DEEPSEEK_API_KEY is unavailable")
            model = getattr(args, "dsh_deepseek_model", _DEFAULT_DEEPSEEK_MODEL)
            source, upstream, pricing, context = "deepseek", _DEEPSEEK_URL, _deepseek_pricing(), 1_000_000

        root = Path(save) / "usage"
        root.mkdir(parents=True, exist_ok=True)
        if source == "openrouter":
            atomic_json(root / "model_metadata.json", metadata)
        account = UsageAccounting(root, budget, pricing)
        local_key = "MDBENCH_DSH_GATEWAY_KEY"
        local_token = secrets.token_urlsafe(32)
        server = start_chat_gateway(account, key, model,
            max(1, int(getattr(args, "probe_workers", 1))), upstream_base_url=upstream,
            local_token=local_token, max_output_tokens=maximum,
            openrouter=source == "openrouter")
        runtime = {"provider": "mdbench-gateway", "backend": source, "model": model,
                   "base_url": f"http://127.0.0.1:{server.server_port}/api/v1",
                   "local_key": local_key, "local_token": local_token,
                   "pricing": pricing, "context_window": context,
                   "max_output_tokens": maximum, "usage_file": str(root / "usage.json")}
        _gateways[identity] = {"server": server, "account": account, "runtime": runtime}
        _logger.info("DSH metered backend: provider=%s model=%s budget_usd=%s",
                     source, model, budget)
        return runtime


def _shutdown_gateways():
    with _gateway_lock:
        runtimes = list(_gateways.values())
        _gateways.clear()
    for item in runtimes:
        item["server"].shutdown()
        item["server"].server_close()
        account = item["account"]
        # ThreadingHTTPServer uses daemon request threads. If the supervised DSH
        # process is interrupted, an upstream stream can still be in flight as
        # Python exits. Persist its reserved upper bound instead of silently
        # under-reporting usage.
        with account.lock:
            pending = list(account.pending)
        for request_id in pending:
            with account.lock:
                if request_id in account.pending:
                    account.finish(request_id, {}, None, "gateway_shutdown_incomplete_request")
        account.snapshot()


atexit.register(_shutdown_gateways)


def _invoke(command: list[str], cwd: Path, env: dict[str, str], timeout: float,
            stdout: Path, stderr: Path, process_log: Path) -> dict[str, Any]:
    if timeout <= 0:
        raise ValueError("Timeout must be positive.")
    start = time.monotonic()
    timed_out = False
    with stdout.open("w") as output, stderr.open("w") as error:
        process = subprocess.Popen(command, stdout=output, stderr=error, text=True,
                                   cwd=cwd, env=env, start_new_session=True)
        identity = None
        try:
            identity = Path(f"/proc/{process.pid}/stat").read_text().split()[21]
        except OSError:
            pass
        process_log.write_text(json.dumps({"pid": process.pid, "start_ticks": identity,
                                           "cwd": str(cwd)}, indent=2) + "\n")
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            for sig, grace in ((signal.SIGINT, 5), (signal.SIGTERM, 3), (signal.SIGKILL, 3)):
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    process.wait(timeout=grace)
                    break
                except subprocess.TimeoutExpired:
                    continue
    stderr.write_text(clean_ansi(stderr.read_text()))
    return {"returncode": process.returncode, "timed_out": timed_out,
            "elapsed_seconds": time.monotonic() - start}


def _run_headless(args, prompt: str, output: Path, runtime: dict[str, Any],
                  timeout: float, *, probe: bool) -> tuple[str, dict[str, Any], Path]:
    output.mkdir(parents=True, exist_ok=True)
    audit = output / "audit"
    audit.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mdbench-dsh-") as directory:
        root = Path(directory)
        home, workspace = root / "home", root / "workspace"
        home.mkdir()
        workspace.mkdir()
        blocked = _blocked_command_directory(root)
        patch = _write_dsh_home(home, runtime, probe=probe)
        env = _safe_environment(runtime["local_key"], runtime["local_token"], home, blocked)
        command = [*_dsh_prefix(args), "--profile", "headless", "--patch", str(patch), prompt]
        status = _invoke(command, workspace, env, timeout, audit / "stdout.txt",
                         audit / "stderr.txt", audit / "process.json")
        (audit / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        text = clean_ansi((audit / "stdout.txt").read_text()).strip()
        sessions = sorted(path for path in home.rglob("*.jsonl*") if path.is_file())
        checkpoint = output / ("dsh.session.jsonl.zstd" if any(
            path.name.endswith(".zstd") for path in sessions) else "dsh.session.jsonl")
        if not sessions:
            raise RuntimeError(f"DSH persisted no session; see {audit}")
        # A fresh headless home owns exactly one top-level session.  Subagent
        # logs are disabled by the profile patch.
        shutil.copy2(max(sessions, key=lambda path: path.stat().st_size), checkpoint)
        if status["returncode"] != 0 and not (status["timed_out"] and text):
            raise RuntimeError(f"DSH headless run failed: {status}; see {audit}")
        if not text:
            raise RuntimeError(f"DSH returned no final answer; see {audit}")
        return text, status, checkpoint


def _prompt(args, feedback_server_url: str) -> str:
    return f'''# Objective
Reconstruct the scientific mechanism that generated the observations in
problem.json and train.npy and maxmimize the feedback score provided by {feedback_server_url}.

The primary goal is a mechanism model, not merely a phenomenological model. A
phenomenological model can fit the observable input-target relation directly;
while your mechanism model must instead describe scientifically meaningful
unobserved internal components, states, or activities and how they are organized
so that their equations generate the observed relation; otherwise, even a
perfectly accurate direct fit is insufficient.

Please ensure that your mechanistic model fully describes the underlying mechanisms
of the observations. Upon completion of the run, you will be asked several questions
regarding your proposed mechanistic model to verify your understanding of the underlying
mechanisms that generated the data.

You will have {args.timeout} seconds to iteratively refine your mechanistic model and
maximize the feedback score.

# Provided data
- Only observed variables are provided.
- train.npy has shape (variables, samples).
- problem.json data_columns gives the row order; problem.json variables gives
  each observed variable's role, scientific meaning, and unit when available.
- Scientific Python is available at {sys.executable} for numpy/sympy analysis.

# Model requirements
- Submit a self-contained system of algebraic equations involving the supplied
  inputs, the target, and scientifically meaningful unobserved internal variables.
- The equations must jointly and uniquely solve the target and every introduced
  internal variables as explicit expressions of the supplied inputs.
- Do not leave free symbolic parameters. Estimate necessary constants and define
  each one numerically as an equation in your model (for example, k = 1.2345).
- Use only ordinary algebra, powers (^ or **), sqrt, exp, log, trigonometric
  functions, and abs. Do not use differential equations.
- Prefer the smallest scientifically coherent mechanism that explains the data.
  Use variable meanings, units, scaling, and numerical behavior to distinguish
  causal/mechanistic hypotheses from curve fits.

# Investigation
Inspect the metadata and data, formulate candidate mechanisms, test their
observable consequences using the provided URL, and refine the best mechanism.
Write a valid initial submission.txt immediately, then keep it updated while
improving it within {args.timeout} seconds so it survives a time-limit interruption.

# Get Feedback
The provided URL give an objective accuracy feedback that you have to maximize. To
use this URL, you should keep the environment-provided HTTP proxy enabled and POST
three multipart file fields to the URL, including problem (problem.json), train_data
(train.npy), and submission (submission.txt). For example, you can use Python:
session.post({feedback_server_url}, files={{
    "problem": open("problem.json", "rb"),
    "train_data": open("train.npy", "rb"),
    "submission": open("submission.txt", "rb")
}}).json()
Do not set trust_env=False or override it, because the sandbox's controlled route
to the feedback endpoint is required.

submission.txt must contain only equations, with one equality per line and no
Markdown or prose. Do not use any filename other than submission.txt to submit
your answers. Upon completion of the run, ensure that the submission.txt file
represents the mechanism model that maximizes URL feedback.

# Boundaries
Do NOT modify problem.json or train.npy. Do NOT use external reference answers,
run git commands, or read shell startup files or credentials. The mdbench command
and benchmark package are intentionally unavailable; use only the feedback
endpoint described above. Do NOT inspect files outside this workspace except the
provided Python environment.
'''


def run(args, problem_file: Path, train_data_npy_file: Path,
        feedback_server_url) -> tuple[list[str], Any]:
    save = Path(args.save_path).resolve()
    save.mkdir(parents=True, exist_ok=True)
    runtime = _ensure_gateway(args, save)
    prompt = _prompt(args, feedback_server_url)
    (save / "prompt.txt").write_text(prompt)
    with tempfile.TemporaryDirectory(prefix="mdbench-dsh-input-") as directory:
        root = Path(directory)
        workspace = root / "workspace"
        workspace.mkdir()
        shutil.copy2(problem_file, workspace / "problem.json")
        shutil.copy2(train_data_npy_file, workspace / "train.npy")
        # _run_headless normally creates its own workspace.  Bind the prepared
        # inputs by temporarily using a small specialized invocation here.
        audit = save / "audit"
        audit.mkdir(exist_ok=True)
        runtime_root = root / "runtime"
        runtime_root.mkdir()
        home = runtime_root / "home"
        home.mkdir()
        blocked = _blocked_command_directory(runtime_root)
        patch = _write_dsh_home(home, runtime, probe=False)
        env = _safe_environment(runtime["local_key"], runtime["local_token"], home, blocked)
        status = _invoke([*_dsh_prefix(args), "--profile", "headless", "--patch", str(patch), prompt],
                         workspace, env, args.timeout, audit / "stdout.txt",
                         audit / "stderr.txt", audit / "process.json")
        (audit / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        final = clean_ansi((audit / "stdout.txt").read_text()).strip()
        (audit / "last_message.txt").write_text(final + "\n")
        submission_path = workspace / "submission.txt"
        submission = submission_path.read_text() if submission_path.is_file() else final
        from ..scoring import submission_formulas
        formulas = submission_formulas(clean_ansi(submission))
        sessions = sorted(path for path in home.rglob("*.jsonl*") if path.is_file())
        checkpoint_dir = save / "saved_checkpoint"
        checkpoint_dir.mkdir(exist_ok=True)
        session = checkpoint_dir / ("dsh.session.jsonl.zstd" if any(
            path.name.endswith(".zstd") for path in sessions) else "dsh.session.jsonl")
        if not sessions:
            raise RuntimeError(f"DSH persisted no session; see {audit}")
        shutil.copy2(max(sessions, key=lambda path: path.stat().st_size), session)
        session.chmod(0o444)
        if status["returncode"] != 0 and not (status["timed_out"] and formulas):
            raise RuntimeError(f"DSH discovery run failed: {status}; see {audit}")
        model = {"algorithm": "dsh", "backend": runtime["backend"],
                 "provider": runtime["provider"], "model": runtime["model"],
                 "session": str(session), "usage_file": runtime["usage_file"],
                 "dsh_command": getattr(args, "dsh_command", None), **status}
        usage_path = Path(runtime["usage_file"])
        if usage_path.is_file():
            model["usage"] = json.loads(usage_path.read_text())
        (checkpoint_dir / "model.json").write_text(json.dumps(model, indent=2) + "\n")
        return formulas, model


class DshConversation:
    def __init__(self, args, model):
        self.args, self.model = args, model

    def ask(self, prompt: str, probe_name: str, probe_description: str,
            *, output_dir: str | Path) -> str:
        del probe_name, probe_description
        output = Path(output_dir).resolve()
        (output / "prompt.txt").parent.mkdir(parents=True, exist_ok=True)
        (output / "prompt.txt").write_text(prompt)
        save = Path(self.model["session"]).resolve().parents[1]
        runtime = _ensure_gateway(self.args, save)
        text, _, _ = _run_headless(self.args, prompt, output, runtime,
                                   getattr(self.args, "probe_timeout", 120), probe=True)
        (output / "reply.txt").write_text(text + "\n")
        from ..scoring import submission_formulas
        formulas = submission_formulas(text)
        if len(formulas) != 1:
            raise ValueError("DSH probe response must contain exactly one equation.")
        return formulas[0]


def get_ask(args, checkpoint: Any) -> Callable:
    return DshConversation(args, checkpoint).ask
