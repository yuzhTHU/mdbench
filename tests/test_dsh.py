import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import shutil
import threading
from types import SimpleNamespace

import pytest

import src.algorithms.dsh as dsh


def test_dsh_defaults_and_command_protection():
    parser = argparse.ArgumentParser()
    dsh.update_parser(parser)
    args = parser.parse_args([])
    assert args.timeout == 900
    assert args.dsh_provider == "openrouter"
    assert args.dsh_model == "deepseek/deepseek-v4-flash-0731"
    assert args.dsh_budget_usd is None
    assert dsh._dsh_prefix(SimpleNamespace(dsh_command="custom-dsh --version")) == [
        "custom-dsh", "--version"]
    with pytest.raises(ValueError, match="profile"):
        dsh._dsh_prefix(SimpleNamespace(dsh_command="dsh --profile web"))


def test_prompt_matches_codex_benchmark_contract():
    prompt = dsh._prompt(SimpleNamespace(timeout=321), "http://127.0.0.1:8123/evaluate")
    required = [
        "Only observed variables are provided.",
        "phenomenological model can fit the observable input-target relation directly",
        "You will have 321 seconds",
        "Do not leave free symbolic parameters.",
        "Write a valid initial submission.txt immediately",
        '"problem": open("problem.json", "rb")',
        '"train_data": open("train.npy", "rb")',
        '"submission": open("submission.txt", "rb")',
        "Do not set trust_env=False",
        "Do not use any filename other than submission.txt",
        "Do NOT inspect files outside this workspace except the",
    ]
    assert all(text in prompt for text in required)
    assert prompt.count("http://127.0.0.1:8123/evaluate") == 2


def test_isolated_home_selects_gateway_and_disables_probe_tools(tmp_path):
    runtime = {"provider": "mdbench-gateway", "model": "provider/model",
               "context_window": 123456, "max_output_tokens": 4096,
               "pricing": {"prompt": 1e-6, "completion": 2e-6,
                           "input_cache_read": 0.1e-6},
               "local_key": "LOCAL_KEY", "base_url": "http://127.0.0.1:1/api/v1"}
    patch_path = dsh._write_dsh_home(tmp_path, runtime, probe=True)
    patch = patch_path.read_text()
    settings = (tmp_path / "settings.yaml").read_text()
    assert "mdbench-gateway" in patch and "provider/model" in patch
    assert "tool-bash" in patch and "disabled: true" in patch
    assert "LOCAL_KEY" in settings
    assert "provider/model" in settings
    assert "http://127.0.0.1:1/api/v1" in settings


def test_openrouter_backend_records_pricing(tmp_path, monkeypatch):
    dsh._shutdown_gateways()
    monkeypatch.setenv("OPENROUTER_API_KEY", "upstream-secret")
    metadata = {"id": "deepseek/deepseek-v4-flash-0731", "context_length": 999,
                "pricing": {"prompt": "0.000001", "completion": "0.000002",
                            "input_cache_read": "0.0000001"}}
    monkeypatch.setattr(dsh, "_openrouter_metadata", lambda key, model: metadata)

    class Server:
        server_port = 43210
        def shutdown(self): pass
        def server_close(self): pass

    captured = {}
    def start(account, key, model, parallel, **kwargs):
        captured.update(key=key, model=model, parallel=parallel, kwargs=kwargs)
        return Server()
    monkeypatch.setattr(dsh, "start_chat_gateway", start)
    args = SimpleNamespace(dsh_provider="openrouter", dsh_model=metadata["id"],
                           dsh_env_file=None, dsh_budget_usd=None,
                           dsh_max_output_tokens=1234, probe_workers=2)
    try:
        runtime = dsh._ensure_gateway(args, tmp_path)
        assert runtime["backend"] == "openrouter"
        assert runtime["model"] == metadata["id"]
        assert captured["key"] == "upstream-secret"
        usage = json.loads((tmp_path / "usage" / "usage.json").read_text())
        assert usage["input_tokens"] == usage["output_tokens"] == 0
    finally:
        dsh._shutdown_gateways()


def test_openrouter_failure_never_falls_back_to_deepseek(tmp_path, monkeypatch):
    dsh._shutdown_gateways()
    monkeypatch.setenv("OPENROUTER_API_KEY", "broken-openrouter-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setattr(dsh, "_openrouter_metadata",
                        lambda *args: (_ for _ in ()).throw(RuntimeError("unavailable")))

    args = SimpleNamespace(dsh_provider="openrouter", dsh_model="missing/model",
                           dsh_deepseek_model="deepseek-flash", dsh_env_file=None,
                           dsh_budget_usd=None, dsh_max_output_tokens=1024,
                           probe_workers=1)
    with pytest.raises(RuntimeError, match="unavailable"):
        dsh._ensure_gateway(args, tmp_path)
    assert not dsh._gateways


def test_safe_environment_does_not_forward_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    monkeypatch.setenv("UNRELATED_TOKEN", "also-secret")
    monkeypatch.setenv("SAFE_VALUE", "ok")
    blocked = tmp_path / "bin"
    blocked.mkdir()
    env = dsh._safe_environment("LOCAL_GATEWAY_KEY", "local", tmp_path / "home", blocked)
    assert "OPENROUTER_API_KEY" not in env and "UNRELATED_TOKEN" not in env
    assert env["LOCAL_GATEWAY_KEY"] == "local" and env["SAFE_VALUE"] == "ok"


def test_shutdown_accounts_for_incomplete_requests(tmp_path):
    dsh._shutdown_gateways()
    account = dsh.UsageAccounting(tmp_path / "usage", None, {
        "prompt": 1e-6, "completion": 2e-6})
    request_id = account.begin("dsh", 100, 200)

    class Server:
        def shutdown(self): pass
        def server_close(self): pass

    dsh._gateways[str(tmp_path)] = {
        "server": Server(), "account": account, "runtime": {}}
    dsh._shutdown_gateways()
    usage = json.loads((tmp_path / "usage" / "usage.json").read_text())
    assert request_id is not None
    assert usage["inflight"] == 0
    assert usage["estimated_or_uncertain_usd"] > 0


def test_real_headless_profile_against_offline_chat_provider(tmp_path):
    executable = shutil.which("dsh")
    if not executable:
        pytest.skip("DeepSeek Harness is not installed")

    class Provider(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert request["model"] == "provider/model"
            events = [
                {"id": "chat-test", "object": "chat.completion.chunk",
                 "model": "provider/model", "choices": [{"index": 0,
                 "delta": {"role": "assistant", "content": "y=x"},
                 "finish_reason": None}]},
                {"id": "chat-test", "object": "chat.completion.chunk",
                 "model": "provider/model", "choices": [{"index": 0,
                 "delta": {}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 10, "completion_tokens": 3,
                           "total_tokens": 13}},
            ]
            payload = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
            payload += "data: [DONE]\n\n"
            data = payload.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        def log_message(self, *args): pass

    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    except PermissionError:
        pytest.skip("test sandbox forbids loopback listeners")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    runtime = {"provider": "mdbench-gateway", "backend": "test",
               "model": "provider/model", "context_window": 100000,
               "max_output_tokens": 4096,
               "pricing": {"prompt": 1e-6, "completion": 2e-6},
               "local_key": "LOCAL_KEY", "local_token": "local-token",
               "base_url": f"http://127.0.0.1:{server.server_port}",
               "usage_file": str(tmp_path / "usage.json")}
    args = SimpleNamespace(dsh_command=executable)
    try:
        reply, status, checkpoint = dsh._run_headless(
            args, "Return only y=x", tmp_path / "run", runtime, 30, probe=True)
        assert reply == "y=x"
        assert status["returncode"] == 0
        assert checkpoint.is_file()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.skipif(os.environ.get("MDBENCH_DSH_ONLINE") != "1",
                    reason="opt-in test spends a small amount of provider credit")
def test_online_openrouter_dsh_and_usage_accounting(tmp_path):
    if not shutil.which("dsh"):
        pytest.skip("DeepSeek Harness is not installed")
    args = SimpleNamespace(
        dsh_command=None, dsh_bin="dsh", dsh_provider="openrouter",
        dsh_model="deepseek/deepseek-v4-flash-0731", dsh_env_file=None,
        dsh_budget_usd=0.02, dsh_max_output_tokens=128, probe_workers=1)
    try:
        runtime = dsh._ensure_gateway(args, tmp_path)
        reply, status, checkpoint = dsh._run_headless(
            args, "Reply with exactly: y=x", tmp_path / "online", runtime, 90, probe=True)
        assert "y=x" in reply and status["returncode"] == 0 and checkpoint.is_file()
        usage = json.loads(Path(runtime["usage_file"]).read_text())
        assert usage["request_count"] >= 1
        assert usage["input_tokens"] > 0 and usage["output_tokens"] > 0
        assert usage["charged_usd"] > 0
    finally:
        dsh._shutdown_gateways()
