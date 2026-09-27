"""One-turn, tool-free Codex answers to mechanism probes for a supplied P.

Each probe starts a fresh ephemeral Codex process in an empty temporary working
directory.  All agent tools and web search are disabled.  The CLI writes the
single final response to ``answer.txt``; only that file is read back and scored.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable

from .codex import (_command_option, _codex_prefix, _isolated_home, _invoke,
                    _selected_model, _selected_profile, clean_ansi)
from .pysr_and_llm_direct import _extract_formula
from ..scoring import submission_formulas

__all__ = ["update_parser", "run", "get_ask"]


def update_parser(parser):
    parser.add_argument(
        "--submission-source-dir", type=Path, required=True,
        help="Directory containing the exact phenomenal submission.txt to freeze.")
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument(
        "--codex-command", default=None,
        help='Codex executable plus global options, e.g. "codex --profile lab".')
    parser.add_argument("--codex-model", default=None)
    parser.add_argument("--codex-reasoning-effort", default=None)
    parser.add_argument("--codex-http-proxy", default="http://127.0.0.1:6789")
    return parser


def run(args, problem_file: Path, train_data_npy_file: Path,
        feedback_server_url) -> tuple[list[str], Any]:
    """Freeze the caller-supplied exact phenomenal equation without discovery."""
    del problem_file, train_data_npy_file, feedback_server_url
    source = Path(args.submission_source_dir).resolve()
    submission = source / "submission.txt"
    if not submission.is_file():
        raise FileNotFoundError(f"Phenomenal submission not found: {submission}")
    formulas = submission_formulas(submission.read_text())
    if len(formulas) != 1:
        raise ValueError("Direct Ask based on P requires exactly one phenomenal equation.")
    # Resolve configuration now so an unusable local Codex setup fails before
    # the benchmark begins its independent probe evaluations.
    _codex_prefix(args)
    return formulas, {
        "algorithm": "direct_ask_codex",
        "formula": formulas,
        "codex_model": _selected_model(args),
        "codex_profile": _selected_profile(args),
        "codex_command": getattr(args, "codex_command", None),
    }


def _command(args, answer_file: Path) -> list[str]:
    prefix = _codex_prefix(args)
    command = [
        *prefix, "exec", "--ephemeral", "--json", "--skip-git-repo-check",
        "--ignore-rules", "--sandbox", "read-only", "--color", "never",
        "-c", 'approval_policy="never"',
        "-c", 'web_search="disabled"',
        "-c", "features.apps=false",
        "-c", "features.plugins=false",
        "-c", "features.hooks=false",
        "-c", "features.multi_agent=false",
        "-c", "features.memories=false",
        "-c", "features.shell_snapshot=false",
        "-c", "features.shell_tool=false",
        "-c", "features.unified_exec=false",
        "-c", "features.code_mode=false",
        "-c", "features.code_mode_host=false",
        "-c", "features.view_image=false",
        "-c", "features.image_generation=false",
        "-c", "features.browser_use=false",
        "-c", "features.computer_use=false",
        "-c", "features.sleep_tool=false",
        # Codex's model transport may require the host network proxy.  This does
        # not expose network tools to the agent: every such tool is disabled,
        # web_search is disabled, and the execution sandbox remains read-only.
        "-c", "features.network_proxy=true",
        "-c", "sandbox_workspace_write.network_access=false",
    ]
    if (getattr(args, "codex_model", None)
            and not _command_option(prefix, "--model", "-m")):
        command += ["--model", args.codex_model]
    if effort := getattr(args, "codex_reasoning_effort", None):
        command += ["-c", f"model_reasoning_effort={json.dumps(effort)}"]
    command += ["-o", str(answer_file), "-"]
    return command


class DirectCodexConversation:
    def __init__(self, args, checkpoint: Any):
        self.args = args
        self.checkpoint = checkpoint

    def ask(self, prompt: str, probe_name: str, probe_description: str,
            *, output_dir: str | Path) -> str:
        del probe_description
        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        audit = output / "audit"
        audit.mkdir(exist_ok=True)
        wrapped_prompt = f"""{prompt}

This is a single-turn, tool-free evaluation. Do not access the filesystem,
network, shell, external sources, or any other tools. Produce exactly one final
answer containing only the requested plain-text equation and no reasoning,
Markdown, or prose. The Codex CLI will save that final answer to answer.txt in
the current empty working directory.
"""
        (output / "prompt.txt").write_text(wrapped_prompt)
        with tempfile.TemporaryDirectory(prefix="mdbench-direct-codex-") as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            if any(workspace.iterdir()):
                raise RuntimeError("Direct Codex workspace must start empty.")
            home = root / "home"
            safe_env = {key: value for key, value in os.environ.items()
                        if not re.search(r"(API_KEY|SECRET|TOKEN|PASSWORD)", key, re.I)}
            if proxy := getattr(self.args, "codex_http_proxy", None):
                safe_env.update({name: proxy for name in (
                    "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")})
            env = _isolated_home(
                home, _selected_profile(self.args), base_env=safe_env, copy_auth=True)
            # Avoid exposing inherited shell hooks; provider authentication is
            # supplied only through the isolated Codex home/configuration.
            env.pop("BASH_ENV", None)
            env.pop("ENV", None)
            env["SHELL"] = "/bin/bash"
            answer = workspace / "answer.txt"
            status = _invoke(
                _command(self.args, answer), wrapped_prompt, workspace, env,
                audit / "codex.events.jsonl", audit / "stderr.txt",
                getattr(self.args, "probe_timeout", 120),
                stdout=audit / "stdout.txt", process_log=audit / "process.json")
            final_entries = sorted(path.name for path in workspace.iterdir())
            (audit / "status.json").write_text(json.dumps(
                {**status, "initial_workspace_entries": [],
                 "final_workspace_entries": final_entries}, indent=2) + "\n")
            if status["timed_out"] or status["returncode"] != 0 or not answer.is_file():
                raise RuntimeError(f"Direct Codex probe failed: {status}; see {output}")
            text = clean_ansi(answer.read_text()).strip()
        (output / "reply.txt").write_text(text + "\n")
        formula = _extract_formula(text, probe_name)
        (output / "formula.txt").write_text(formula + "\n")
        return formula


def get_ask(args, checkpoint: Any) -> Callable:
    return DirectCodexConversation(args, checkpoint).ask
