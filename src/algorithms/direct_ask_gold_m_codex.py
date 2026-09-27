"""One-turn Codex answers to description-only probes given the exact mechanism.

The model sees the gold mechanism equations, but the requested probe symbol is
deliberately omitted from the question and output contract.  Codex returns only
the right-hand-side expression; this adapter restores the hidden probe name
after the isolated turn so the standard benchmark evaluator can score it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable

import sympy as sp

from .codex import (_command_option, _codex_prefix, _isolated_home, _invoke,
                    _selected_model, _selected_profile, clean_ansi)
from ..scoring import submission_formulas
from ..validate_problem import (expand_expression, parse_equation,
                                parse_expression, split_equation,
                                task_from_dict, validate_task)

__all__ = ["update_parser", "run", "get_ask"]


def update_parser(parser):
    parser.add_argument(
        "--gold-answer-file", type=Path, required=True,
        help="Private exported answer whose task supplies the exact P and Gold-M.")
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--codex-command", default=None)
    parser.add_argument("--codex-model", default=None)
    parser.add_argument("--codex-reasoning-effort", default=None)
    parser.add_argument("--codex-http-proxy", default="http://127.0.0.1:6789")
    return parser


def run(args, problem_file: Path, train_data_npy_file: Path,
        feedback_server_url) -> tuple[list[str], Any]:
    """Freeze exact P for scoring and Gold-M for description-only probe turns."""
    del problem_file, train_data_npy_file, feedback_server_url
    answer_file = Path(args.gold_answer_file).resolve()
    raw = json.loads(answer_file.read_text())
    task = task_from_dict(raw["task"])
    validate_task(task)
    expression = expand_expression(
        task.phenomenal_model, task, lhs=task.target.name)
    submission = [f"{task.target.name} = {sp.sstr(expression)}"]
    parse_equation(submission[0])
    _codex_prefix(args)  # Fail before evaluation if the CLI is unavailable.
    return submission, {
        "algorithm": "direct_ask_gold_m_codex",
        "task_name": task.task_name,
        "gold_mechanism": [item.formula_str for item in task.mechanism_model],
        "answer_variables": [
            {"name": variable.name, "description": variable.description,
             "unit": variable.unit}
            for variable in task.by_role("input", "auxiliary")
        ],
        "codex_model": _selected_model(args),
        "codex_profile": _selected_profile(args),
        "codex_command": getattr(args, "codex_command", None),
        "probe_target_disclosure": "description_only",
    }


def _command(args, answer_file: Path) -> list[str]:
    prefix = _codex_prefix(args)
    command = [
        *prefix, "exec", "--ephemeral", "--json", "--skip-git-repo-check",
        "--ignore-rules", "--sandbox", "read-only", "--color", "never",
        "-c", 'approval_policy="never"', "-c", 'web_search="disabled"',
        "-c", "features.apps=false", "-c", "features.plugins=false",
        "-c", "features.hooks=false", "-c", "features.multi_agent=false",
        "-c", "features.memories=false", "-c", "features.shell_snapshot=false",
        "-c", "features.shell_tool=false", "-c", "features.unified_exec=false",
        "-c", "features.code_mode=false", "-c", "features.code_mode_host=false",
        "-c", "features.view_image=false", "-c", "features.image_generation=false",
        "-c", "features.browser_use=false", "-c", "features.computer_use=false",
        "-c", "features.sleep_tool=false", "-c", "features.network_proxy=true",
        "-c", "sandbox_workspace_write.network_access=false",
    ]
    if (getattr(args, "codex_model", None)
            and not _command_option(prefix, "--model", "-m")):
        command += ["--model", args.codex_model]
    if effort := getattr(args, "codex_reasoning_effort", None):
        command += ["-c", f"model_reasoning_effort={json.dumps(effort)}"]
    command += ["-o", str(answer_file), "-"]
    return command


def _rhs_from_reply(text: str) -> str:
    """Accept a requested bare expression, tolerating one accidental equation."""
    cleaned = clean_ansi(text).strip()
    fenced = re.fullmatch(r"```(?:text|plaintext|python)?\s*(.*?)\s*```", cleaned,
                          flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()
    if "=" not in cleaned and cleaned and "\n" not in cleaned:
        rhs = cleaned
    else:
        formulas = submission_formulas(cleaned)
        if len(formulas) != 1:
            raise ValueError(
                "Gold-M Codex response must contain one bare expression (or one equation).")
        _, rhs = split_equation(formulas[0])
    parse_expression(rhs)
    return rhs


class GoldMDirectCodexConversation:
    def __init__(self, args, checkpoint: Any):
        self.args = args
        self.checkpoint = checkpoint

    def ask(self, prompt: str, probe_name: str, probe_description: str,
            *, output_dir: str | Path) -> str:
        # The evaluator's standard prompt contains the probe symbol, so it must
        # never be forwarded in this description-only experimental condition.
        del prompt
        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        audit = output / "audit"
        audit.mkdir(exist_ok=True)
        equations = "\n".join(self.checkpoint["gold_mechanism"])
        variables = "\n".join(
            f'- {item["name"]}: {item["description"]}'
            + (f' [{item["unit"]}]' if item.get("unit") else "")
            for item in self.checkpoint["answer_variables"])
        model_prompt = f"""Gold mechanism model (equations only):
{equations}

Input and auxiliary variables allowed in the answer:
{variables}

Mechanism probe description:
{probe_description}

Identify the quantity described by the probe from the gold mechanism model and
derive it as a function only of the allowed input and auxiliary variables.
The requested quantity's variable name is intentionally omitted. Return only
the right-hand-side mathematical expression: no left-hand side, equals sign,
reasoning, Markdown, or prose. Do not use the filesystem, network, shell,
external sources, or any tools.
"""
        (output / "prompt.txt").write_text(model_prompt)
        (output / "prompt_metadata.json").write_text(json.dumps({
            "probe_description": probe_description,
            "probe_name_forwarded_to_model": False,
            "output_contract": "right_hand_side_only",
        }, indent=2) + "\n")
        with tempfile.TemporaryDirectory(prefix="mdbench-gold-m-codex-") as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            home = root / "home"
            safe_env = {key: value for key, value in os.environ.items()
                        if not re.search(r"(API_KEY|SECRET|TOKEN|PASSWORD)", key, re.I)}
            if proxy := getattr(self.args, "codex_http_proxy", None):
                safe_env.update({name: proxy for name in (
                    "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")})
            env = _isolated_home(
                home, _selected_profile(self.args), base_env=safe_env, copy_auth=True)
            env.pop("BASH_ENV", None)
            env.pop("ENV", None)
            env["SHELL"] = "/bin/bash"
            answer = workspace / "answer.txt"
            status = _invoke(
                _command(self.args, answer), model_prompt, workspace, env,
                audit / "codex.events.jsonl", audit / "stderr.txt",
                getattr(self.args, "probe_timeout", 120),
                stdout=audit / "stdout.txt", process_log=audit / "process.json")
            final_entries = sorted(path.name for path in workspace.iterdir())
            (audit / "status.json").write_text(json.dumps(
                {**status, "initial_workspace_entries": [],
                 "final_workspace_entries": final_entries}, indent=2) + "\n")
            if status["timed_out"] or status["returncode"] != 0 or not answer.is_file():
                raise RuntimeError(f"Gold-M Codex probe failed: {status}; see {output}")
            text = clean_ansi(answer.read_text()).strip()
        (output / "reply.txt").write_text(text + "\n")
        rhs = _rhs_from_reply(text)
        formula = f"{probe_name} = {rhs}"
        parse_equation(formula)
        (output / "formula.txt").write_text(formula + "\n")
        return formula


def get_ask(args, checkpoint: Any) -> Callable:
    return GoldMDirectCodexConversation(args, checkpoint).ask
