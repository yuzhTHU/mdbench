"""PySR phenomenological fit with direct LLM mechanism-probe answers.

The discovery stage is identical to :mod:`my_pysr`, but it can also reuse a
previous PySR submission.  Each mechanism probe is then sent, unchanged, to a
single OpenRouter chat-completions request.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import time
from typing import Any, Callable

import requests

from . import my_pysr
from .codex import atomic_json, load_api_key
from ..scoring import submission_formulas
from ..validate_problem import parse_equation, split_equation

__all__ = ["update_parser", "run", "get_ask"]

_OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
_DEFAULT_MODEL = "deepseek/deepseek-v4-flash-0731"


def update_parser(parser):
    parser = my_pysr.update_parser(parser)
    parser.add_argument(
        "--pysr-source-dir",
        type=Path,
        default=None,
        help="Existing PySR task result containing submission.txt; skips PySR.",
    )
    parser.add_argument(
        "--submission-source-dir",
        type=Path,
        default=None,
        help="Existing result containing submission.txt; skips PySR (generic alias).",
    )
    parser.add_argument("--pysr-llm-model", default=_DEFAULT_MODEL)
    parser.add_argument(
        "--pysr-llm-env-file",
        type=Path,
        default=None,
        help="Credential file containing OPENROUTER_API_KEY (default: repository .env).",
    )
    parser.add_argument("--pysr-llm-max-output-tokens", type=int, default=8192)
    parser.add_argument("--pysr-llm-request-timeout", type=float, default=600)
    parser.add_argument("--pysr-llm-retries", type=int, default=4)
    parser.add_argument(
        "--pysr-llm-reasoning-effort",
        choices=("minimal", "low", "medium", "high"),
        default="minimal",
        help="OpenRouter reasoning effort for each direct mechanism-probe request.",
    )
    return parser


def _api_key(args) -> str:
    if key := os.environ.get("OPENROUTER_API_KEY"):
        return key
    env_file = getattr(args, "pysr_llm_env_file", None)
    path = Path(env_file) if env_file else Path(__file__).resolve().parents[2] / ".env"
    if not path.is_file():
        raise RuntimeError("OPENROUTER_API_KEY is not set and the credential file is absent.")
    return load_api_key(path, "OPENROUTER_API_KEY")


def run(args, problem_file: Path, train_data_npy_file: Path,
        feedback_server_url) -> tuple[list[str], Any]:
    """Reuse a frozen PySR submission when requested, otherwise run PySR."""
    pysr_source = getattr(args, "pysr_source_dir", None)
    generic_source = getattr(args, "submission_source_dir", None)
    if pysr_source is not None and generic_source is not None:
        raise ValueError("Use only one of --pysr-source-dir and --submission-source-dir.")
    source = generic_source if generic_source is not None else pysr_source
    if source is None:
        formulas, checkpoint = my_pysr.run(
            args, problem_file, train_data_npy_file, feedback_server_url)
        checkpoint = dict(checkpoint)
    else:
        source = Path(source).resolve()
        submission = source / "submission.txt"
        if not submission.is_file():
            raise FileNotFoundError(f"Existing PySR submission not found: {submission}")
        formulas = submission_formulas(submission.read_text())
        checkpoint = {
            "algorithm": "pysr_and_llm_direct",
            "reused_submission_result": str(source),
        }

    # Fail before evaluation if the externally supplied model configuration is
    # unusable, without ever persisting the credential in the checkpoint.
    _api_key(args)
    checkpoint.update({
        "algorithm": "pysr_and_llm_direct",
        "formula": formulas,
        "probe_model": getattr(args, "pysr_llm_model", _DEFAULT_MODEL),
        "probe_reasoning_effort": getattr(
            args, "pysr_llm_reasoning_effort", "minimal"),
    })
    return formulas, checkpoint


def _extract_formula(text: str, probe_name: str) -> str:
    """Extract one benchmark equation while rejecting extra equations/prose."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    text = re.sub(r"^```(?:text|math|latex)?\s*|\s*```$", "", text,
                  flags=re.DOTALL | re.IGNORECASE).strip()
    try:
        formulas = submission_formulas(text)
    except Exception:
        formulas = []
        pattern = re.compile(rf"^\s*{re.escape(probe_name)}\s*=\s*.+?\s*$")
        for line in text.splitlines():
            candidate = line.strip().strip("$ ")
            if pattern.fullmatch(candidate):
                try:
                    parse_equation(candidate)
                except Exception:
                    continue
                formulas.append(candidate)
    matching = []
    for formula in formulas:
        try:
            left, _ = split_equation(formula)
        except Exception:
            continue
        if left == probe_name:
            matching.append(formula)
    if len(formulas) != 1 or len(matching) != 1:
        raise ValueError(
            f"LLM probe response must contain exactly one {probe_name} equation; "
            f"received: {text[:500]!r}")
    return matching[0]


class DirectProbeConversation:
    def __init__(self, args, checkpoint: Any):
        self.args = args
        self.checkpoint = checkpoint

    def ask(self, prompt: str, probe_name: str, probe_description: str,
            *, output_dir: str | Path) -> str:
        del probe_description
        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)

        # A batch may be interrupted after an expensive probe request has
        # completed but before the task-level performance file is written.
        # Treat the already parsed formula as a durable checkpoint so recovery
        # runs only ask for probes that are genuinely missing.
        cached_formula = output / "formula.txt"
        if cached_formula.is_file():
            formula = _extract_formula(cached_formula.read_text(), probe_name)
            (output / "cache-reused.txt").write_text("formula.txt\n")
            return formula

        (output / "prompt.txt").write_text(prompt)

        model = getattr(self.args, "pysr_llm_model", _DEFAULT_MODEL)
        maximum = int(getattr(self.args, "pysr_llm_max_output_tokens", 2048))
        timeout = float(getattr(self.args, "pysr_llm_request_timeout", 180))
        retries = int(getattr(self.args, "pysr_llm_retries", 4))
        reasoning_effort = getattr(
            self.args, "pysr_llm_reasoning_effort", "minimal")
        if maximum < 1 or timeout <= 0 or retries < 1:
            raise ValueError("LLM token limit, request timeout, and retry count must be positive.")
        body = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Answer the user's mechanism probe directly from the supplied frozen "
                        "equation. Follow its output contract exactly: emit one plain-text "
                        "equation and no reasoning, prose, Markdown, or additional equations."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": maximum,
            "reasoning": {"effort": reasoning_effort},
            "usage": {"include": True},
        }
        atomic_json(output / "request.json", body)
        headers = {
            "Authorization": "Bearer " + _api_key(self.args),
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/MechanismDiscoveryBench",
            "X-Title": "MechanismDiscoveryBench",
        }
        response_data: dict[str, Any] | None = None
        text: str | None = None
        last_error: Exception | None = None
        with requests.Session() as session:
            for attempt in range(retries):
                try:
                    response = session.post(
                        _OPENROUTER_CHAT_URL, headers=headers, json=body, timeout=timeout)
                    if response.status_code == 429 or response.status_code >= 500:
                        raise requests.HTTPError(
                            f"OpenRouter returned HTTP {response.status_code}: {response.text[:500]}",
                            response=response)
                    response.raise_for_status()
                    response_data = response.json()
                    atomic_json(output / f"response-attempt-{attempt + 1}.json", response_data)
                    candidate = response_data["choices"][0]["message"].get("content")
                    if not isinstance(candidate, str) or not candidate.strip():
                        finish = response_data["choices"][0].get("finish_reason")
                        raise ValueError(f"OpenRouter returned no assistant content (finish_reason={finish})")
                    text = candidate.strip()
                    break
                except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as exc:
                    last_error = exc
                    if attempt + 1 == retries:
                        raise RuntimeError(f"OpenRouter probe request failed: {exc}") from exc
                    time.sleep(min(2 ** attempt, 8))
        if response_data is None or text is None:
            raise RuntimeError(f"OpenRouter probe request failed: {last_error}")
        atomic_json(output / "response.json", response_data)
        (output / "reply.txt").write_text(text + "\n")
        formula = _extract_formula(text, probe_name)
        (output / "formula.txt").write_text(formula + "\n")
        return formula


def get_ask(args, checkpoint: Any) -> Callable:
    return DirectProbeConversation(args, checkpoint).ask
