"""Minimal linear baseline used to demonstrate the algorithm interface."""
from __future__ import annotations
import json
import random
from typing import Any, Callable
import numpy as np
from pathlib import Path
from ..scoring import public_variables
from ..validate_problem import ValidationError

def run(args, problem_file: Path, train_data_npy_file: Path,
        feedback_server_url) -> tuple[list[str], Any]:
    """Fit a zero-intercept linear model and return submission plus checkpoint.

    ``feedback_server_url`` is intentionally unused: this baseline performs one
    ordinary least-squares fit and makes no iterative feedback requests.
    """
    problem = json.loads(Path(problem_file).read_text())
    variables, target, columns = public_variables(problem)
    train = np.load(train_data_npy_file, allow_pickle=False)
    # train 与 columns 的一致性由上层框架保证。

    inputs = [variable.name for variable in variables if variable.role == "input"]
    if not inputs:
        raise ValidationError("The linear demo requires at least one input variable.")

    # Dummy 用训练数据做无截距最小二乘拟合，得到 y = sum(a_i*x_i) 中的系数 a_i。
    # 它把每个线性项写成中间变量 z_i = a_i*x_i，作为一个最简单的“机制模型”。
    # 最终 submission 先定义所有 a_i 和 z_i，再用 target = z1 + z2 + ... 汇总预测。
    design = np.column_stack([train[columns.index(name)] for name in inputs])
    response = train[columns.index(target)]
    coefficients, _, _, _ = np.linalg.lstsq(design, response, rcond=None)
    generated = {f"{prefix}{i}" for prefix in ("a", "z") for i in range(1, len(inputs) + 1)}
    if collisions := sorted(set(columns) & generated):
        raise ValidationError(f"Public variables collide with dummy names: {collisions}")
    lines, terms = [], []
    for i, (name, coefficient) in enumerate(zip(inputs, coefficients), 1):
        terms.append(f"a{i}*{name}")
        lines += [f"a{i} = {float(coefficient):.17g}", f"z{i} = {terms[-1]}"]
    lines.append(f"{target} = " + " + ".join(f"z{i}" for i in range(1, len(terms) + 1)))
    submission = lines

    return submission, {'terms': tuple(terms), 'random_state': random.getstate()}


def get_ask(args, checkpoint: Any) -> Callable:
    """Restore one independent probe callable from a frozen checkpoint.

    An ``ask`` callable's behavior must be determined entirely by its checkpoint:
    restoring equivalent checkpoints must produce equivalent behavior for the
    same sequence of calls, without relying on shared mutable runtime state. If a
    checkpoint contains a path or other reference to an on-disk artifact, both
    ``get_ask`` and the returned callable must treat that artifact as immutable;
    a backend that requires writes must first work on a private copy.
    """
    del args
    terms = tuple(checkpoint['terms'])
    rng = random.Random()
    rng.setstate(checkpoint['random_state'])

    def ask(prompt: str, probe_name: str, probe_description: str, *, output_dir=None) -> str:
        # Custom baselines may use the explicit probe metadata, though algorithms
        # should normally send the standardized prompt through unchanged.
        del prompt, probe_description, output_dir
        return f"{probe_name} = {rng.choice(terms)}"

    return ask
