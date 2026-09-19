"""Minimal linear baseline used to demonstrate the algorithm interface."""
from __future__ import annotations
import re
import json
import random
import numpy as np
from pathlib import Path
from ..scoring import public_variables
from ..validate_problem import ValidationError

def run(args, problem_file: Path, train_data_npy_file: Path, feedback_server_url):
    """Fit a zero-intercept linear model and return its submission and model.

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

    def ask(question: str, *, output_dir=None) -> str:
        del output_dir
        PROBE_NAME = re.compile(r"\bderive\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
        match = PROBE_NAME.search(question)
        if not match:
            raise ValidationError("Could not find the requested probe variable in the question.")
        return f"{match.group(1)} = {random.choice(terms)}"

    return submission, ask
