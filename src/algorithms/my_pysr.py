"""Phenomenological symbolic-regression baseline powered by PySR."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np
import sympy as sp

from ..scoring import public_variables, validate_data
from ..validate_problem import ValidationError, parse_equation

__all__ = ["update_parser", "run", "get_ask"]


def update_parser(parser):
    """Add PySR-specific command-line arguments."""
    parser.add_argument(
        "--timeout",
        type=float,
        default=15 * 60,
        help="PySR search time limit in seconds.",
    )
    parser.add_argument(
        "--pysr-maxsize",
        type=int,
        default=30,
        help="Maximum complexity of a discovered expression.",
    )
    parser.add_argument(
        "--pysr-populations",
        type=int,
        default=31,
        help="Number of PySR populations.",
    )
    return parser


def _prepare_julia_environment() -> None:
    """Keep JuliaCall's project beside the active Python environment.

    A conda Python can have ``sys.prefix == sys.base_prefix``.  JuliaPkg then
    consults an inherited ``CONDA_PREFIX``, which may point at an unrelated or
    read-only base environment.  An explicit project avoids that ambiguity.
    """
    os.environ.setdefault(
        "PYTHON_JULIAPKG_PROJECT", str(Path(sys.prefix).resolve() / "julia_env")
    )


def _expression_text(model, source_names: list[str]) -> str:
    """Translate PySR's private feature names back to benchmark names."""
    expression = model.sympy()
    if isinstance(expression, list):
        if len(expression) != 1:
            raise RuntimeError("PySR unexpectedly returned multiple outputs.")
        expression = expression[0]
    expression = sp.sympify(expression)
    replacements = {
        sp.Symbol(f"mdx{i}"): sp.Symbol(name, real=True)
        for i, name in enumerate(source_names)
    }
    return sp.sstr(expression.xreplace(replacements))


def _linear_fallback(X: np.ndarray, y: np.ndarray, source_names: list[str]) -> str:
    """Return a valid result even when a very short search finds no candidate."""
    design = np.column_stack([np.ones(X.shape[0]), X])
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    expression = sp.Float(float(coefficients[0]), 17)
    for coefficient, name in zip(coefficients[1:], source_names):
        expression += sp.Float(float(coefficient), 17) * sp.Symbol(name, real=True)
    return sp.sstr(expression)


def run(args, problem_file: Path, train_data_npy_file: Path,
        feedback_server_url) -> tuple[list[str], Any]:
    """Fit the observed target directly and return the best symbolic equation.

    This is deliberately a phenomenological baseline: unlike the agent-based
    algorithms it does not query the feedback server or introduce latent
    mechanistic variables.
    """
    del feedback_server_url
    timeout = float(getattr(args, "timeout", 15 * 60))
    if timeout <= 0:
        raise ValueError("Timeout must be positive.")

    problem = json.loads(Path(problem_file).read_text())
    variables, target, columns = public_variables(problem)
    train = np.load(train_data_npy_file, allow_pickle=False)
    validate_data(train, columns)

    sources = [variable.name for variable in variables if variable.role != "target"]
    if not sources:
        raise ValidationError("PySR requires at least one observed source variable.")
    X = np.column_stack([train[columns.index(name)] for name in sources])
    y = train[columns.index(target)]

    save = Path(getattr(args, "save_path", ".")).resolve()
    output_directory = save / "pysr"
    output_directory.mkdir(parents=True, exist_ok=True)

    # Import lazily so listing algorithms and using other baselines does not
    # initialize Julia.  A very large iteration count makes the wall-clock
    # timeout, rather than an arbitrary iteration limit, stop the search.
    _prepare_julia_environment()
    from pysr import PySRRegressor

    seed = getattr(args, "seed", -1)
    random_state = None if seed is None or int(seed) < 0 else int(seed)
    model = PySRRegressor(
        model_selection="accuracy",
        niterations=1_000_000,
        timeout_in_seconds=timeout,
        populations=int(getattr(args, "pysr_populations", 31)),
        maxsize=int(getattr(args, "pysr_maxsize", 30)),
        binary_operators=["+", "-", "*", "/", "pow_abs(x, y) = abs(x)^y"],
        unary_operators=[
            "square(x) = x^2",
            "cube(x) = x^3",
            "sqrt_abs(x) = sqrt(abs(x))",
            "exp",
            "log_abs(x) = log(abs(x))",
            "sin",
            "cos",
            "abs",
        ],
        constraints={"pow_abs": (-1, 3), "exp": 8},
        nested_constraints={
            "exp": {"exp": 0},
            "sin": {"sin": 0, "cos": 0},
            "cos": {"sin": 0, "cos": 0},
        },
        extra_sympy_mappings={
            "pow_abs": lambda x, y: sp.Abs(x) ** y,
            "square": lambda x: x ** 2,
            "cube": lambda x: x ** 3,
            "sqrt_abs": lambda x: sp.sqrt(sp.Abs(x)),
            "log_abs": lambda x: sp.log(sp.Abs(x)),
        },
        precision=64,
        random_state=random_state,
        parallelism="multithreading",
        progress=False,
        verbosity=1 if getattr(args, "verbose", False) else 0,
        output_directory=str(output_directory),
        run_id="search",
    )
    private_names = [f"mdx{i}" for i in range(len(sources))]
    model.fit(X, y, variable_names=private_names)

    equations = model.equations_
    has_candidate = (
        equations is not None
        and not isinstance(equations, list)
        and not equations.empty
        and "loss" in equations
        and np.isfinite(equations["loss"].to_numpy(dtype=float)).any()
    )
    expression = (
        _expression_text(model, sources)
        if has_candidate
        else _linear_fallback(X, y, sources)
    )
    formula = f"{target} = {expression}"
    parse_equation(formula)
    checkpoint = {
        "algorithm": "my_pysr",
        "formula": formula,
        "run_directory": str(output_directory / "search"),
    }
    return [formula], checkpoint


def get_ask(args, checkpoint: Any) -> Callable:
    """Return an intentionally empty mechanism-probe implementation."""
    del args, checkpoint

    def ask(prompt: str, probe_name: str, probe_description: str, *, output_dir=None) -> str:
        del prompt, probe_name, probe_description, output_dir
        return ""

    return ask
