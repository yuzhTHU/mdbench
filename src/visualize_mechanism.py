"""Render a mechanism variable-dependency graph."""
from __future__ import annotations
import shutil
import argparse
import subprocess
from pathlib import Path
from .utils import logger, tag2ansi
from .features.io import load_problem
from .features.visualization import MechanismGraphBuilder


def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(description="Visualize a mechanism dependency graph")
    parser.add_argument("--problem", required=True, help="Problem YAML path")
    parser.add_argument("--output", required=True, help="Output DOT, SVG, PNG, or PDF path")
    parser.add_argument("--format", choices=["dot", "svg", "png", "pdf"], help="Output format; inferred from --output by default")
    parser.add_argument("--hide-constants", action="store_true", help="Omit constant nodes and their edges")
    return parser


def main(args):
    try:
        problem = load_problem(args.problem, solve=True)
        dot = MechanismGraphBuilder().build(
            problem,
            include_constants=not args.hide_constants,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output_format = args.format or output.suffix.lower().lstrip(".")
        if output_format == "dot":
            output.write_text(dot, encoding="utf-8")
        elif (executable := shutil.which("dot")) is None:
            raise RuntimeError(
                "Graphviz 'dot' is required to render SVG, PNG, or PDF. "
                "Install Graphviz or request --format dot."
            )
        else:
            subprocess.run(
                [executable, f"-T{output_format}", "-o", str(output)],
                input=dot, text=True, check=True,
            )
        logger.info(f"Mechanism graph written to {output}")
        return 0
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        logger.error(tag2ansi(f"[red bold]Error:[reset] {exc}"))
        return 1
