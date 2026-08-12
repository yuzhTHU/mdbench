"""Export the benchmark problem library bundled with MDBench."""
from __future__ import annotations
import argparse
from pathlib import Path
from importlib import resources
from .utils.logger import logger
from .utils.tag2ansi import tag2ansi


def _problem_resources():
    """Return bundled YAML resources in deterministic filename order."""
    return sorted(
        (
            item
            for item in resources.files("problems").iterdir()
            if item.is_file() and item.name.endswith((".yaml", ".yml"))
        ),
        key=lambda item: item.name,
    )


def _confirm_overwrite(paths: list[Path]) -> bool:
    print("The following files already exist:")
    for path in paths:
        print(f"  {path}")
    try:
        reply = input("Overwrite these files? [y/N]: ")
    except EOFError:
        return False
    return reply.strip().lower() in {"y", "yes"}


def format_report(output_dir: Path, filenames: list[str], overwritten: set[str]) -> str:
    lines = [
        "[blue bold]MDBench · Problem export[reset]",
        f"  [blue bold]Path[reset]: {output_dir}/",
        f"  [blue bold]Files[reset]: {len(filenames)} problem definitions",
    ]
    for filename in filenames:
        suffix = " [yellow bold](Overwritten)[reset]" if filename in overwritten else ""
        lines.append(f"    - {filename}{suffix}")
    return tag2ansi("\n".join(lines))


def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(
            description="Export MDBench's bundled problem YAML files",
        )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Destination directory for the exported problem YAML files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing problem files without prompting",
    )
    return parser


def main(args) -> int:
    output_dir = Path(args.output_dir)
    bundled = _problem_resources()
    destinations = [output_dir / item.name for item in bundled]
    collisions = [path for path in destinations if path.exists()]

    if collisions and not args.force and not _confirm_overwrite(collisions):
        logger.error(
            "Problem export cancelled. Use --force to overwrite existing files."
        )
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    overwritten = {path.name for path in collisions}
    for resource, destination in zip(bundled, destinations):
        destination.write_bytes(resource.read_bytes())

    logger.info(
        format_report(
            output_dir,
            [resource.name for resource in bundled],
            overwritten,
        )
    )
    return 0
