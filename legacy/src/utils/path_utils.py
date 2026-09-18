"""Generic path discovery and portable filename helpers."""
from __future__ import annotations

from pathlib import Path
import re


def discover_yaml_files(inputs: list[str | Path]) -> list[Path]:
    """Expand YAML files and directories into a sorted, duplicate-free list."""
    files = []
    for raw_path in inputs:
        path = Path(raw_path).expanduser()
        if path.is_dir():
            files.extend(sorted(p for p in path.rglob("*") if p.suffix.lower() in {".yaml", ".yml"}))
        elif path.is_file() and path.suffix.lower() in {".yaml", ".yml"}:
            files.append(path)
        else:
            raise ValueError(f"Input is not a YAML file or directory: {path}")
    unique = list(dict.fromkeys(path.resolve() for path in files))
    if not unique:
        raise ValueError("No YAML files were found.")
    return unique


def safe_name(value: str) -> str:
    """Convert arbitrary text to a portable filename."""
    name = re.sub(r"[^\w.-]+", "_", value, flags=re.UNICODE).strip("._")
    return name or "problem"
