"""Human-readable terminal output for CLI entry points."""
from __future__ import annotations
from .tag2ansi import tag2ansi
from .logger import logger

def confirm_overwrite(paths) -> bool:
    """Ask for explicit confirmation before overwriting existing files."""
    logger.error(tag2ansi("[yellow bold]The following files already exist:[reset]"))
    for path in paths:
        print(f"  {path}")
    try:
        response = input("Overwrite these files? [y/N]: ").strip().lower()
    except EOFError:
        return False
    return response in {"y", "yes", "1", "ok"}
