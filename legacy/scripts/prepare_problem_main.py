"""Standalone entry point for benchmark task preparation."""
from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(os.getcwd())
while not (ROOT / 'src').exists():
    ROOT = ROOT.parent
os.chdir(ROOT)

from src.prepare_problem import get_parser, main
from src.utils.logger import config_logger


if __name__ == "__main__":
    config_logger()
    parser = get_parser()
    args = parser.parse_args()
    exitcode = main(args)
    raise SystemExit(exitcode)
