"""Isolated worker for the time-bounded exact symbolic-equivalence proof."""
from __future__ import annotations

import pickle
import sys

import sympy as sp


def main() -> int:
    try:
        expression = pickle.load(sys.stdin.buffer)
        result = bool(sp.simplify(expression) == 0)
    except BaseException:
        result = False
    sys.stdout.buffer.write(b'1' if result else b'0')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
