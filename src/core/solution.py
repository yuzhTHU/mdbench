"""Executable solution steps derived from a problem's mechanism equations."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable

SolutionFunction = Callable[
    [dict[str, np.ndarray]], 
    list[np.ndarray]
]


@dataclass
class SolutionItem:
    """One ordered mechanism-solving step.

    A single-variable step is an ordinary causal evaluation. Multiple
    ``variables`` denote values that must be obtained simultaneously from one
    coupled equation system. ``formulas`` is aligned with ``variables`` when a
    closed form exists, and is empty when numerical solving is required.
    """

    variables: list[str]
    formulas: list[str]
    function: SolutionFunction
