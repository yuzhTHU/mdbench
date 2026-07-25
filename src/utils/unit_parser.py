"""Parse dependency-free SI unit strings used by MDBench schemas."""
import re
from typing import get_args

from ..core import SI


_SI_UNITS = set(get_args(SI))
_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", "0123456789+-")


def parse_unit(text: str) -> dict[str, int | float]:
    """Parse common SI notations into a base-unit exponent dictionary."""
    text = text.strip()
    if text.lower() in {"", "1", "1 (dimensionless)", "dimensionless", "无量纲", "-"}:
        return {}

    text = text.replace("**", "^").replace("·", "*")
    text = re.sub(
        r"([A-Za-z]+)([⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+)",
        lambda match: f"{match.group(1)}^{match.group(2).translate(_SUPERSCRIPTS)}",
        text,
    )

    result: dict[str, int | float] = {}
    sections = text.split("/")
    if any(not section.strip() for section in sections):
        raise ValueError(f"Unit inference failed: Invalid unit expression '{text}'.")

    for section_index, section in enumerate(sections):
        if (
            section.strip().startswith("*")
            or section.strip().endswith("*")
            or re.search(r"\*\s*\*", section)
        ):
            raise ValueError(f"Unit inference failed: Invalid unit expression '{text}'.")
        for part in section.replace("*", " ").split():
            if part == "1":
                continue
            match = re.fullmatch(r"([A-Za-z]+)(?:\^([-+]?\d+(?:\.\d+)?))?", part)
            if not match or match.group(1) not in _SI_UNITS:
                raise ValueError(f"Unit inference failed: Invalid unit '{part}' in '{text}'.")
            name, exponent = match.groups()
            value = float(exponent) if exponent and "." in exponent else int(exponent or 1)
            if section_index > 0:
                value = -value
            result[name] = result.get(name, 0) + value
    return {name: value for name, value in result.items() if value != 0}
