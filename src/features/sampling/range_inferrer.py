"""Replaceable variable-range inference policy."""
from __future__ import annotations

from ...core import Problem

class RangeInferrer:
    """Convert variable sampling specifications to train and OOD ranges.

    Input is a ``Problem``. Output maps each source variable to
    ``(train_min, train_max, ood_min, ood_max, distribution)``. This
    domain-dependent component is isolated so a curated or model-backed
    implementation can replace it later.
    """

    def infer(
        self,
        problem: Problem,
    ) -> dict[str, tuple[float, float, float, float, str]]:
        result = {}
        produced = {variable.name for variable in problem.intermediate_variables}
        for variable in [*problem.input_variables, *problem.auxiliary_input_variables]:
            if variable.name in produced:
                continue
            if variable.sampling is None:
                raise ValueError(f"Missing sampling specification for source variable {variable.name}.")
            required = {"min", "max", "ood_boundary", "distribution"}
            if missing := sorted(required - variable.sampling.keys()):
                raise ValueError(
                    f"Sampling specification for {variable.name} is missing: {', '.join(missing)}."
                )
            spec = variable.sampling
            low = float(spec["min"])
            high = float(spec["max"])
            if not low < high:
                raise ValueError(f"Invalid sampling range for {variable.name}: min must be below max.")
            split = float(spec["ood_boundary"])
            if not low < split < high:
                raise ValueError(f"Invalid OOD boundary for {variable.name}.")
            distribution = str(spec["distribution"])
            if distribution not in {"uniform", "log_uniform"}:
                raise ValueError(f"Unsupported distribution for {variable.name}: {distribution}.")
            if distribution == "log_uniform" and low <= 0:
                raise ValueError(f"Log-uniform range for {variable.name} must be positive.")
            result[variable.name] = (low, split, split, high, distribution)
        return result
