#!/usr/bin/env python3
"""Recompute both published Core80 metrics from stored JSON; no model or solver calls.

Paper task SA(M) requires every planned probe to be symbolically correct.
Legacy probe M is the number of symbolically correct probes / planned probes.
Missing and invalid observations contribute zero only to these aggregate policies;
the original files are never modified and nullable native observations stay null.
"""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path


EXPERIMENTS = {
    "aifeynman": "aifeynman_direct_ask_deepseek-v4-flash-0731_core80_20261109",
    "e2esr": "e2esr_direct_ask_deepseek-v4-flash-0731_core80_20261109",
    "llmsr": "llmsr_deepseek-v4-flash-0731_direct_ask_core80_20261109",
}
DOMAINS = ("physics", "chemistry", "biology", "material")
STRATA = ("original", "single_variant", "multiple_variant")
SPLITS = ("train", "id_test", "ood_test")
METRICS = ("numerically_equivalent", "symbolically_equivalent")
EXPECTED = {
    "aifeynman": {"result_files": 75, "strict_complete": 26, "P": 0, "M_task": 0, "M_probe": 0},
    "e2esr": {"result_files": 75, "strict_complete": 27, "P": 0, "M_task": 1, "M_probe": 1},
    "llmsr": {"result_files": 69, "strict_complete": 41, "P": 0, "M_task": 1, "M_probe": 2},
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def stratum(name):
    if name.endswith(" - Original"):
        return "original"
    parts = name.split(" - Variant ", 1)
    assert len(parts) == 2 and all(x.isdigit() for x in parts[1].split("-")), name
    return "single_variant" if len(parts[1].split("-")) == 1 else "multiple_variant"


def symbolic(scores):
    """Audit replicated symbolic flags rather than choosing a favorable split."""
    scores = scores or {}
    values = [(scores.get(split) or {}).get("symbolically_equivalent") for split in SPLITS]
    present = [value for value in values if type(value) is bool]
    assert not present or len(present) == len(SPLITS), ("partial symbolic flags", values)
    assert len(set(present)) <= 1, ("contradictory symbolic flags", values)
    return present[0] if present else None


def valid_four(scores):
    scores = scores or {}
    return all(type((scores.get(split) or {}).get(metric)) is bool
               for split in ("id_test", "ood_test") for metric in METRICS)


def make_row(algorithm, experiment, job, result, relative_path):
    evaluation = (result or {}).get("evaluation") or {}
    p = evaluation.get("phenomenal") or {}
    probes = evaluation.get("mechanism_probes") or []
    expected_names = job["probe_names"]
    assert expected_names and len(expected_names) == len(set(expected_names)), job["job_id"]
    names = [probe.get("probe") for probe in probes]
    assert len(names) == len(set(names)), (job["job_id"], "duplicate actual probe names")
    assert set(names).issubset(set(expected_names)), (job["job_id"], "unexpected probe names")
    names_complete = collections.Counter(names) == collections.Counter(expected_names)
    p_symbolic = symbolic(p)
    probe_symbolic = {probe.get("probe"): symbolic(probe.get("scores")) for probe in probes}
    p_valid = p.get("ok") is True and valid_four(p)
    probe_valid = sum(probe.get("ok") is True and valid_four(probe.get("scores")) for probe in probes)
    strict = bool(p_valid and names_complete and probe_valid == len(expected_names)
                  and evaluation.get("probe_count") == len(expected_names))
    assert type(job["strict_complete"]) is bool, job["job_id"]
    assert strict == job["strict_complete"], (job["job_id"], "strict_complete disagrees with setup", strict)
    probe_ok = {probe.get("probe"): probe.get("ok") is True for probe in probes}
    native_probe_flags = {
        name: probe_symbolic.get(name) if probe_ok.get(name) else None for name in expected_names
    }
    known_probe_flags = [value for value in native_probe_flags.values() if type(value) is bool]
    all_probe_native = all(known_probe_flags) if len(known_probe_flags) == len(expected_names) else None
    probe_pass = sum(value is True for value in native_probe_flags.values())
    p_errors = collections.Counter([p.get("error_type") or "unknown"] if p and p.get("ok") is not True else [])
    probe_errors = collections.Counter(probe.get("error_type") or "unknown"
                                       for probe in probes if probe.get("ok") is not True)
    return {
        "algorithm": algorithm,
        "experiment": experiment,
        "job_id": job["job_id"],
        "task_index": job["task_index"],
        "task_name": job["task_name"],
        "task_rel": job["task_rel"],
        "domain": job["domain"],
        "stratum": stratum(job["task_name"]),
        "outcome": job["outcome"],
        "result_present": result is not None,
        "result_path": relative_path,
        "strict_complete": strict,
        "P_valid_native": p_valid,
        "P_symbolic_native": p_symbolic if p.get("ok") is True else None,
        "probe_symbolic_native": native_probe_flags,
        "M_all_probes_symbolic_native": all_probe_native,
        "paper_task_SA_P_pass_policy": p.get("ok") is True and p_symbolic is True,
        "paper_task_SA_M_pass_policy": all_probe_native is True,
        "planned_probe_count": len(expected_names),
        "actual_probe_count": len(probes),
        "valid_native_probe_count": probe_valid,
        "symbolic_pass_probe_count": probe_pass,
        "legacy_probe_M_percent_policy": 100 * probe_pass / len(expected_names),
        "P_error_records": dict(p_errors),
        "probe_error_records": dict(probe_errors),
        "no_assistant_content_length_probes": sum(
            "no assistant content (finish_reason=length)" in (probe.get("error") or "")
            for probe in probes),
    }


def aggregate(rows):
    n = len(rows)
    probes = sum(row["planned_probe_count"] for row in rows)
    p_pass = sum(row["paper_task_SA_P_pass_policy"] for row in rows)
    m_task = sum(row["paper_task_SA_M_pass_policy"] for row in rows)
    m_probe = sum(row["symbolic_pass_probe_count"] for row in rows)
    p_errors, probe_errors = collections.Counter(), collections.Counter()
    for row in rows:
        p_errors.update(row["P_error_records"])
        probe_errors.update(row["probe_error_records"])
    return {
        "planned_tasks": n,
        "result_files": sum(row["result_present"] for row in rows),
        "missing_result_files": sum(not row["result_present"] for row in rows),
        "strict_complete": sum(row["strict_complete"] for row in rows),
        "P_valid_native": sum(row["P_valid_native"] for row in rows),
        "P_valid_probes_incomplete": sum(row["P_valid_native"] and not row["strict_complete"] for row in rows),
        "paper_task_SA_P_pass_count": p_pass,
        "paper_task_SA_P_percent": 100 * p_pass / n if n else None,
        "paper_task_SA_M_pass_count": m_task,
        "paper_task_SA_M_percent": 100 * m_task / n if n else None,
        "planned_probes": probes,
        "valid_native_probes": sum(row["valid_native_probe_count"] for row in rows),
        "legacy_probe_M_pass_count": m_probe,
        "legacy_probe_M_percent": 100 * m_probe / probes if probes else None,
        "one_minus_M_given_P": 1 - sum(row["paper_task_SA_P_pass_policy"] and row["paper_task_SA_M_pass_policy"]
                                       for row in rows) / p_pass if p_pass else None,
        "outcomes": dict(collections.Counter(row["outcome"] for row in rows)),
        "P_error_records": dict(p_errors),
        "probe_error_records": dict(probe_errors),
        "runs_with_probe_error": sum(bool(row["probe_error_records"]) for row in rows),
        "no_assistant_content_length_probes": sum(row["no_assistant_content_length_probes"] for row in rows),
        "runs_with_no_assistant_content_length": sum(row["no_assistant_content_length_probes"] > 0 for row in rows),
    }


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                             if isinstance(value, (dict, list)) else ("null" if value is None else value)
                             for key, value in row.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--out", type=Path,
                        help="Default: RESULTS_ROOT/_summaries/rebuttal-sr-core80-20261109")
    args = parser.parse_args()
    root = args.results_root.resolve()
    out = args.out or root / "_summaries" / "rebuttal-sr-core80-20261109"
    rows, inputs = [], []
    identities = {}
    for algorithm, experiment in EXPERIMENTS.items():
        experiment_root = root / experiment
        setup_path = experiment_root / "setup.json"
        setup = load_json(setup_path)
        jobs = setup["planned_runs"]
        assert len(jobs) == 80, (experiment, "expected Core80")
        assert len({job["job_id"] for job in jobs}) == 80
        assert len({job["task_rel"] for job in jobs}) == 80
        assert collections.Counter(job["domain"] for job in jobs) == {domain: 20 for domain in DOMAINS}
        identities[algorithm] = {(job["task_index"], job["task_rel"]) for job in jobs}
        inputs.append({"path": f"{experiment}/setup.json", "sha256": sha256(setup_path)})
        seen_paths = set()
        for job in sorted(jobs, key=lambda item: item["task_index"]):
            result_path = job["result_path"]
            result, relative_path = None, None
            if result_path is not None:
                relative = Path(result_path)
                assert not relative.is_absolute() and ".." not in relative.parts, result_path
                assert result_path not in seen_paths, result_path
                seen_paths.add(result_path)
                path = experiment_root / relative
                assert path.is_file(), path
                result = load_json(path)
                assert isinstance(result, dict) and "evaluation" in result, path
                relative_path = f"{experiment}/{relative.as_posix()}"
                inputs.append({"path": relative_path, "sha256": sha256(path)})
            rows.append(make_row(algorithm, experiment, job, result, relative_path))
    assert all(value == identities["aifeynman"] for value in identities.values())
    totals = {}
    breakdown = []
    for algorithm in EXPERIMENTS:
        selected = [row for row in rows if row["algorithm"] == algorithm]
        total = aggregate(selected)
        expected = EXPECTED[algorithm]
        assert total["planned_tasks"] == 80 and total["planned_probes"] == 165
        assert total["result_files"] == expected["result_files"], (algorithm, total)
        assert total["strict_complete"] == expected["strict_complete"], (algorithm, total)
        assert total["paper_task_SA_P_pass_count"] == expected["P"], (algorithm, total)
        assert total["paper_task_SA_M_pass_count"] == expected["M_task"], (algorithm, total)
        assert total["legacy_probe_M_pass_count"] == expected["M_probe"], (algorithm, total)
        totals[algorithm] = total
        for domain in ("all", *DOMAINS):
            for variant in ("all", *STRATA):
                subset = [row for row in selected if (domain == "all" or row["domain"] == domain)
                          and (variant == "all" or row["stratum"] == variant)]
                breakdown.append({"algorithm": algorithm, "domain": domain, "stratum": variant,
                                  **aggregate(subset)})
    output = {
        "metric_definitions": {
            "paper_task_SA_P": "Symbolically equivalent phenomenal tasks / all planned tasks.",
            "paper_task_SA_M": "Tasks with every planned probe symbolically equivalent / all planned tasks; independent of P.",
            "legacy_probe_M": "Symbolically equivalent individual probes / all planned probes (matches Legacy Core Excel).",
            "missing_policy": "Missing/invalid/error observations contribute zero in derived policy metrics only. Native nulls and original errors are retained.",
            "strict_complete": "Stored result with valid P and every unique planned probe, each with explicit boolean ID/OOD numerical and symbolic flags. False is a valid score.",
            "symbolic_split_audit": "All available train/ID/OOD symbolic flags are explicit booleans and agree; absent flags remain null.",
            "scope": "Three Core80 independent attempts, 20 tasks per domain, 8 original/16 single-variant/56 multiple-variant tasks. Not Full512.",
        },
        "inputs": inputs,
        "core80": totals,
        "breakdown": breakdown,
        "positive_tasks_or_probes": [row for row in rows if row["paper_task_SA_P_pass_policy"]
                                     or row["symbolic_pass_probe_count"]],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    write_csv(out / "domain_variant_breakdown.csv", breakdown)
    write_csv(out / "per_task.csv", rows)
    print(json.dumps({"out": str(out), "core80": totals}, ensure_ascii=False))


if __name__ == "__main__":
    main()
