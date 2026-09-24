"""Check the published 24 + 72 derived metrics offline; never invoke an evaluator/model."""
import csv
import hashlib
import json
from pathlib import Path

COHORTS = {'ds_core_3runs': (1, 4), 'glm_core_3runs': (2, 5), 'glm_full_first': (2, 3)}
DOMAINS = ('physics', 'chemistry', 'biology', 'material')
VARIANTS = ('original', 'single', 'multi')


def both(value):
    if value is None:
        return None
    values = [value[s]['numerical_pass'] for s in ('id_test', 'ood_test')]
    assert all(type(v) is bool for v in values)
    return all(values)


def aggregate(rows, scope, domain, variant, measure):
    chosen = [r for r in rows if r['group'] in COHORTS[scope] and r['domain'] == domain
              and (variant == 'all' or r['variant'] == variant)]
    values = [r['decisions'][measure] for r in chosen]
    assert all(v is None or type(v) is bool for v in values)
    passed = sum(v is True for v in values)
    judged = sum(v is not None for v in values)
    return {'passed': passed, 'judged': judged, 'planned': len(chosen), 'missing': len(chosen) - judged,
            'unique_tasks': len({r['task_index'] for r in chosen}),
            'posthoc_runs': sum(r['posthoc_recovery'] for r in chosen),
            'rate': passed / judged if judged else None,
            'percentage': 100 * passed / judged if judged else None,
            'confirmed_lower_bound_rate_all_planned': passed / len(chosen) if chosen else None,
            'contribution_ids': [r['id'] for r in chosen]}


def verify(directory):
    directory = Path(directory)
    results_root = directory.parent.parent
    doc = json.loads((directory / 'contributions_912.json').read_text())
    rows = doc['rows']
    assert len(rows) == len({r['id'] for r in rows}) == 912
    assert sum(r['posthoc_recovery'] for r in rows) == 43
    assert sum(r['decisions']['P'] is not None for r in rows) == 899
    assert sum(r['decisions']['M'] is not None for r in rows) == 898
    for r in rows:
        assert both(r['reporting_P']) == r['decisions']['P']
        assert both(r['reporting_mechanism']) == r['decisions']['M']
        if r['P_validation_failure']:
            assert r['decisions'] == {'P': False, 'M': False}
        else:
            probes = r['reporting_probes']
            assert set(probes) <= set(r['expected_probes'])
            if set(probes) == set(r['expected_probes']):
                expected = all(both(probes[p]) for p in r['expected_probes'])
                assert r['decisions']['M'] is expected
            else:
                assert r['decisions']['M'] is None
        native = r['archived_native_result']
        if native:
            path = results_root / native['results_relative_path']
            assert path.is_relative_to(results_root)
            assert hashlib.sha256(path.read_bytes()).hexdigest() == native['exported_file_sha256']
            performance = json.loads(path.read_text())
            assert performance['source'] == r['native_source']
        else:
            assert r['missing_native_performance']
    count = 0
    for filename, variants, size in [('summary_24', ('all',), 24), ('variants_72', VARIANTS, 72)]:
        table = json.loads((directory / (filename + '.json')).read_text())['rows']
        assert len(table) == size
        keys = {(r['scope'], r['domain'], r['variant'], r['measure']) for r in table}
        assert keys == {(s, d, v, m) for s in COHORTS for d in DOMAINS for v in variants for m in ('P', 'M')}
        for r in table:
            expected = aggregate(rows, r['scope'], r['domain'], r['variant'], r['measure'])
            for key, value in expected.items():
                assert r[key] == value, (filename, r['id'], key)
        with (directory / (filename + '.csv')).open(encoding='utf-8-sig', newline='') as f:
            csv_rows = list(csv.DictReader(f))
        assert len(csv_rows) == size
        for left, right in zip(table, csv_rows):
            assert left['id'] == right['id']
            for field in ('passed', 'judged', 'planned', 'missing'):
                assert left[field] == int(right[field])
            assert abs(left['percentage'] - float(right['percentage'])) < 1e-12
        count += size
    setup = json.loads((directory / 'setup.json').read_text())
    for rel, digest in setup['published_file_sha256'].items():
        assert hashlib.sha256((directory / rel).read_bytes()).hexdigest() == digest, rel
    return {'status': 'passed', 'primary_metrics': 24, 'variant_metrics': 72,
            'recomputed_metrics': count, 'selected_identities': len(rows),
            'archived_native_files_verified': sum(r['archived_native_result'] is not None for r in rows),
            'posthoc_identities': 43, 'model_calls': 0, 'native_results_modified': 0}


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), ensure_ascii=False))
