#!/usr/bin/env python3
"""Export stored results only. No evaluation, fitting, or network requests."""
import argparse
import copy
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

NAMES = {
    'aifeynman': 'aifeynman_direct_ask_deepseek-v4-flash-0731_core80_20261109',
    'e2esr': 'e2esr_direct_ask_deepseek-v4-flash-0731_core80_20261109',
    'llmsr': 'llmsr_deepseek-v4-flash-0731_direct_ask_core80_20261109',
}
SUMMARY = '_summaries/rebuttal-sr-core80-20261109'
VENDORS = {
    'aifeynman': {'repository': 'https://github.com/SJ001/AI-Feynman',
                 'commit': 'a05bc4a5be23d6eb3e1d0b2f7eb1ab5b78a920ad',
                 'bf_try_time': 10, 'nn_epochs': 500, 'polyfit_degree': 4,
                 'selection': 'Lowest public training error among reconstructed original-coordinate Pareto candidates.',
                 'compatibility': 'CUDA scalar metrics converted to Python numbers in adapter; upstream files unchanged.'},
    'e2esr': {'repository': 'https://github.com/facebookresearch/symbolicregression',
             'commit': 'c4144c99d078dd611795338e24fb3da49a32b9d8',
             'checkpoint_url': 'https://dl.fbaipublicfiles.com/symbolicregression/model1.pt',
             'checkpoint_sha256': '169569a0648ae1a4f1ac2bbb376fa5ebb70ebce4bfcf23d6666323596990ff91',
             'sympytorch_commit': 'cb4cd0f516c2f1495ebbddd78fa93279e5f24c56',
             'points_per_bag': 200, 'max_trees_to_refine': 100,
             'selection': 'One official inference/BFGS pass; all 1000 training samples and all public inputs; lowest full-training MSE.',
             'compatibility': 'Boolean attention mask; supported column-shaped target; inverse normalization to public variable names. No feature dropping.'},
    'llmsr': {'repository': 'https://github.com/deep-symbolic-mathematics/LLM-SR',
             'commit': '41c212312df6c16d936c9cb395356a62774c47e3',
             'generator_model': 'deepseek/deepseek-v4-flash-0731',
             'generator_reasoning_effort': 'low', 'generator_temperature': 1,
             'generator_max_output_tokens': 2048, 'generator_max_candidates': 1000,
             'islands': 10, 'functions_per_prompt': 2, 'samples_per_prompt': 4,
             'candidate_evaluation_seconds': 30, 'parameter_count': 10,
             'parameter_initialization': 'all ones', 'optimizer': 'SciPy BFGS',
             'selection': 'Lowest public training MSE; initial affine seed may win; no ID/OOD selection.',
             'adaptation': 'Official pipeline.main/buffer/evaluator/sampler with transport injection and restricted mathematical AST interpreter; not unrestricted Python search.'},
}


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_json(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def costs(algorithm, files):
    terminal = files['terminal.json']['value']
    discovery = files.get('discovery/result.json', {}).get('value', {})
    probe = terminal.get('api_usage')
    generation = discovery.get('generation_usage') if algorithm == 'llmsr' else None
    reported = sum(float(x.get('provider_reported_cost_usd', 0)) for x in [probe, generation] if x)
    unknown = sum(int(x.get('unknown_cost_requests', 0)) for x in [probe, generation] if x)
    # An absent evaluation ledger is not evidence of zero billed requests.
    complete = bool(probe and probe.get('cost_complete') is True and
                    (algorithm != 'llmsr' or generation is not None) and unknown == 0)
    return {'provider_reported_usd': reported if probe or generation else None, 'total_cost_usd': reported if complete else None,
            'cost_complete': complete, 'known_unknown_cost_requests': unknown,
            'unknown_cost_requests': unknown if probe else None,
            'evaluation_usage_available': probe is not None,
            'generation_usage_available': generation is not None if algorithm == 'llmsr' else 'not_applicable',
            'pricing_estimate_usd': None, 'uncertain_upper_bound_usd': None,
            'reserved_inflight_usd': None,
            'note': 'Stored terminal/discovery usage only. Null means unavailable/not estimated, not free. No account-balance subtraction.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='New staging directory containing results/')
    args = parser.parse_args()
    root = args.batch_root.resolve()
    snap_path = root/'analysis/paper_metric_summary_20261109T171331Z/raw_snapshot.json'
    snapshot = json.loads(snap_path.read_text())
    source_record = json.loads((root/'source_record.json').read_text())
    now = datetime.now(timezone.utc).isoformat()
    results = args.out/'results'
    assert not results.exists(), 'Refuse to overwrite a previously built export'
    audit = {'created_at': now, 'snapshot_sha256': digest(snap_path), 'experiments': {}}
    for alg, name in NAMES.items():
        batch = next(b for b in snapshot['batches'].values() if alg in b['manifest']['value']['algorithms'])
        manifest = batch['manifest']['value']
        jobs = [j for j in manifest['jobs'] if j['algorithm'] == alg]
        assert len(jobs) == 80 and len({j['task_rel'] for j in jobs}) == 80
        exp = results/name
        planned, records = [], []
        for job in jobs:
            files = batch['jobs'][job['job_id']]
            terminal = files['terminal.json']['value']
            perf = files.get('evaluation/performance.json')
            discovery = files.get('discovery/result.json', {}).get('value', {})
            relative = Path(job['task_rel']).relative_to('tasks').with_suffix('.json')
            entry = {k: job[k] for k in ('task_index', 'task_name', 'task_rel', 'task_sha256', 'domain',
                                        'task_group', 'probe_names', 'job_id', 'seed', 'replicate', 'attempt')}
            entry.update(result_path=relative.as_posix() if perf else None,
                         outcome=terminal['outcome'], strict_complete=terminal.get('strict_complete') is True,
                         source_terminal_sha256=files['terminal.json']['sha256'])
            planned.append(entry)
            record = {'job_id': job['job_id'], 'task_rel': job['task_rel'],
                      'source_batch': batch['batch'], 'records': {k: v for k, v in files.items()
                       if k in ('terminal.json', 'discovery/freeze.json', 'discovery/result.json')},
                      'accounting': costs(alg, files)}
            records.append(record)
            if not perf:
                continue
            exported = copy.deepcopy(perf['value'])
            submitted = exported.get('evaluation', {}).get('submitted_solution')
            if isinstance(submitted, dict):
                # Native normalized formulas, explicitly not the original submission.txt bytes.
                exported['submission'] = ''.join(f'{key} = {value}\n' for key, value in submitted.items())
            else:
                exported['submission'] = None
            exported['cost_in_USD'] = record['accounting']['total_cost_usd']
            exported['used_token'] = None
            exported['accounting'] = record['accounting']
            exported['run_status'] = {'outcome': terminal['outcome'], 'strict_complete': entry['strict_complete']}
            exported['source'] = {
                'hostname': manifest['evaluation_host'],
                'path': batch['batch']+'/jobs/'+job['job_id']+'/evaluation/performance.json',
                'file_sha256': perf['sha256'], 'task_definition': job['task_rel'],
                'task_definition_sha256': job['task_sha256'], 'manifest_job_id': job['job_id'],
                'manifest_sha256': batch['manifest']['sha256'],
                'snapshot_collected_at_utc': snapshot['collected_at_utc'],
                'snapshot_sha256': digest(snap_path),
                'submission_representation': 'Normalized formulas from evaluation.submitted_solution; original frozen bytes not re-downloaded.',
                'frozen_submission_sha256': files.get('discovery/freeze.json', {}).get('value', {}).get('submission_sha256'),
                'supplemental_records_repo_relative': 'results/'+SUMMARY+'/provenance/'+alg+'.jsonl',
                'transformation': 'Native performance fields preserved; only provenance, normalized submission, run status and stored usage added. No rescoring.'}
            exported['update'] = [{'updated_at': now, 'update_message': 'Import completed Core80 SR baseline from frozen snapshot; preserve native errors and missing values.'}]
            write_json(exp/relative, exported)
        counts = Counter(p['outcome'] for p in planned)
        setup = {'schema_version': 1, 'experiment_name': name, 'algorithm': alg,
                 'pipeline': alg+' + DeepSeek Direct-Ask', 'model': 'deepseek/deepseek-v4-flash-0731',
                 'benchmark': {'name': 'MechBench', 'split': 'core_set', 'task_count': 80,
                               'planned_probe_count': 165, 'domains': ['physics','chemistry','biology','material'],
                               'samples_per_task': 1000, 'replicate': 1, 'attempt': 1},
                 'configuration': {k: manifest[k] for k in ['source_commit','probe_model','probe_reasoning_effort',
                    'probe_temperature','probe_max_output_tokens','probe_request_timeout_seconds',
                    'probe_transport_max_attempts','probe_workers','seed_rule','evaluation_watchdog_seconds']},
                 'symbolic_regression': {'search_seconds': 900, **VENDORS[alg]},
                 'evaluation_environment': {'numpy': '2.3.5', 'sympy': '1.13.1', 'python-flint': '0.9.0'},
                 'source': {'batch_path': batch['batch'], 'manifest_sha256': batch['manifest']['sha256'],
                            'code_commit': source_record['commit'],
                            'code_files': [x for x in source_record['files'] if x['path'] in
                             ['src/evaluate.py','src/scoring.py','src/algorithms/pysr_and_llm_direct.py']]},
                 'data_provenance': 'Reuse frozen Core80 data; Thermoelastic 030/046/054/062/070/078 imported unchanged from collaborator; generating environment/time unknown. No regeneration.',
                 'result_selection': {'policy': 'One attempt per task; import every existing performance.json including invalid results. No placeholders for missing performance; terminal evidence in summary provenance.',
                                      'planned': 80, 'imported': sum(p['result_path'] is not None for p in planned),
                                      'missing': sum(p['result_path'] is None for p in planned),
                                      'strict_complete': sum(p['strict_complete'] for p in planned), 'outcomes': dict(counts)},
                 'metrics': {'P': 'symbolic phenomenal task pass / all planned tasks',
                             'paper_task_SA_M': 'all planned probes symbolically pass / all planned tasks',
                             'legacy_probe_M': 'symbolically passing probes / all planned probes',
                             'missing_policy': 'Zero contribution in aggregates only; native errors/nulls are preserved.',
                             'numeric_warning': 'Native numerically_equivalent is not the paper Acc_0.1 metric.'},
                 'planned_runs': planned, 'updated_at': now}
        if alg == 'llmsr':
            setup['symbolic_regression'].update({k: manifest[k] for k in
             ['generator_request_timeout_seconds','generator_transport_max_attempts']})
        write_json(exp/'setup.json', setup)
        provenance = results/SUMMARY/'provenance'/f'{alg}.jsonl'
        provenance.parent.mkdir(parents=True, exist_ok=True)
        provenance.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n' for r in records))
        (exp/'README.md').write_text(f'''# {setup['pipeline']} — Core80 (2026-11-09)

本目录代表同一次 Core80 实验，80 题均已结束。实际 performance.json 共 {setup['result_selection']['imported']} 份，严格完整原生评分 {setup['result_selection']['strict_complete']} 题；完整评分不代表答对。

遵循 results/README.md，不为缺失 performance 创建占位结果。setup.json 的 planned_runs 列出全部80题、结果路径或缺失原因；原始错误终态和冻结哈希见 ../{SUMMARY}/provenance/{alg}.jsonl。原始运行不重跑、不择优、不拼接不同尝试。

新增 submission 字段来自原生 evaluation.submitted_solution 的规范化方程，不冒充原始冻结文件字节。source.file_sha256 是服务器原 performance.json 的 SHA256。费用仅使用已保存的原始 usage，金额/Token未知保持 null；未用账户余额差估算。

汇总同时提供论文整题 SA(M) 与 Legacy 探针通过比例，二者不能混用；缺失和失败在固定分母汇总中贡献0，原生指标不补造为 False。详见 ../{SUMMARY}/README.md。
''')
        audit['experiments'][alg] = setup['result_selection']
    write_json(results/SUMMARY/'source_code_record.json', source_record)
    write_json(args.out/'export_audit.json', audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
