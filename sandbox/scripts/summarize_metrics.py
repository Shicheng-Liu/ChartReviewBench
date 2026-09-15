"""Read saved episodes; report rule and judge metrics separately (no API)."""
import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def summarize(root):
    values = defaultdict(list)
    count = errors = 0
    for file in sorted(root.glob('*/result.json')):
        result = json.loads(file.read_text())
        count += 1
        rule = result.get('rule_based') or {}
        full = rule.get('full') or {}
        episode_error = bool(result.get('evaluation_errors')) or 'error' in rule
        metrics = {}
        if rule and 'error' not in rule:
            for name in ('executability', 'data_fidelity', 'recovery', 'preservation',
                         'recovery_binary', 'preservation_binary'):
                metrics['rule_' + name] = rule.get(name)
            metrics['rule_data_fidelity_binary'] = (full.get('data_fidelity_binary') or {}).get('strict')
        for sub in result.get('subgoal_results', []):
            details = [v.get('detail', '') for v in sub.get('verifiers', [])]
            bad = any(d.startswith(('SIMULATED', 'verifier error:')) or
                      'judge returned unparseable output' in d for d in details)
            episode_error |= any(d.startswith('verifier error:') or 'judge returned unparseable output' in d for d in details)
            name = {'code_valid': 'artifact_validity', 'flaw_fixed': 'judge_repair',
                    'visual_quality': 'judge_visual_quality'}.get(sub['id'])
            if name and not bad:
                metrics[name] = sub.get('score')
        errors += episode_error
        metrics['agent_turns'] = (result.get('agent_stats') or {}).get('turns_taken')
        metrics['wall_time_s'] = result.get('wall_time_s')
        metrics['python_executions'] = result.get('tool_counts', {}).get('execute_python', 0)
        for role in ('agent', 'judge', 'total'):
            for key in ('input_tokens', 'output_tokens', 'total_tokens'):
                metrics[f'{role}_{key}'] = result.get('token_usage', {}).get(role, {}).get(key)
        cfg_file = file.parent / 'task_config.json'
        if cfg_file.exists() and json.loads(cfg_file.read_text()).get('clean'):
            metrics['clean_preservation'] = rule.get('preservation')
            metrics['clean_preservation_binary'] = rule.get('preservation_binary')
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and math.isfinite(value):
                values[key].append(value)
    return {'episodes': count, 'episodes_with_evaluation_errors': errors,
            'metrics': {k: {'n': len(v), 'mean': statistics.mean(v)} for k, v in sorted(values.items())}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run_root', type=Path)
    args = ap.parse_args()
    roots = [args.run_root] if any(args.run_root.glob('*/result.json')) else [
        p for p in sorted(args.run_root.iterdir()) if p.is_dir() and not p.name.startswith('.')]
    print(json.dumps({p.name: summarize(p) for p in roots}, indent=2))


if __name__ == '__main__':
    main()
