"""Audit recorded counts/costs offline using Python's standard library only.

No provider, SDK, database, network, reference-label store or model is imported.
This verifies aggregation of recorded judgments, not those judgments themselves.
"""
import argparse
import json
from decimal import Decimal, ROUND_CEILING
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def money(value):
    require(type(value) is int and value >= 0, 'API costs must be nonnegative integer micro-USD')
    return value


def response(value, kind):
    require(set(value) == {'draft_reference_match', 'runtime_check_pass', 'api_cost_microusd'}, 'Unexpected response fields')
    require(type(value['draft_reference_match']) is bool, 'Match must be a recorded boolean')
    require(type(value['runtime_check_pass']) is bool, 'Runtime check must be a boolean')
    money(value['api_cost_microusd'])
    if kind == 'local':
        require(value['api_cost_microusd'] == 0, 'Local API cost must be zero; operating costs are excluded')


def compute(data):
    require(data['format_version'] == 'llm-router-recorded-results-v1', 'Unsupported export format')
    rows = data['evaluation']
    ids = [r['id'] for r in rows]
    require(len(ids) == len(set(ids)) == 100, 'Expected 100 unique evaluation IDs')
    require(set(ids) == {f'E-{i:03d}' for i in range(1, 101)}, 'Unexpected evaluation IDs')
    for row in rows:
        require(set(row) == {'id', 'local', 'cloud', 'checkpoint_choice'}, 'Unexpected evaluation fields')
        response(row['local'], 'local')
        if row['cloud'] is not None:
            response(row['cloud'], 'cloud')
        choices = row['checkpoint_choice']
        require(set(choices) == {'10', '20', '40'}, 'Expected three checkpoints')
        require(all(v in ['local', 'cloud'] for v in choices.values()), 'Unknown model choice')
    cloud_rows = [r for r in rows if r['cloud'] is not None]
    summary = {
        'local_baseline': {
            'completed': len(rows),
            'draft_reference_matches': sum(r['local']['draft_reference_match'] for r in rows),
            'actual_api_cost_microusd': sum(r['local']['api_cost_microusd'] for r in rows),
        },
        'cloud_baseline': {
            'completed': len(cloud_rows),
            'missing_ids': sorted(r['id'] for r in rows if r['cloud'] is None),
            'draft_reference_matches': sum(r['cloud']['draft_reference_match'] for r in cloud_rows),
            'actual_api_cost_microusd': sum(r['cloud']['api_cost_microusd'] for r in cloud_rows),
        },
        'checkpoints': {},
    }
    for checkpoint in ['10', '20', '40']:
        complete = []
        missing = []
        selected = []
        escalated = []
        cost = 0
        same_case_cloud = 0
        for row in rows:
            kind = row['checkpoint_choice'][checkpoint]
            if kind == 'local':
                selected.append(row['id'])
            first = row[kind]
            if first is None:
                missing.append(row['id'])
                continue
            attempts = [first]
            if kind == 'local' and not first['runtime_check_pass']:
                if row['cloud'] is None:
                    missing.append(row['id'])
                    continue
                attempts.append(row['cloud'])
                escalated.append(row['id'])
            complete.append((row, first, attempts[-1]))
            cost += sum(a['api_cost_microusd'] for a in attempts)
            if row['cloud'] is not None:
                same_case_cloud += row['cloud']['api_cost_microusd']
        summary['checkpoints'][checkpoint] = {
            'completed': len(complete),
            'missing_ids': sorted(missing),
            'initial_draft_reference_matches': sum(first['draft_reference_match'] for _, first, _ in complete),
            'final_draft_reference_matches': sum(final['draft_reference_match'] for _, _, final in complete),
            'local_selection_ids': sorted(selected),
            'escalation_ids': sorted(escalated),
            'same_case_cloud_api_cost_microusd': same_case_cloud,
            'counterfactual_replay_api_cost_microusd': cost,
            'counterfactual_savings_microusd': same_case_cloud - cost,
            'actual_replay_spending_added_microusd': 0,
        }
    seeds = data['cloud_seeding']
    require(len(seeds) == 40 and len({s['id'] for s in seeds}) == 40, 'Expected 40 unique cloud seeds')
    require({s['id'] for s in seeds} == {f'S-{i:03d}' for i in range(1, 41)}, 'Unexpected seed IDs')
    for row in seeds:
        require(set(row) == {'id', 'api_cost_microusd'}, 'Unexpected seeding fields')
    seed_cost = sum(money(s['api_cost_microusd']) for s in seeds)
    summary['cloud_seeding'] = {'completed': len(seeds), 'actual_api_cost_microusd': seed_cost}
    checkpoint = summary['checkpoints']['40']
    savings = checkpoint['counterfactual_savings_microusd']
    require(savings > 0, 'No positive saving supports the reported break-even calculation')
    count = checkpoint['completed']
    saving_per_request = Decimal(savings) / Decimal(count)
    break_even = Decimal(seed_cost) / saving_per_request
    summary['api_only_calculation'] = {
        'same_case_requests': count,
        'counterfactual_savings_percent': str(Decimal(savings) * 100 / Decimal(checkpoint['same_case_cloud_api_cost_microusd'])),
        'counterfactual_saving_usd_per_request': str(saving_per_request / 1_000_000),
        'break_even_subsequent_requests': str(break_even),
        'break_even_whole_requests': int(break_even.to_integral_value(rounding=ROUND_CEILING)),
    }
    return summary


def verify_reported_figures(data):
    summary = compute(data)
    require(summary['local_baseline']['draft_reference_matches'] == 58, 'Qwen figure differs from 58/100')
    cloud = summary['cloud_baseline']
    require(cloud['completed'] == 99 and cloud['missing_ids'] == ['E-001'], 'Cloud missing-case denominator differs')
    require(cloud['draft_reference_matches'] == 95, 'Cloud figure differs from 95/99')
    require(cloud['actual_api_cost_microusd'] == 287472, 'Cloud API cost differs from $0.287472')
    for n, row in summary['checkpoints'].items():
        require(row['completed'] == 99 and row['missing_ids'] == ['E-001'], f'Checkpoint {n} denominator differs')
        require(row['initial_draft_reference_matches'] == row['final_draft_reference_matches'] == 95, f'Checkpoint {n} agreement differs')
        require(row['escalation_ids'] == [], f'Checkpoint {n} has an escalation')
        require(row['local_selection_ids'] == (['E-041', 'E-043'] if n == '40' else []), f'Checkpoint {n} selections differ')
        require(row['counterfactual_replay_api_cost_microusd'] == (281686 if n == '40' else 287472), f'Checkpoint {n} cost differs')
    selected = [r for r in data['evaluation'] if r['id'] in ['E-041', 'E-043']]
    require(all(r['local']['draft_reference_match'] and r['local']['runtime_check_pass'] for r in selected), 'Selected local outcomes do not both pass/match')
    require(summary['checkpoints']['40']['counterfactual_savings_microusd'] == 5786, 'Difference differs from $0.005786')
    require(summary['cloud_seeding']['actual_api_cost_microusd'] == 116129, 'Cloud seeding cost differs from $0.116129')
    require(summary['api_only_calculation']['break_even_whole_requests'] == 1987, 'API-only break-even differs from 1,987')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path(__file__).with_name('recorded_results.json'))
    args = parser.parse_args(argv)
    summary = verify_reported_figures(json.loads(args.data.read_text(encoding='utf-8')))
    print(json.dumps({'verification': 'PASS', 'totals': summary}, indent=2))


if __name__ == '__main__':
    main()
