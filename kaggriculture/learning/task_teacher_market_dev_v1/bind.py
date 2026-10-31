"""Bind a new teacher-task diagnostic to the accepted market panel.

Only existing private metadata and receipt hashes are read here; no game,
model, replay or official engine is executed. All output stays local.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess


REPO = Path(__file__).resolve().parents[3]
GAME = REPO.parent.parent / 'kaggriculture'
PRIVATE = REPO.parent / 'private'
BASE = 'be65a2533e2373d0ebf54b1eedfd792a662b40ac'
MARKET_COMMIT = 'e5b613d8b9eab5daa0abeb7d3d160e0f20f877cd'
MARKET_STAGE = 'task-pair-historical-compare-v1-consumed-market-20261031'
MARKET_PRIVATE = 'task-pair-market-consumed-v1-20261031'
PRIVATE_STAGE = 'task-teacher-market-dev-v1-20261031'
PUBLIC_LEDGER = 'review/evidence/task-pair-market-consumed-v1-result-20261031/ledger.json'
PROJECTED_SHA = '78d838482703d535ee4e237572f23e97efb1c2bfa30b70cc603724cdc3d70073'
ENGINE_SHA = 'bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e'
HEX = re.compile(r'[0-9a-f]{64}\Z')
GAMES = tuple(f'dev-{alias}-market-{side}-{arm}' for alias in ('S02', 'S03')
              for side in (0, 1) for arm in ('baseline', 'control'))
ALIASES = {'S02': 'D01', 'S03': 'D02'}
RECEIPTS = {'row': '.row.json', 'audit': '.audit.json', 'complete': '.complete.json',
            'storage': '.storage.json', 'metrics': '.metrics.local.json',
            'metrics_receipt': '.metrics-receipt.local.json'}


def require(value):
    if not value:
        raise ValueError('Prior market identity differs')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path, limit=8 * 1024**2):
    path = Path(path).absolute()
    require(path.resolve(strict=True) == path and path.is_file() and not path.is_symlink()
            and not (getattr(path.lstat(), 'st_file_attributes', 0) & 0x400)
            and path.stat().st_size <= limit)
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    require(0 < len(raw) <= limit)
    return raw


def decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result)
            result[key] = value
        return result
    def invalid(_):
        raise ValueError('Nonfinite JSON')
    value = json.loads(raw.decode('utf-8-sig'), object_pairs_hook=pairs, parse_constant=invalid)
    require(type(value) is dict)
    return value


def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args], stderr=subprocess.DEVNULL)


def completed_run(exit_raw, stdout_raw, stderr_size):
    result = decode(stdout_raw)
    require(type(exit_raw) is bytes and exit_raw.strip() == b'0'
            and type(stderr_size) is int and stderr_size == 0
            and set(result) == {'status', 'games', 'official_replay_passes', 'comparisons',
                                'training', 'independent_improvement', 'promotion'}
            and result['status'] == 'passed'
            and all(type(result[key]) is int and result[key] == expected for key, expected in
                    (('games', 8), ('official_replay_passes', 8), ('comparisons', 4)))
            and all(result[key] is False for key in
                    ('training', 'independent_improvement', 'promotion')))
    return True


def prior_game_identity(name, binding, pair, job, row, audit, complete, metrics, supervision):
    """Check saved game coordinates and public economics, without replaying it."""
    match = re.fullmatch(r'dev-(S02|S03)-market-([01])-(baseline|control)', name)
    require(match is not None)
    alias, side, arm = match.group(1), int(match.group(2)), match.group(3)
    require(pair.get('seed_alias') == ALIASES[alias] and type(pair.get('side')) is int
            and pair['side'] == side and pair.get('opponent') == 'market')
    expected = dict(game=name, seed_alias=alias, seed=binding['seeds'][alias],
                    opponent='market', side=side, policy=arm, split='dev', parity_kind=None,
                    task_checkpoint_sha256=binding['models'][arm]['sha256'],
                    rival_sha256=binding['opponents']['market']['sha256'])
    require(type(job) is dict and all(type(job.get(key)) is type(value) and job[key] == value
                                      for key, value in expected.items())
            and all(type(row.get(key)) is type(value) and row[key] == value
                    for key, value in job.items())
            and row.get('crop_checkpoint_sha256') == binding['crop']['sha256']
            and row.get('engine_sha256') == ENGINE_SHA
            and row.get('passed') is True and row.get('states') == 720
            and row.get('action_turns') == 719 and row.get('status') == ['DONE', 'DONE']
            and type(row.get('agent_errors')) is int and row['agent_errors'] == 0)
    official = audit.get('official')
    require(audit.get('game') == name and audit.get('passed') is True and audit.get('errors') == 0
            and all(type(audit.get(key)) is int and audit[key] == 719 for key in
                    ('actor_actions_reproduced', 'rival_actions_reproduced', 'trace_turns_reproduced'))
            and type(official) is dict and official.get('passed') is True
            and official.get('official_transitions_reexecuted') == 719
            and official.get('both_public_and_private_states_match_each_step') is True
            and official.get('inventory_order_checked_each_step') is True
            and official.get('engine_sha256') == row.get('engine_sha256')
            and official.get('semantic_sha256') == row.get('semantic_replay_sha256')
            and official.get('semantic_sha256_v2') == row.get('semantic_replay_sha256_v2')
            and official.get('final_rewards') == row.get('banks') == row.get('rewards')
            and audit.get('trace_sha256') == row.get('trace_sha256')
            and audit.get('replay_sha256') == row.get('replay_gzip_sha256')
            and complete.get('game') == name and complete.get('passed') is True)
    sup = supervision.get('supervision')
    require(supervision.get('game') == name and supervision.get('retry_allowed') is False
            and type(sup) is dict and all(sup.get(k) is True for k in
                ('passed', 'assigned', 'resumed', 'launcher_reaped', 'job_members_zero'))
            and all(sup.get(k) is None for k in
                ('error_kind', 'cleanup_error_kind', 'stop_reason', 'acceptance_error_kind'))
            and type(sup.get('exit_code')) is int and sup['exit_code'] == 0
            and type(sup.get('rss_complete_samples')) is int and sup['rss_complete_samples'] > 0)
    require(metrics.get('game') == name and metrics.get('seed_alias') == alias
            and type(metrics.get('seed')) is int and metrics['seed'] == binding['seeds'][alias]
            and metrics.get('opponent') == 'market' and type(metrics.get('side')) is int
            and metrics['side'] == side and metrics.get('arm') == arm
            and metrics.get('task_checkpoint_sha256') == expected['task_checkpoint_sha256']
            and metrics.get('crop_checkpoint_sha256') == binding['crop']['sha256']
            and metrics.get('trace_sha256') == row.get('trace_sha256')
            and metrics.get('replay_sha256') == row.get('replay_gzip_sha256'))
    result = metrics.get('result'); public = pair.get('reference' if arm == 'baseline' else 'candidate')
    require(type(result) is dict and type(public) is dict and result.get('actor') == side
            and result.get('official_turns') == 719 and type(result.get('economy')) is dict)
    economy = result['economy']
    for perspective in ('own', 'opponent'):
        values = economy.get(perspective)
        require(type(values) is dict and values.get('cash_reconciled') is True)
        require(type(values.get('initial_cash')) in (int, float)
                and math.isfinite(values['initial_cash'])
                and values.get('net_profit') == values.get('final_cash', 0) - values['initial_cash'])
        for key in ('net_profit', 'final_cash', 'sales_units', 'sales_revenue', 'ending_warehouse_units'):
            value = values.get(key)
            require(type(value) in (int, float) and math.isfinite(value)
                    and type(public.get(perspective + '_' + key)) in (int, float)
                    and public[perspective + '_' + key] == value)
    require(result.get('margin') == public.get('margin') == row.get('margin')
            and result.get('score') == public.get('score') == row.get('score')
            and result['margin'] == economy['own']['final_cash'] - economy['opponent']['final_cash']
            and result['score'] == (1.0 if result['margin'] > 0 else 0.5 if result['margin'] == 0 else 0.0)
            and economy['own']['net_profit'] == row.get('net_profit')
            and economy['opponent']['net_profit'] == row.get('opponent_net_profit')
            and economy['own']['final_cash'] == row.get('bank')
            and economy['opponent']['final_cash'] == row.get('opponent_bank'))
    initial, banks = row.get('initial_banks'), row.get('banks')
    require(type(initial) is list and len(initial) == 2 and type(banks) is list and len(banks) == 2
            and all(type(v) in (int, float) and math.isfinite(v) for v in (*initial, *banks))
            and banks[side] == row.get('bank') and banks[1 - side] == row.get('opponent_bank')
            and initial[side] == row.get('initial_bank') == economy['own']['initial_cash']
            and initial[1 - side] == row.get('opponent_initial_bank') == economy['opponent']['initial_cash'])
    return True


def prior_comparison_identity(data, reference, binding, rows):
    match = re.fullmatch(r'dev-(S02|S03)-market-([01])-baseline', reference)
    require(match is not None)
    alias, side = match.group(1), int(match.group(2))
    candidate = f'dev-{alias}-market-{side}-control'
    comparison = data.get('comparison')
    require(data.get('reference_game') == reference and data.get('candidate_game') == candidate
            and data.get('reference_replay_sha256') == rows[reference]['replay_gzip_sha256']
            and data.get('candidate_replay_sha256') == rows[candidate]['replay_gzip_sha256']
            and data.get('reference_trace_sha256') == rows[reference]['trace_sha256']
            and data.get('candidate_trace_sha256') == rows[candidate]['trace_sha256']
            and type(comparison) is dict and comparison.get('seed_alias') == alias
            and type(comparison.get('seed')) is int and comparison['seed'] == binding['seeds'][alias]
            and type(comparison.get('side')) is int and comparison['side'] == side
            and comparison.get('opponent') == 'market'
            and comparison.get('reference') == 'baseline' and comparison.get('candidate') == 'control')
    return True


def collect(salt):
    require(type(salt) is str and HEX.fullmatch(salt))
    git('merge-base', '--is-ancestor', BASE, 'HEAD')
    source = GAME / 'artifacts' / MARKET_STAGE
    market_binding = read(PRIVATE / MARKET_PRIVATE / 'binding.local.json', 65536)
    previous = decode(read(source / 'comparison-result.local.json'))
    projected_raw = read(source / 'projected-summary.local.json')
    projected = decode(projected_raw)
    ledger_raw = read(REPO / PUBLIC_LEDGER)
    require(sha(projected_raw) == PROJECTED_SHA
            and ledger_raw == git('show', BASE + ':' + PUBLIC_LEDGER)
            and decode(ledger_raw) == projected
            and projected.get('registration_commit') == MARKET_COMMIT
            and projected.get('games') == 8 and projected.get('promotion') is False
            and previous.get('schema') == 'task-pair-market-consumed-private-result-v1'
            and previous.get('registration_commit') == MARKET_COMMIT
            and previous.get('binding_sha256') == sha(market_binding)
            and previous.get('games') == previous.get('official_replay_passes') == 8
            and previous.get('comparisons') == 4 and previous.get('promotion') is False
            and read(source / 'manifest.local.json', 65536) == market_binding)
    attempt = decode(read(PRIVATE / MARKET_PRIVATE / 'attempt.local.json', 65536))
    require(attempt.get('kind') == 'attempt' and attempt.get('registration_commit') == MARKET_COMMIT
            and attempt.get('binding_sha256') == sha(market_binding) and attempt.get('retry_allowed') is False)
    run_folder = PRIVATE / MARKET_PRIVATE
    exit_raw = read(run_folder / 'campaign-exit.local.txt', 64)
    stdout_raw = read(run_folder / 'campaign-stdout.local.log', 65536)
    stderr = run_folder / 'campaign-stderr.local.log'
    require(stderr.resolve(strict=True) == stderr and stderr.is_file()
            and not stderr.is_symlink()
            and not (getattr(stderr.lstat(), 'st_file_attributes', 0) & 0x400))
    completed_run(exit_raw, stdout_raw, stderr.stat().st_size)
    receipts = previous.get('game_receipts')
    require(type(receipts) is list and len(receipts) == 8
            and {item.get('game') for item in receipts if type(item) is dict} == set(GAMES))
    folder = source / 'job-results'
    market = decode(market_binding)
    require(type(projected.get('pairs')) is list and len(projected['pairs']) == 4)
    pairs = {(p.get('seed_alias'), p.get('side')): p for p in projected['pairs'] if type(p) is dict}
    require(len(pairs) == 4 and set(pairs) == {(alias, side) for alias in ALIASES.values() for side in (0, 1)})
    prior_rows = {}
    for item in receipts:
        name = item['game']
        sup_raw = read(folder / (name + '.supervision.local.json'), 512 * 1024)
        accepted_raw = read(folder / (name + '.acceptance.local.json'), 512 * 1024)
        accepted = decode(accepted_raw)
        job_raw = read(source / 'job-inputs' / (name + '.json'), 512 * 1024)
        bundle = {key: read(folder / (name + suffix), 8 * 1024**2) for key, suffix in RECEIPTS.items()}
        require(sha(sup_raw) == item['supervision_sha256']
                and sha(accepted_raw) == item['acceptance_sha256']
                and accepted.get('schema') == 'task-pair-dev-job-acceptance-v1'
                and accepted.get('game') == name and accepted.get('passed') is True
                and accepted.get('retry_allowed') is False
                and accepted.get('supervision_sha256') == sha(sup_raw)
                and accepted.get('job_input_sha256') == sha(job_raw)
                and accepted.get('receipt_sha256') == {key: sha(raw) for key, raw in bundle.items()}
                and set(item['marker_sha256']) == {'permit', 'started', 'consumed'})
        values = {key: decode(raw) for key, raw in bundle.items()}
        require(values['complete'].get('row_sha256') == sha(bundle['row'])
                and values['complete'].get('certificate_sha256') == sha(bundle['audit'])
                and values['storage'].get('game') == name and values['storage'].get('passed') is True
                and values['storage'].get('error_kind') is None
                and values['metrics_receipt'].get('game') == name
                and values['metrics_receipt'].get('official_replay_passes') == 1
                and values['metrics_receipt'].get('metrics_sha256') == sha(bundle['metrics']))
        match = re.fullmatch(r'dev-(S02|S03)-market-([01])-(baseline|control)', name)
        require(match is not None)
        prior_game_identity(name, market, pairs[(ALIASES[match.group(1)], int(match.group(2)))],
                            decode(job_raw), values['row'], values['audit'], values['complete'],
                            values['metrics'], decode(sup_raw))
        prior_rows[name] = values['row']
        for kind, expected in item['marker_sha256'].items():
            raw = read(folder / (name + '.' + kind + '.local.json'), 65536)
            mark = decode(raw)
            require(sha(raw) == expected and mark.get('kind') == kind and mark.get('game') == name
                    and mark.get('registration_commit') == MARKET_COMMIT
                    and mark.get('binding_sha256') == sha(market_binding)
                    and mark.get('retry_allowed') is False)
    comparisons = previous.get('comparison_receipts')
    require(type(comparisons) is list and len(comparisons) == 4)
    expected_reference = {f'dev-{alias}-market-{side}-baseline'
                          for alias in ('S02', 'S03') for side in (0, 1)}
    require({item.get('reference_game') for item in comparisons if type(item) is dict} == expected_reference)
    for item in comparisons:
        name = item['reference_game'] + '-to-control.comparison.local.json'
        raw = read(folder / name, 512 * 1024)
        data = decode(raw)
        reference = item['reference_game']
        require(sha(raw) == item['sha256'])
        prior_comparison_identity(data, reference, market, prior_rows)
    return dict(schema='task-teacher-market-dev-binding-v1', base_commit=BASE, salt=salt,
        market_binding_sha256=sha(market_binding), market_result_sha256=sha(read(source / 'comparison-result.local.json')),
        market_projection_sha256=sha(projected_raw), market_attempt_sha256=sha(read(PRIVATE / MARKET_PRIVATE / 'attempt.local.json', 65536)),
        market_exit_sha256=sha(exit_raw), market_stdout_sha256=sha(stdout_raw),
        bind_source_sha256=sha(read(Path(__file__))), entry_source_sha256=sha(read(Path(__file__).with_name('entry.py'))))


def prepare():
    require(git('rev-parse', 'HEAD').decode().strip() == BASE
            and not git('diff', '--name-only', '-z').strip(b'\0')
            and not git('diff', '--cached', '--name-only', '-z').strip(b'\0'))
    untracked = [x.decode('utf8') for x in git('ls-files', '--others', '--exclude-standard', '-z').split(b'\0') if x]
    require(untracked and all(x.startswith('kaggriculture/learning/task_teacher_market_dev_v1/') for x in untracked))
    document = collect(os.urandom(32).hex())
    folder = PRIVATE / PRIVATE_STAGE
    folder.mkdir(exist_ok=False)
    raw = (json.dumps(document, separators=(',', ':'), allow_nan=False) + '\n').encode()
    with (folder / 'binding.local.json').open('xb') as stream:
        require(stream.write(raw) == len(raw))
    with (folder / 'binding-sha256.local.txt').open('xb') as stream:
        require(stream.write((sha(raw) + '\n').encode()) == 65)
    print(json.dumps({'status': 'bound', 'prior_games': 8, 'prior_pairs': 4}))


def main():
    argparse.ArgumentParser().parse_args()
    try:
        prepare()
    except BaseException:
        print(json.dumps({'status': 'failed', 'phase': 'prior_binding'}))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
