"""Holdout role adapter: unchanged official control audit and economic consumer.

Uses the same two actual replay passes, with explicit holdout/entry roles.
No stage alias, development-statistics reuse, new rules or deployment code.
"""
import gzip
import json
from pathlib import Path
from learning.bc_crop_economic_diagnostic.analysis import EconomicConsumer, FAMILIES, METRICS, require


def check(job):
    """job=(child_panel_path, ledger_row); returns a sanitized checked row.

    Ledger must carry model_family and stage='holdout' (or 'entry' for
    separately counted entry checks). Both use the unchanged trained policy.
    """
    from learning.bc_crop.audit import check as control_check
    from learning.bc_crop.agent import sha
    from decision.audit_market_teacher import observation
    from decision.full_audit import reexecute

    child, row = job
    child = Path(child)
    require(row.get('model_family') in FAMILIES, 'Unknown model family')
    require(child.name == row['model_family'], 'Panel path/model family mismatch')
    registration = json.loads((child/'registration.json').read_text(encoding='utf-8'))
    require(registration['model_family'] == row['model_family'], 'Registration family mismatch')
    expected_model = registration['model_sha256'][row['model_family']]
    require(row.get('checkpoint_sha256') == expected_model, 'Ledger/model registration mismatch')
    require(row.get('policy') == 'trained', 'Unexpected worker policy')
    require(row.get('stage') in ('holdout', 'entry'), 'Explicit stage required')
    require(row.get('passed') is True, 'Unpassed worker game')
    require(type(row.get('side')) is int and row['side'] in (0, 1), 'Invalid side')
    control = control_check((str(child), row))
    require(control['checkpoint_sha256'] == expected_model, 'Audited model identity mismatch')
    replay_path = child/'games'/row['game']/'replay.json.gz'
    require(sha(replay_path) == row['replay_gzip_sha256'], 'Replay changed after control audit')
    with gzip.open(replay_path, 'rt', encoding='utf-8') as stream:
        replay = json.load(stream)
    consumer = EconomicConsumer()
    economic_audit = reexecute(replay, trace=True, label_sink=consumer.consume)
    require(economic_audit == control['audit'], 'Economic/control official audit mismatch')
    initial = [observation(replay['steps'][0], p) for p in (0, 1)]
    final = [observation(replay['steps'][-1], p) for p in (0, 1)]
    players = consumer.finish(initial, final)
    require([p['final_cash'] for p in players] == row['banks'], 'Final bank mismatch')
    side = row['side']
    own, rival = players[side], players[1-side]
    margin = own['final_cash'] - rival['final_cash']
    require(own['net_profit'] == row['net_profit'] and margin == row['margin'], 'Ledger economic mismatch')
    result = {k: row[k] for k in ('game', 'model_family', 'stage', 'policy', 'seed', 'opponent', 'side')}
    result.update(passed=True, control_audit=control, economic_audit=economic_audit,
                  official_replay_passes=2, official_transitions_reexecuted=1438,
                  economic=dict(own=own, opponent=rival),
                  checkpoint_sha256=control['checkpoint_sha256'],
                  net_profit=own['net_profit'], opponent_net_profit=rival['net_profit'],
                  margin=margin, match_score=1. if margin > 0 else .5 if margin == 0 else 0.)
    for key in METRICS[4:10]:
        result[key] = own[key]
        result['opponent_' + key] = rival[key]
    return result

