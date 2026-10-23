"""Offline official replay consumption and paired DEVELOPMENT statistics.

No policy selection, model fitting, promotion, or private state publication.
The two replay passes are counted explicitly: original BC control audit, then
economic consumption of the same official transitions for both players.
"""
from collections import Counter
import gzip
import json
import math
from numbers import Real
from pathlib import Path
from statistics import mean


FAMILIES = ('original_student', 'fixed_epoch6')
OPPONENTS = ('v1', 'market', 'round2')
METRICS = ('net_profit', 'opponent_net_profit', 'margin', 'match_score',
           'successful_plants', 'seed_units_bought', 'sales_units',
           'sales_revenue', 'ending_seed_units', 'ending_warehouse_units',
           'opponent_successful_plants', 'opponent_seed_units_bought',
           'opponent_sales_units', 'opponent_sales_revenue',
           'opponent_ending_seed_units', 'opponent_ending_warehouse_units')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, name):
    require(isinstance(value, Real) and not isinstance(value, bool)
            and math.isfinite(value), 'Invalid numeric ' + name)
    return value


def quantity(value, name):
    value = number(value, name)
    require(value >= 0 and int(value) == value, 'Invalid quantity ' + name)
    return int(value)


def operation(request):
    """Malformed protocol requests carry no operation/item classification."""
    if not isinstance(request, list) or not request or not isinstance(request[0], str):
        return None, None
    return request[0], request[1] if len(request) > 1 and isinstance(request[1], str) else None


class EconomicConsumer:
    """Consume tracer EFFECTS; requests are only guarded operation labels."""
    def __init__(self):
        self.players = [dict(counts=Counter(), planted=Counter(), seed_bought=Counter(),
                             sold=Counter(), sales_revenue_by_item=Counter()) for _ in (0, 1)]

    def consume(self, event):
        player = event.get('player')
        require(type(player) is int and player in (0, 1), 'Unknown event player')
        phase, status = event.get('phase'), event.get('status')
        require(phase in ('unit', 'market'), 'Unknown execution phase')
        require(isinstance(status, str) and status != 'error', 'Execution error event')
        p = self.players[player]
        c = p['counts']
        c[phase + '_events'] += 1
        c[phase + '_status_' + status] += 1
        op, item = operation(event.get('requested'))
        if op is None:
            c[phase + '_unclassified_requests'] += 1
        if phase == 'unit':
            # Effective action, not request, determines a real successful plant.
            before, after = event.get('before'), event.get('after')
            require(isinstance(before, dict) and isinstance(after, dict), 'Missing unit resources')
            c['unit_cash_delta'] += (number(after['money'], 'unit after money')
                                     - number(before['money'], 'unit before money'))
            if event.get('successful_plant'):
                executed_op, crop = operation(event.get('effective'))
                require(executed_op == 'PLANT' and crop is not None, 'Unclassified successful plant')
                c['successful_plants'] += 1
                p['planted'][crop] += 1
            return
        units = quantity(event.get('executed_units', 0), 'executed market units')
        cash = number(event.get('cash_delta', 0), 'market cash delta')
        c['market_cash_delta'] += cash
        # Ignored/malformed requests are counted but have no fictitious fills.
        if op is None:
            require(units == 0 and cash == 0, 'Unclassified market effect')
        if op in ('BUY_SEED', 'SELL'):
            require(item is not None or units == 0, 'Missing filled item')
            if op == 'BUY_SEED':
                c['seed_units_bought'] += units
                if units:
                    p['seed_bought'][item] += units
            else:
                require(cash >= 0 and (units > 0 or cash == 0), 'Invalid sale effect')
                c['sales_units'] += units
                c['sales_revenue'] += cash
                if units:
                    p['sold'][item] += units
                    p['sales_revenue_by_item'][item] += cash

    def finish(self, initial_observations, final_observations):
        results = []
        for player, p in enumerate(self.players):
            start, end = initial_observations[player], final_observations[player]
            before = number(start['farms'][player]['money'], 'starting cash')
            after = number(end['farms'][player]['money'], 'ending cash')
            c = p['counts']
            cash_delta = c['unit_cash_delta'] + c['market_cash_delta']
            require(cash_delta == after - before, 'Official cash effects do not reconcile')
            private = end['private']
            seeds = {k: quantity(v, 'ending seed') for k, v in private['seeds'].items()}
            shed = {k: quantity(v, 'ending warehouse') for k, v in private['shed'].items()}
            results.append(dict(
                initial_cash=before, final_cash=after, net_profit=after-before,
                successful_plants=c['successful_plants'], seed_units_bought=c['seed_units_bought'],
                sales_units=c['sales_units'], sales_revenue=c['sales_revenue'],
                ending_seed_units=sum(seeds.values()), ending_warehouse_units=sum(shed.values()),
                ending_seeds_by_crop=seeds, ending_warehouse_by_item=shed,
                planted_by_crop=dict(p['planted']), seed_bought_by_crop=dict(p['seed_bought']),
                sold_by_item=dict(p['sold']), sales_revenue_by_item=dict(p['sales_revenue_by_item']),
                average_sale_price_by_item={k: p['sales_revenue_by_item'][k]/n for k, n in p['sold'].items()},
                cash_reconciled=True, market_cash_delta=c['market_cash_delta'],
                unit_cash_delta=c['unit_cash_delta'], execution_counts=dict(c)))
        return results


def check(job):
    """job=(child_panel_path, ledger_row); returns a sanitized checked row.

    Ledger must carry model_family and stage='development' (or 'entry' for
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
    require(row.get('stage') in ('development', 'entry'), 'Explicit stage required')
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


def summarize(rows, cfg):
    """Require the exact 48-game development Cartesian product; no entries."""
    seeds = cfg['development_seeds']
    require(len(seeds) == 4 and all(type(s) is int for s in seeds) and len(set(seeds)) == 4,
            'Exactly four distinct development seeds required')
    require(len(cfg['opponents']) == 3 and set(cfg['opponents']) == set(OPPONENTS), 'Wrong opponents')
    require(len(cfg['seats']) == 2 and set(cfg['seats']) == {0, 1}
            and all(type(s) is int for s in cfg['seats']), 'Wrong seats')
    require(len(rows) == 48, 'Exactly 48 development rows required')
    expected = {(f, s, o, side) for f in FAMILIES for s in seeds for o in OPPONENTS for side in (0, 1)}
    index, game_ids = {}, set()
    for r in rows:
        require(r.get('stage') == 'development' and not r.get('is_entry', False)
                and not str(r.get('game', '')).startswith('entry'), 'Entry/nondevelopment row in panel')
        require(r.get('policy') == 'trained' and r.get('passed') is True, 'Unverified row')
        require(type(r.get('seed')) is int and type(r.get('side')) is int, 'Invalid paired coordinate')
        key = (r['model_family'], r['seed'], r['opponent'], r['side'])
        require(key in expected and key not in index, 'Duplicate or unexpected panel coordinate')
        require(r.get('checkpoint_sha256') == cfg['model_sha256'][r['model_family']]
                and r.get('control_audit', {}).get('checkpoint_sha256') == r['checkpoint_sha256'],
                'Paired model identity mismatch')
        game = (r['model_family'], r['game'])
        require(game not in game_ids, 'Duplicate family/game identity')
        game_ids.add(game)
        for metric in METRICS:
            number(r[metric], metric)
        require(r['match_score'] == (1. if r['margin'] > 0 else .5 if r['margin'] == 0 else 0.),
                'Match score disagrees with cash margin')
        require(r.get('economic_audit', {}).get('passed') is True
                and r.get('control_audit', {}).get('audit', {}).get('passed') is True,
                'Both official audits required')
        require(r.get('economic', {}).get('own', {}).get('cash_reconciled') is True
                and r.get('economic', {}).get('opponent', {}).get('cash_reconciled') is True,
                'Both cash ledgers must reconcile')
        index[key] = r
    require(set(index) == expected, 'Incomplete paired panel')
    pairs = []
    for seed in seeds:
        for opponent in OPPONENTS:
            for side in (0, 1):
                baseline = index[FAMILIES[0], seed, opponent, side]
                candidate = index[FAMILIES[1], seed, opponent, side]
                pairs.append(dict(seed=seed, opponent=opponent, side=side,
                    original_student_game=baseline['game'], fixed_epoch6_game=candidate['game'],
                    delta={m: candidate[m]-baseline[m] for m in METRICS}))
    def delta(part):
        return {m: mean(r['delta'][m] for r in part) for m in METRICS}
    by_seed = {str(s): delta([p for p in pairs if p['seed'] == s]) for s in seeds}
    by_opponent = {o: delta([p for p in pairs if p['opponent'] == o]) for o in OPPONENTS}
    overall = {m: mean(part[m] for part in by_seed.values()) for m in METRICS}
    policies = {}
    for family in FAMILIES:
        part = [r for r in rows if r['model_family'] == family]
        policies[family] = dict(games=len(part), means={m: mean(r[m] for r in part) for m in METRICS},
            wins=sum(r['margin'] > 0 for r in part), draws=sum(r['margin'] == 0 for r in part),
            losses=sum(r['margin'] < 0 for r in part))
    gate = (all(s['net_profit'] > 0 for s in by_seed.values())
            and all(overall[m] >= 0 for m in ('margin', 'match_score'))
            and all(by_opponent['market'][m] >= 0 for m in ('net_profit', 'margin', 'match_score')))
    return dict(development_only=True, independent_significance_claim=False,
                independent_promotion=False, promotion=False, training_runs=0,
                development_games=48, entry_repeats_in_statistics=0, whole_seed_clusters=4,
                bootstrap_performed=False, exploration_gate_passed=gate,
                fixed_epoch6_minus_original_student=overall, by_seed=by_seed,
                by_opponent=by_opponent, by_model=policies, paired=pairs,
                limitations=['Four development seed clusters; descriptive means only.',
                             'Original fullbatch offline gate remains failed; this does not amend it.',
                             'Requested actions are not fills; sales cash is not final net profit.',
                             'Ending warehouse excludes carried inventories and crops still in fields.'])
