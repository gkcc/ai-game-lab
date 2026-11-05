"""Read only the completed F01--F04 campaign, never the seed inventory.

The old six-receipt acceptance and compressed file hashes are checked again;
no old replay or trace is decoded here and no old game is reexecuted. All
returned values, including identities and the original cash rows, are private.
"""
from dataclasses import dataclass

from learning.task_pair_dev_campaign_v1 import acceptance, driver
from learning.task_pair_dev_campaign_v1.manifest import build_manifest
from learning.task_pair_dev_games_v1 import panel
from learning.task_pair_dev_metrics_v1.panel import _game
from learning.task_pair_fresh_multi_v1 import entry as fresh_entry, prepare as fresh
from learning.task_pair_fresh_multi_v1.projection import FIELDS as PUBLIC_FIELDS
from learning.task_student_continuation_v1 import sources as old
from . import contract


RESULT_COMMIT = '0e9734a0109a0ee4822ef1310e8b56fac21eaa48'
LEDGER = 'review/evidence/task-pair-fresh-multi-v1-result-20261031/ledger.json'
ROLES = ('F01', 'F02', 'F03', 'F04')


@dataclass(repr=False)
class Completed:
    manifests: dict
    prior_public: dict
    rows: dict
    identities: dict

    def __repr__(self):
        return '<CompletedTeacherComparisonSources>'


def verify_receipt_chain(*, plan, cohort, commitment, expected, accepted_raw,
                         supervision_raw, bundle, job_raw, markers):
    """Authenticate the old summary chain before its existing accept() call."""
    require = contract.require
    accepted = old._json(accepted_raw, old.SMALL_MAX)
    supervision = old._json(supervision_raw, old.SMALL_MAX)
    require(type(expected) is dict and expected.get('cohort') == cohort
        and expected.get('game') == plan['game']
        and old._sha(accepted_raw) == expected.get('acceptance_sha256')
        and old._sha(supervision_raw) == expected.get('supervision_sha256')
        and accepted.get('schema') == 'task-pair-dev-job-acceptance-v1'
        and accepted.get('game') == plan['game'] and accepted.get('passed') is True
        and accepted.get('retry_allowed') is False
        and accepted.get('supervision_sha256') == old._sha(supervision_raw)
        and accepted.get('job_input_sha256') == old._sha(job_raw)
        and accepted.get('receipt_sha256') == {k: old._sha(v) for k, v in bundle.items()}
        and set(bundle) == acceptance.FILES
        and supervision.get('schema') == 'task-pair-dev-job-supervision-v1'
        and supervision.get('game') == plan['game'] and supervision.get('retry_allowed') is False
        and type(markers) is dict and set(markers) == {'permit', 'started', 'consumed'})
    for kind, raw in markers.items():
        require(raw == fresh_entry.marker(kind, commit=old.OLD_REGISTRATION_COMMIT,
            binding_sha=commitment, cohort=cohort, game=plan['game']))
    require(expected.get('marker_sha256') == {k: old._sha(v) for k, v in markers.items()})
    return supervision.get('supervision')


def verify_public_pair(row, metrics, pair, arm):
    """Compare verified cash/consumer summaries with the already public ledger."""
    values = _game(metrics)
    actual = {field: values[field] for field in PUBLIC_FIELDS}
    contract.require(type(pair) is dict and type(pair.get(arm)) is dict
        and set(actual) == set(pair[arm]) and actual == pair[arm]
        and actual['own_net_profit'] == row['net_profit']
        and actual['opponent_net_profit'] == row['opponent_net_profit']
        and actual['margin'] == row['margin'] and actual['score'] == row['score'])


def load(*, guard, verify_game_files=True):
    """Authenticate only this explicitly completed 32-game source set.

    ``verify_game_files=False`` is for a child whose parent already bound the
    compressed file hashes. It still authenticates every summary/receipt and
    immutable job identity; the child caller additionally hashes registered
    runtime/model assets. It never relaxes the parent's full file checks.
    """
    contract.require(callable(guard) and type(verify_game_files) is bool)
    guard()
    registration_raw = old._registration_blob()
    registration = old._json(registration_raw, old.RESULT_MAX)
    ledger_raw = fresh._blob(LEDGER, RESULT_COMMIT)
    ledger = old._json(ledger_raw, old.RESULT_MAX)
    contract.require(ledger.get('schema') == 'task-pair-fresh-multi-public-v1'
        and ledger.get('registration_commit') == old.OLD_REGISTRATION_COMMIT
        and ledger.get('games') == 32 and ledger.get('official_replays') == 32
        and type(ledger.get('pairs')) is list and len(ledger['pairs']) == 16)
    identities = {'registration': old._sha(registration_raw), 'public_ledger': old._sha(ledger_raw)}

    def read(root, path, maximum=old.RESULT_MAX):
        return old._small(root, path, maximum, guard)

    binding_raw = read(fresh.PRIVATE_ROOT, fresh.PRIVATE / 'binding.local.json')
    market_raw = read(fresh.PRIVATE_ROOT, fresh.MARKET_PRIVATE / 'binding.local.json')
    campaign_raw = read(fresh.PRIVATE_ROOT, fresh.PRIVATE / 'campaign-result.local.json')
    closed_raw = read(fresh.PRIVATE_ROOT, fresh.PRIVATE / 'closed.local.json')
    binding, market, campaign, closed = (old._json(v, old.RESULT_MAX)
        for v in (binding_raw, market_raw, campaign_raw, closed_raw))
    commitment = old._sha(binding_raw)
    contract.require(binding.get('market_binding_sha256') == old._sha(market_raw)
        and binding.get('cohorts') == contract.COHORTS
        and type(binding.get('seed_roles')) is dict and set(binding['seed_roles']) == set(ROLES))
    seed_values = list(binding['seed_roles'].values())
    contract.require(all(type(s) is int and 0 <= s < 2**32 for s in seed_values)
        and len(set(seed_values)) == 4)
    accepted = old._verify_campaign(registration, binding_raw, market_raw, campaign, closed)
    receipts = {(r['cohort'], r['game']): r for r in campaign['receipts']}
    identities.update(binding=commitment, market=old._sha(market_raw),
        campaign=old._sha(campaign_raw), closed=old._sha(closed_raw))
    contract.require(old._rule_sha(guard) == panel.ENGINE_SHA)
    pairs = {}
    for pair in ledger['pairs']:
        key = (pair.get('seed_alias'), pair.get('opponent'), pair.get('side'))
        contract.require(key not in pairs and key[0] in ROLES and key[1] in contract.OPPONENTS
            and type(key[2]) is int and key[2] in contract.SIDES)
        pairs[key] = pair
    contract.require(len(pairs) == 16)
    manifests, rows, game_ids = {}, {}, {}
    for cohort, mapping in contract.COHORTS.items():
        manifest = build_manifest(seeds={a: binding['seed_roles'][r] for a, r in mapping.items()},
            models=market['models'], crop=market['crop'], opponents=market['opponents'],
            worker_sha256=market['worker_sha256'])
        manifests[cohort] = manifest
        stage = fresh.GAME / 'artifacts' / fresh_entry.STAGES[cohort]
        local = driver.LocalDriver(manifest=manifest, game_root=fresh.GAME, stage=stage,
            commit=old.OLD_REGISTRATION_COMMIT, command_for=lambda _name: [], check=guard)
        for plan in manifest.panel.plans():
            if plan['policy'] not in contract.ARMS:
                continue
            guard()
            name, key = plan['game'], (cohort, plan['game'])
            bundle = local.read_bundle(plan)
            job_raw = read(stage, stage / 'job-inputs' / (name + '.json'), 16384)
            job = manifest.panel.validate_plan(old._json(job_raw, 16384), job=True)
            accepted_raw = read(stage, stage / 'job-results' / (name + '.acceptance.local.json'), old.SMALL_MAX)
            supervision_raw = read(stage, stage / 'job-results' / (name + '.supervision.local.json'), old.SMALL_MAX)
            markers = {k: read(stage, stage / 'job-results' / (name + '.' + k + '.local.json'), 65536)
                       for k in ('permit', 'started', 'consumed')}
            supervision = verify_receipt_chain(plan=plan, cohort=cohort, commitment=commitment,
                expected=receipts[key], accepted_raw=accepted_raw, supervision_raw=supervision_raw,
                bundle=bundle, job_raw=job_raw, markers=markers)
            contract.require(old._sha(accepted_raw) == accepted[key])
            def files(p, row):
                guard()
                contract.require(all(type(row.get(k)) is type(v) and row[k] == v for k, v in job.items()))
                result = local.verify_files(p, row) if verify_game_files else True
                guard()
                return result
            metrics = acceptance.accept(manifest.panel, plan, bundle, supervision, files)
            row = old._json(bundle['row'], old.SMALL_MAX)
            pair = pairs[(mapping[plan['seed_alias']], plan['opponent'], plan['side'])]
            verify_public_pair(row, metrics, pair, plan['policy'])
            rows[key] = row
            game_ids[cohort + '/' + name] = dict(acceptance=old._sha(accepted_raw),
                supervision=old._sha(supervision_raw), job=old._sha(job_raw),
                receipts={k: old._sha(v) for k, v in bundle.items()},
                markers={k: old._sha(v) for k, v in markers.items()})
    contract.require(len(rows) == 32 and set(rows) == set(accepted))
    identities['games'] = game_ids
    guard()
    return Completed(manifests, ledger, rows, identities)


def initial_cash_for(context, cohort, plan):
    """Require both old student arms to agree before checking a new result."""
    values = [context.rows[(cohort, f"dev-{plan['seed_alias']}-{plan['opponent']}-{plan['side']}-{arm}")]
              ['initial_banks'] for arm in contract.ARMS]
    contract.require(values[0] == values[1] and type(values[0]) is list and len(values[0]) == 2)
    return list(values[0])
