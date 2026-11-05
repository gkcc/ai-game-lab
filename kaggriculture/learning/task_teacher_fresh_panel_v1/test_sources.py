"""Completed-source chain regressions; all data is synthetic and in memory."""
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest

from . import sources


def raw(value):
    return (json.dumps(value, separators=(',', ':')) + '\n').encode()


def chain():
    plan = dict(game='dev-S02-round2-1-control')
    bundle = {k: raw({'receipt': k}) for k in sources.acceptance.FILES}
    job = raw({'job': 'synthetic'})
    supervised = raw(dict(schema='task-pair-dev-job-supervision-v1', game=plan['game'],
        retry_allowed=False, supervision={'synthetic': True}))
    accepted = raw(dict(schema='task-pair-dev-job-acceptance-v1', game=plan['game'],
        passed=True, retry_allowed=False, supervision_sha256=sources.old._sha(supervised),
        job_input_sha256=sources.old._sha(job), receipt_sha256={k:sources.old._sha(v) for k,v in bundle.items()}))
    markers = {k:sources.fresh_entry.marker(k, commit=sources.old.OLD_REGISTRATION_COMMIT,
        binding_sha='a'*64, cohort='C01', game=plan['game']) for k in ('permit','started','consumed')}
    expected = dict(cohort='C01', game=plan['game'], acceptance_sha256=sources.old._sha(accepted),
        supervision_sha256=sources.old._sha(supervised), marker_sha256={k:sources.old._sha(v) for k,v in markers.items()})
    return dict(plan=plan, cohort='C01', commitment='a'*64, expected=expected,
        accepted_raw=accepted, supervision_raw=supervised, bundle=bundle, job_raw=job, markers=markers)


class SourceTests(unittest.TestCase):
    def test_completed_chain_accepts_the_linked_receipt_identity(self):
        self.assertEqual(sources.verify_receipt_chain(**chain()), {'synthetic':True})

    def test_each_link_cannot_change_without_the_prior_completed_receipt(self):
        for key in ('accepted_raw', 'supervision_raw', 'job_raw'):
            value = chain(); value[key] += b' '
            with self.subTest(key=key), self.assertRaises(ValueError):
                sources.verify_receipt_chain(**value)
        for key in sources.acceptance.FILES:
            value = chain(); value['bundle'][key] += b' '
            with self.subTest(receipt=key), self.assertRaises(ValueError):
                sources.verify_receipt_chain(**value)

    def test_missing_receipt_or_changed_source_coordinate_is_rejected(self):
        for key in sources.acceptance.FILES:
            value = chain(); del value['bundle'][key]
            with self.subTest(receipt=key), self.assertRaises(ValueError):
                sources.verify_receipt_chain(**value)
        value = chain(); value['expected']['cohort'] = 'C02'
        with self.assertRaises(ValueError): sources.verify_receipt_chain(**value)

    def test_spliced_marker_is_rejected_even_if_local_marker_hash_is_updated(self):
        value = chain()
        value['markers']['consumed'] = sources.fresh_entry.marker('consumed',
            commit=sources.old.OLD_REGISTRATION_COMMIT, binding_sha='a'*64,
            cohort='C02', game=value['plan']['game'])
        value['expected']['marker_sha256']['consumed'] = sources.old._sha(value['markers']['consumed'])
        with self.assertRaises(ValueError): sources.verify_receipt_chain(**value)

    def test_initial_balances_must_match_both_prior_arms(self):
        plan = dict(seed_alias='S02', opponent='round2', side=1)
        context = SimpleNamespace(rows={('C01', 'dev-S02-round2-1-'+arm):
            {'initial_banks':[80,100]} for arm in ('baseline','control')})
        self.assertEqual(sources.initial_cash_for(context, 'C01', plan), [80,100])
        context.rows[('C01','dev-S02-round2-1-control')]['initial_banks'] = [100,80]
        with self.assertRaises(ValueError): sources.initial_cash_for(context, 'C01', plan)

    def test_public_cash_does_not_replace_authenticated_consumer_values(self):
        own = dict(cash_reconciled=True,initial_cash=10,final_cash=40,net_profit=30,
            sales_units=6,sales_revenue=42,ending_warehouse_units=0,
            successful_plants=3,seed_units_bought=3,ending_seed_units=0)
        other = dict(cash_reconciled=True,initial_cash=10,final_cash=35,net_profit=25,
            sales_units=5,sales_revenue=40,ending_warehouse_units=2,
            successful_plants=3,seed_units_bought=3,ending_seed_units=0)
        metrics = dict(side=0,result=dict(actor=0,official_turns=719,economy=dict(own=own,opponent=other),
            margin=5,score=1.))
        values = sources._game(metrics)
        pair = {'baseline':{k:values[k] for k in sources.PUBLIC_FIELDS}}
        row = dict(net_profit=30,opponent_net_profit=25,margin=5,score=1.)
        sources.verify_public_pair(row,metrics,pair,'baseline')
        changed = deepcopy(pair); changed['baseline']['own_sales_revenue'] += 1
        with self.assertRaises(ValueError): sources.verify_public_pair(row,metrics,changed,'baseline')
        changed = dict(row,net_profit=31)
        with self.assertRaises(ValueError): sources.verify_public_pair(changed,metrics,pair,'baseline')


if __name__ == '__main__':
    unittest.main()
