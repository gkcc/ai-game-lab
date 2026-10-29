"""Real output-budget compatibility with no official game execution.

The full worker-path test uses synthetic plans and patches unrelated frozen
identity inputs.  It reaches an explicit sentinel in place of official_entry;
it is not a complete game or proof that any new holdout is independent.
"""
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from learning.panel_safety_v1.memory_gate import MIN_AVAILABLE_BYTES
from learning.task_anchor_once_v1 import storage
from learning.task_teacher_merged_eval_v1 import evaluation, games
from . import epoch, experiment as e, prepare, prior_v2_guard, worker


def pinned_loader():
    return e.old.read(e.old.MERGE/'registration.json')['official_loader_sha256']


class RuntimeBudgetTests(unittest.TestCase):
    def test_real_official_identity_inside_original_budget_preserves_write_guard(self):
        loader=pinned_loader()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); stage=root/'output'; stage.mkdir()
            protected=root/'outside'; protected.write_bytes(b'unchanged')
            with storage.owned_writes(stage,1024*1024) as budget:
                e._check_official_runtime(loader)
                with self.assertRaises(storage.UnsafeWriteError): protected.write_bytes(b'changed')
            self.assertEqual(protected.read_bytes(),b'unchanged')
            self.assertEqual(budget.used_bytes,0)

    def test_real_git_error_is_caught_without_disclosing_output_or_path(self):
        loader=pinned_loader()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); stage=root/'output'; stage.mkdir()
            absent=root/'SYNTHETIC_PRIVATE_PATH_MARKER'
            with patch.object(e,'ROOT',absent), storage.owned_writes(stage,1024*1024):
                with self.assertRaises(ValueError) as raised: e._check_official_runtime(loader)
            text=''.join(traceback.format_exception(raised.exception))
            self.assertNotIn('SYNTHETIC_PRIVATE_PATH_MARKER',text)
            self.assertIn('Official runtime identity unavailable',text)

    def test_git_timeout_is_sanitized_under_budget(self):
        loader=pinned_loader()
        timeout=subprocess.TimeoutExpired(['git','SYNTHETIC_PRIVATE_MARKER'],15,
                                          output=b'SYNTHETIC_PRIVATE_MARKER')
        with tempfile.TemporaryDirectory() as temp:
            stage=Path(temp)/'output'; stage.mkdir()
            with patch.object(e.subprocess,'check_output',side_effect=timeout), storage.owned_writes(stage,1024*1024):
                with self.assertRaises(ValueError) as raised: e._check_official_runtime(loader)
            text=''.join(traceback.format_exception(raised.exception))
            self.assertNotIn('SYNTHETIC_PRIVATE_MARKER',text)
            self.assertIn('Official runtime identity unavailable',text)

    def test_old_worker_reaches_sentinel_through_v3_context_inside_budget(self):
        loader=pinned_loader()
        with tempfile.TemporaryDirectory() as temp:
            artifacts=Path(temp); stage=artifacts/'synthetic-stage'; stage.mkdir()
            for name in ('sources','agents','job-inputs'):
                (stage/name).mkdir()
            opponents={}
            for name in ('round2','market','teacher'):
                path=stage/'sources'/(name+'.py'); path.write_text('synthetic source\n')
                opponents[name+'.py']=path
            seeds=[101,202,303,404]
            document={'version':1,'seeds':seeds,'salt':'ab'*32}
            raw=json.dumps(document).encode()
            (stage/'private-seeds.local.json').write_bytes(raw)
            deadline=(datetime.now(timezone.utc)+timedelta(minutes=5)).isoformat()
            cfg={'deadline_utc':deadline,'seed_document_sha256':hashlib.sha256(raw).hexdigest(),
                 'seed_commitment':e.old.seed_commitment(document),'official_loader_sha256':loader,
                 'source_sha256':{name:e.old.sha(path) for name,path in opponents.items()},
                 'storage':{'game_write_bytes':1024*1024,'small_receipt_bytes':16384}}
            cfg.update(e.run_contract(e.FORMAL))
            for name,value in (('registration.json',cfg),('root-decision.json',{'deadline_utc':deadline}),
                               ('freeze.json',{}),('pre-review.json',{})):
                (stage/name).write_text(json.dumps(value),encoding='utf-8')
            plan=next(p for p in evaluation.plans(seeds) if p['seed_alias']=='S01'
                      and p['opponent']=='round2' and p['side']==0 and p['policy']=='teacher_task')
            key=plan['game']; bot=stage/'agents'/(key+'.py'); bot.write_text('synthetic bot\n')
            plan.update(bot_sha256=e.old.sha(bot),rival_sha256=e.old.sha(opponents['round2.py']),
                        task_checkpoint_sha256=None)
            (stage/'job-inputs'/(key+'.json')).write_text(json.dumps(plan),encoding='utf-8')
            events=[]; runtime_checks=[]
            class SentinelReached(Exception): pass
            def sentinel():
                events.append('entry')
                raise SentinelReached()
            original=e._check_official_runtime
            def checked_runtime(value):
                runtime_checks.append(storage._ACTIVE is not None)
                return original(value)
            args=['worker','--out',str(stage),'--key',key,'--prereg','a'*40,
                  '--scratch',str(artifacts),'--mode','game']
            parent_cfg={'official_loader_sha256':loader}
            prior=e.history.SeedInventory([901,902,903,904],0,0)
            with ExitStack() as patches:
                for target,name,value in (
                    (e.old,'trusted_artifacts',lambda:artifacts),
                    (prior_v2_guard,'require_v2_consumed_chain',lambda *args:prior),
                    (e.registered_rescan,'rescan_registered',lambda *args:prior),
                    (e,'_reference_freeze',lambda:{}),
                    (e.old,'historical_inputs',lambda:(parent_cfg,{},{},opponents)),
                    (e.old,'read',lambda path:{}),
                    (e.old,'environment',lambda:{}),
                    (e,'_validate_contract',lambda *args:None),
                    (e,'_validate_helper_review',lambda *args:None),
                    (e,'_validate_sources',lambda *args:None),
                    (e.old,'clean_head',lambda parent:None),
                    (epoch.bootstrap,'verify_initializers',lambda root:None),
                    (epoch.EpochContext,'_identity_paths',lambda:(cfg,frozenset())),
                    (epoch.EpochContext,'_fingerprints',lambda paths:{}),
                    (e,'_check_official_runtime',checked_runtime),
                    (games.core,'official_entry',sentinel),
                ):
                    patches.enter_context(patch.object(target,name,side_effect=value))
                patches.enter_context(patch.object(sys,'argv',args))
                patches.enter_context(patch('learning.panel_eval_v3.isolation.validate_runtime'))
                patches.enter_context(patch('psutil.virtual_memory',return_value=SimpleNamespace(available=MIN_AVAILABLE_BYTES)))
                with self.assertRaises(SentinelReached): worker.main()
            self.assertEqual(events,['entry'])
            self.assertIn(True,runtime_checks)
            self.assertFalse((stage/'games').exists())
            receipt=json.loads((stage/'job-results'/(key+'.storage.json')).read_text(encoding='utf-8'))
            self.assertEqual(receipt['error_kind'],'SentinelReached')
            self.assertFalse(receipt['passed'])


class ConsumedSeedGuardTests(unittest.TestCase):
    def test_v3_preselection_stops_before_inventory_or_any_seed(self):
        deadline=(datetime.now(timezone.utc)+timedelta(minutes=5)).isoformat()
        with patch.object(prior_v2_guard,'require_v2_consumed_chain',side_effect=ValueError('prior unavailable')), \
             patch.object(prepare.rescan,'preselection_inventory') as inventory, \
             patch.object(prepare.secrets,'randbelow') as random, \
             self.assertRaises(ValueError):
            prepare.prepare('synthetic-stage',deadline,'synthetic-helper',run_kind=e.FORMAL)
        inventory.assert_not_called(); random.assert_not_called()

    def test_v3_context_stops_before_registered_rescan(self):
        with tempfile.TemporaryDirectory() as temp:
            artifacts=Path(temp); stage=artifacts/'synthetic-stage'; stage.mkdir()
            deadline=(datetime.now(timezone.utc)+timedelta(minutes=5)).isoformat()
            (stage/'registration.json').write_text(json.dumps({'deadline_utc':deadline}),encoding='utf-8')
            (stage/'root-decision.json').write_text(json.dumps({'deadline_utc':deadline}),encoding='utf-8')
            with patch.object(e.old,'trusted_artifacts',return_value=artifacts), \
                 patch.object(prior_v2_guard,'require_v2_consumed_chain',side_effect=ValueError('prior unavailable')), \
                 patch.object(e.registered_rescan,'rescan_registered') as rescan:
                context=e.Context(stage,'a'*40)
                with self.assertRaises(ValueError): context.ready()
                rescan.assert_not_called()
                with self.assertRaises(ValueError): context.load_seeds(stage,{})


if __name__ == '__main__':
    unittest.main()
