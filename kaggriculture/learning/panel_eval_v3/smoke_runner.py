"""One public-seed development smoke using the existing official worker.

This is not a holdout, competitive evaluation, training run or promotion.
The old worker necessarily computes and keeps a local private economic row.
Its unchanged acceptance reads and checks those fields; this smoke does not
compare or aggregate earnings, publish them, or use them to select a policy.
Raw rows, replay, trace and worker receipts remain local.
"""
import argparse
import os
from pathlib import Path
import sys
import time

import psutil

from learning.panel_eval_v3 import adapter, isolation
from learning.panel_eval_v3.epoch import EpochContext
from learning.panel_eval_v3.experiment import Context, ROOT, SMOKE, SMOKE_KEY, _same, require_run_kind
from learning.panel_eval_v3.runner import Coordinator, _monotonic_deadline, _sha, canonical_receipts
from learning.panel_safety_v1 import memory_gate, seed_inventory
from learning.task_teacher_merged_eval_v1 import campaign, evaluation, games
from learning.task_training_job_supervision_v1.job import WindowsAPI


def _local_json(out, relative):
    return seed_inventory._json(seed_inventory._read_regular(Path(out)/relative, Path(out)))


def run(stage, prereg_commit, scratch):
    isolation.validate_runtime(scratch,sys.pycache_prefix,ROOT)
    context=EpochContext(Context(stage,prereg_commit))
    cfg=context.refresh()
    require_run_kind(cfg,SMOKE)
    out=context.OUT
    if (out/'results').exists() or (out/'games-started.json').exists():
        raise ValueError('Development smoke attempt already consumed')
    context.write_new(out/'results'/'coordinator-started.json',
        dict(attempt=1,prereg_commit=prereg_commit,run_kind=SMOKE,retry_allowed=False))
    controller=Coordinator(context,scratch,memory_gate.WaitBudget())
    try:
        preflight=memory_gate.wait_for_preflight(budget=memory_gate.WaitBudget(),
            deadline=min(_monotonic_deadline(cfg),time.monotonic()+25),
            available=lambda:psutil.virtual_memory().available,job_not_started=True)
        if not preflight.passed:
            raise RuntimeError('Initial development preflight failed')
        if not _same(context.refresh(),cfg):
            raise ValueError('Development configuration changed')
        with adapter.bind_context(context):
            if not controller.memory_ready(cfg):
                raise RuntimeError('Development resource preflight failed')
            if not _same(context.refresh(),cfg):
                raise ValueError('Development configuration changed during preflight')
            campaign.storage_ready(out,cfg)
            seeds=context.load_seeds(out,cfg)
            plan=evaluation.plans(seeds)[0]
            if plan.get('game') != SMOKE_KEY:
                raise ValueError('Unexpected development game plan')
            state=dict(created=False,complete=False)
            campaign.marker(out,state)
            identities=campaign.identity(out)
            wrapper=games.make_wrapper(out,plan,cfg)
            rival=out/'sources'/(plan['opponent']+'.py')
            _,digest=games.task_identity(out,plan,cfg)
            item=dict(plan,bot_sha256=_sha(wrapper),rival_sha256=_sha(rival),task_checkpoint_sha256=digest)
            games.validate_plan(item,cfg,job=True,out=out)
            context.write_new(out/'job-inputs'/(SMOKE_KEY+'.json'),item)
            supervision=controller.supervise_game(WindowsAPI(),
                [sys.executable,'-B','-m','learning.task_teacher_merged_eval_v1.games',
                 '--out',str(out),'--key',SMOKE_KEY],ROOT,cfg['official_job_supervision'])
            context.write_new(out/'job-supervision'/(SMOKE_KEY+'.json'),
                dict(game=SMOKE_KEY,job_input_sha256=_sha(out/'job-inputs'/(SMOKE_KEY+'.json')),
                     freeze_sha256=identities['freeze.json'],supervision=supervision,error_kind=None,
                     cleanup_state='recorded',retry_allowed=False))
            if not campaign.job_ok(supervision):
                raise RuntimeError('Development worker or cleanup failed')
            campaign.check_identity(out,identities)
            claim=_local_json(out,'results/smoke-worker-claim.json')
            expected_claim=dict(run_kind=SMOKE,game=SMOKE_KEY,wrapper_sha256=item['bot_sha256'],
                                attempt=1,retry_allowed=False)
            if not _same(claim,expected_claim):
                raise ValueError('Development worker claim missing or changed')
            # This unchanged acceptance verifies the official 720 states,
            # 719 actions and 719-step primary replay.  Its private row also
            # contains legacy economic fields, which are deliberately ignored.
            _,audit=controller.accept_job(out,item,cfg)
            receipts=canonical_receipts(out,item)
            if audit.get('official_transitions') != 719:
                raise ValueError('Development primary replay is incomplete')
            if not _same(context.refresh(),cfg):
                raise ValueError('Development configuration changed after match')
            campaign.check_identity(out,identities)
            if canonical_receipts(out,item) != receipts:
                raise ValueError('Development canonical receipts changed')
            paths={
                'worker_complete':out/'job-results'/(SMOKE_KEY+'.complete.json'),
                'worker_audit':out/'job-results'/(SMOKE_KEY+'.audit.json'),
                'worker_storage':out/'job-results'/(SMOKE_KEY+'.storage.json'),
                'replay':out/'games'/SMOKE_KEY/'replay.json.gz',
                'trace':out/'traces'/(SMOKE_KEY+'.jsonl.gz'),
                'job_supervision':out/'job-supervision'/(SMOKE_KEY+'.json'),
            }
            hashes={name:_sha(path) for name,path in paths.items()}
            isolation.validate_runtime(scratch,sys.pycache_prefix,ROOT)
            context.write_new(out/'results'/'coordinator-complete.json',
                dict(passed=True,run_kind=SMOKE,development_only=True,game=SMOKE_KEY,
                     complete_games=1,states=720,action_turns=719,status=['DONE','DONE'],agent_errors=0,
                     primary_replay_transitions=719,extra_replay_transitions=0,
                     economic_evaluation_completed=False,automatic_promotion=False,
                     canonical_receipt_sha256=receipts,local_artifact_sha256=hashes,retry_allowed=False))
    except BaseException as exc:
        try:
            context.write_new(out/'results'/'coordinator-failure.json',
                dict(run_kind=SMOKE,error_kind=type(exc).__name__,retry_allowed=False,
                     cleanup_state=controller.cleanup_state,economic_evaluation_completed=False))
        except BaseException:
            pass
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',required=True)
    parser.add_argument('--prereg',required=True)
    parser.add_argument('--scratch',required=True)
    args=parser.parse_args()
    run(args.out,args.prereg,args.scratch)


if __name__ == '__main__':
    try:
        with open(os.devnull,'w',encoding='utf-8') as sink:
            from contextlib import redirect_stdout, redirect_stderr
            with redirect_stdout(sink), redirect_stderr(sink):
                main()
    except BaseException:
        raise SystemExit(1) from None
    raise SystemExit(0)
