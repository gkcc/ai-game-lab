"""One supervised extra official replay and an alias-only final evaluation."""
from copy import deepcopy
import json
from pathlib import Path
import sys

from learning.task_choice_v1.features import OPS
from learning.task_training_job_supervision_v1.job import WindowsAPI
from learning.task_job_empty_exit_v1.sampling import supervise
from .experiment import OUT, ROOT, load_seeds, read, ready, sha, trusted_out, verify_commit, write_new

BASE_IDENTITIES=('freeze.json','registration.json','pre-execution-commit.json','games-started.json',
                 'games-result.json','game-ledger.json','game-audit-ledger.json','private-seeds.local.json')
RUN_IDENTITIES=BASE_IDENTITIES+('independent-runner-started.json',)


def validate_verification(out,cfg,verification):
    if not isinstance(verification,dict): raise ValueError('Independent verification missing')
    out=Path(out).resolve(); reads=verification.get('raw_reads',{})
    game='holdout-S01-round2-0-merged'
    row_path=out/'job-results'/(game+'.row.json'); audit_path=out/'job-results'/(game+'.audit.json')
    row=read(row_path)
    bindings=dict(freeze_sha256=sha(out/'freeze.json'),registration_sha256=sha(out/'registration.json'),
        games_result_sha256=sha(out/'games-result.json'),game_ledger_sha256=sha(out/'game-ledger.json'),
        audit_ledger_sha256=sha(out/'game-audit-ledger.json'),method_sha256=sha(Path(__file__).with_name('independent_method.py')),
        checkpoint_sha256=cfg['merged_checkpoint_sha256'],crop_checkpoint_sha256=cfg['crop_checkpoint_sha256'],
        seed_document_sha256=cfg['seed_document_sha256'],game=game,engine_sha256=cfg['engine_sha256'],
        semantic_sha256=row['semantic_replay_sha256'],semantic_sha256_v2=row['semantic_replay_sha256_v2'],
        game_row_sha256=sha(row_path),game_audit_sha256=sha(audit_path),
        selection_rule=cfg['independent_selection'])
    counts=verification.get('sample_operation_counts'); selected_hash=verification.get('sample_identity_sha256')
    if (verification.get('passed') is not True or any(verification.get(k)!=v for k,v in bindings.items())
        or verification.get('official_transitions')!=719 or verification.get('sample_count')!=20 or verification.get('matched')!=20
        or not isinstance(reads,dict) or type(reads.get('files_opened')) is not int or reads['files_opened']!=2
        or type(reads.get('files_completed')) is not int or reads['files_completed']!=2
        or type(reads.get('raw_decode_passes_per_file')) is not int or reads['raw_decode_passes_per_file']!=1
        or not isinstance(counts,dict) or set(counts)!=set(OPS)
        or any(type(v) is not int or v<0 for v in counts.values()) or sum(counts.values())!=20
        or not isinstance(selected_hash,str) or len(selected_hash)!=64 or any(c not in '0123456789abcdef' for c in selected_hash)
        or verification.get('both_public_and_private_states_match_each_step') is not True
        or verification.get('inventory_order_checked_each_step') is not True
        or verification.get('final_rewards_match') is not True or verification.get('terminal_status_done') is not True
        or verification.get('no_model_inference') is not True or verification.get('no_new_games') is not True
        or verification.get('retry_allowed') is not False):
        raise ValueError('Independent verification does not bind current panel')


def validate_job(result):
    return (isinstance(result,dict) and result.get('passed') is True and result.get('assigned') is True
        and result.get('resumed') is True and result.get('launcher_reaped') is True
        and result.get('job_members_zero') is True and result.get('error_kind') is None
        and result.get('cleanup_error_kind') is None and result.get('stop_reason') is None
        and type(result.get('exit_code')) is int and result['exit_code']==0)


def validate_completed(out,cfg):
    from .campaign import validate_panel
    out=Path(out).resolve()
    if verify_commit(out)!=cfg: raise ValueError('Independent configuration changed')
    if any((out/name).exists() for name in ('independent-runner-failure.json','independent-failure.local.json')):
        raise ValueError('Independent failure evidence present')
    result=read(out/'independent-runner-result.json'); verification=read(out/'independent-replay-verification.json')
    attempt=read(out/'independent-attempt.local.json'); review=read(out/'independent-result-review.json')
    review_bindings=dict(freeze_sha256=sha(out/'freeze.json'),registration_sha256=sha(out/'registration.json'),
        games_result_sha256=sha(out/'games-result.json'),game_ledger_sha256=sha(out/'game-ledger.json'),
        audit_ledger_sha256=sha(out/'game-audit-ledger.json'))
    if (result.get('passed') is not True or not validate_job(result.get('supervision'))
        or result.get('freeze_sha256')!=sha(out/'freeze.json') or result.get('retry_allowed') is not False
        or result.get('verification_sha256')!=sha(out/'independent-replay-verification.json')
        or result.get('attempt_sha256')!=sha(out/'independent-attempt.local.json')
        or result.get('review_sha256')!=sha(out/'independent-result-review.json')
        or attempt!=dict(attempt=1,retry_allowed=False,freeze_sha256=sha(out/'freeze.json'))
        or review.get('passed') is not True or review.get('selected_game')!='holdout-S01-round2-0-merged'
        or review.get('checkpoint_sha256')!=cfg['merged_checkpoint_sha256']
        or any(review.get(k)!=v for k,v in review_bindings.items())
        or review.get('verification_sha256')!=sha(out/'independent-replay-verification.json')
        or review.get('method_sha256')!=sha(Path(__file__).with_name('independent_method.py'))
        or review.get('official_transitions')!=719 or review.get('sample_count')!=20 or review.get('sample_matched')!=20):
        raise ValueError('Independent replay not accepted')
    identities=result.get('identities')
    if (not isinstance(identities,dict) or set(identities)!=set(RUN_IDENTITIES)
        or any(sha(out/name)!=value for name,value in identities.items())
        or identities['private-seeds.local.json']!=cfg['seed_document_sha256']):
        raise ValueError('Independent identities changed')
    validate_verification(out,cfg,verification); validate_panel(out,cfg)
    return verification


def run(out=OUT):
    from .campaign import memory_ready, storage_ready, validate_panel
    requested_out=out; out=trusted_out(requested_out); cfg=ready(out); verify_commit(out); original_cfg=deepcopy(cfg)
    if any((out/name).exists() for name in ('independent-runner-started.json','independent-attempt.local.json',
        'independent-runner-failure.json','independent-runner-result.json','independent-failure.local.json',
        'independent-replay-verification.json','independent-result-review.json')):
        raise ValueError('Independent attempt consumed')
    validate_panel(out,cfg); load_seeds(out,cfg)
    before={name:sha(out/name) for name in BASE_IDENTITIES}
    if before['private-seeds.local.json']!=cfg['seed_document_sha256']: raise ValueError('Seed document changed')
    if not memory_ready(cfg): raise RuntimeError('Insufficient preflight memory; independent replay not started')
    storage_ready(out,cfg)
    refreshed=ready(out); verify_commit(out); validate_panel(out,refreshed); load_seeds(out,refreshed)
    if refreshed!=original_cfg or any(sha(out/name)!=value for name,value in before.items()):
        raise ValueError('Frozen identity changed during preflight')
    started=False; marker_complete=False; result=None; identities={}
    try:
        if trusted_out(requested_out)!=out: raise ValueError('Experiment output changed')
        path=out/'independent-runner-started.json'
        with path.open('x',encoding='utf-8',newline='\n') as stream:
            started=True
            json.dump(dict(attempt=1,freeze_sha256=sha(out/'freeze.json'),retry_allowed=False),stream)
            stream.write('\n')
        marker_complete=True
        identities={name:sha(out/name) for name in RUN_IDENTITIES}
        result=supervise(WindowsAPI(),[sys.executable,'-B','-m','learning.task_teacher_merged_eval_v1.independent_method'],
                         ROOT,cfg['independent_job_supervision'])
        if not validate_job(result): raise RuntimeError('Independent process or cleanup failed')
        if any(sha(out/name)!=value for name,value in identities.items()): raise ValueError('Independent identity changed')
        verification=read(out/'independent-replay-verification.json')
        validate_verification(out,cfg,verification); validate_panel(out,cfg)
        if any(sha(out/name)!=value for name,value in identities.items()): raise ValueError('Independent identity changed')
        receipt=dict(passed=True,supervision=result,verification_sha256=sha(out/'independent-replay-verification.json'),
            attempt_sha256=sha(out/'independent-attempt.local.json'),review_sha256=sha(out/'independent-result-review.json'),
            freeze_sha256=sha(out/'freeze.json'),identities=identities,same_official_engine=True,
            independent_rules_implementation=False,retry_allowed=False)
        write_new(out/'independent-runner-result.json',receipt)
        validate_completed(out,cfg)
        return receipt
    except BaseException as exc:
        if started:
            try:
                write_new(out/'independent-runner-failure.json',dict(error_kind=type(exc).__name__,marker_created=True,
                    marker_complete=marker_complete,identities=identities,supervision=result,retry_allowed=False))
            except BaseException as receipt:
                raise RuntimeError('Independent failure receipt failed: '+type(exc).__name__+'/'+type(receipt).__name__) from None
        raise


def finalize(out=OUT):
    """Calculate alias-only statistics once the full panel and replay pass."""
    from .campaign import storage_ready, validate_panel
    from .evaluation import evaluate
    requested_out=out; out=trusted_out(requested_out); cfg=ready(out); verify_commit(out)
    if any((out/name).exists() for name in ('evaluation-started.json','evaluation-failure.json','evaluation-result.json')):
        raise ValueError('Evaluation attempt consumed')
    rows=validate_panel(out,cfg); validate_completed(out,cfg); seeds=load_seeds(out,cfg)
    names=RUN_IDENTITIES+('independent-runner-result.json','independent-replay-verification.json',
                          'independent-attempt.local.json','independent-result-review.json')
    before={name:sha(out/name) for name in names}
    storage_ready(out,cfg)
    if ready(out)!=cfg: raise ValueError('Evaluation configuration changed during preflight')
    verify_commit(out)
    if validate_panel(out,cfg)!=rows or load_seeds(out,cfg)!=seeds:
        raise ValueError('Evaluation inputs changed during preflight')
    validate_completed(out,cfg)
    if any(sha(out/name)!=value for name,value in before.items()):
        raise ValueError('Evaluation identities changed during preflight')
    if trusted_out(requested_out)!=out: raise ValueError('Experiment output changed')
    write_new(out/'evaluation-started.json',dict(attempt=1,freeze_sha256=sha(out/'freeze.json'),retry_allowed=False))
    try:
        result=evaluate(rows,seeds)
        if ready(out)!=cfg: raise ValueError('Evaluation configuration changed')
        verify_commit(out); validate_panel(out,cfg); validate_completed(out,cfg)
        if any(sha(out/name)!=value for name,value in before.items()): raise ValueError('Evaluation identities changed')
        output=dict(passed=True,**result,independent_verification_sha256=sha(out/'independent-replay-verification.json'),
                    seed_commitment=cfg['seed_commitment'],no_independent_rules_engine=True,retry_allowed=False)
        write_new(out/'evaluation-result.json',output)
        return output
    except BaseException as exc:
        write_new(out/'evaluation-failure.json',dict(error_kind=type(exc).__name__,retry_allowed=False))
        raise


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=('run','finalize'))
    args=parser.parse_args()
    try:
        outcome=globals()[args.command]()
        print(json.dumps(dict(passed=outcome.get('passed',True))))
    except BaseException as exc:
        print(json.dumps(dict(passed=False,error_kind=type(exc).__name__)))
        raise SystemExit(1) from None
