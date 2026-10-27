"""Run exactly one preselected replay in a separate owned Windows Job."""
import json
from pathlib import Path
import sys
from learning.task_training_job_supervision_v1.job import WindowsAPI
from learning.task_job_sampling_race_v1.sampling import supervise
from .resources import memory_ready
from .experiment import OUT,ROOT,ready,verify_commit,sha,read,write_new


def validate_verification(out,cfg,verification):
    expected_game=f"holdout-{cfg['test_seeds'][0]}-round2-0-weighted"
    bindings=dict(freeze_sha256=sha(out/'freeze.json'),registration_sha256=sha(out/'registration.json'),
        games_result_sha256=sha(out/'games-result.json'),game_ledger_sha256=sha(out/'game-ledger.json'),
        audit_ledger_sha256=sha(out/'game-audit-ledger.json'),method_sha256=sha(Path(__file__).with_name('independent_method.py')),
        checkpoint_sha256=cfg['weighted_checkpoint_sha256'],game=expected_game)
    reads=verification.get('raw_reads',{})
    if (verification.get('passed') is not True or any(verification.get(k)!=v for k,v in bindings.items())
        or verification.get('official_transitions')!=719 or verification.get('sample_count')!=20 or verification.get('matched')!=20
        or reads.get('files_opened')!=2 or reads.get('files_completed')!=2 or reads.get('raw_decode_passes_per_file')!=1):
        raise ValueError('Independent verification does not bind current panel')


def validate_job(result):
    return (isinstance(result,dict) and result.get('passed') is True and result.get('assigned') is True
        and result.get('resumed') is True and result.get('launcher_reaped') is True
        and result.get('job_members_zero') is True and result.get('error_kind') is None
        and result.get('cleanup_error_kind') is None and result.get('stop_reason') is None and result.get('exit_code') == 0)


def run(out=OUT):
    out=Path(out).resolve(); cfg=ready(out); verify_commit(out)
    if any((out/name).exists() for name in ('independent-runner-started.json','independent-attempt.local.json',
        'independent-runner-failure.json','independent-runner-result.json','independent-failure.local.json',
        'independent-replay-verification.json','independent-result-review.json')):
        raise ValueError('Independent attempt consumed')
    from .campaign import validate_panel
    validate_panel(out,cfg)
    started=False; marker_complete=False; result=None; error=None; identities={}
    try:
        path=(out/'independent-runner-started.json').resolve()
        if not path.is_relative_to(OUT.resolve()): raise ValueError('Marker outside experiment')
        with path.open('x',encoding='utf-8',newline='\n') as stream:
            started=True
            json.dump(dict(attempt=1,freeze_sha256=sha(out/'freeze.json'),retry_allowed=False),stream)
            stream.write('\n')
        marker_complete=True
        identities={name:sha(out/name) for name in ('freeze.json','registration.json','pre-execution-commit.json',
                    'games-result.json','game-ledger.json','game-audit-ledger.json','independent-runner-started.json')}
        if not memory_ready(cfg): raise RuntimeError('Insufficient preflight memory; independent replay not started')
        result=supervise(WindowsAPI(),[sys.executable,'-B','-m','learning.task_teacher_weighted_eval_v1.independent_method'],
                         ROOT,cfg['independent_job_supervision'])
        if not validate_job(result):
            raise RuntimeError('Independent process or cleanup failed')
        if any(sha(out/name)!=digest for name,digest in identities.items()): raise ValueError('Independent identity changed')
        verification=read(out/'independent-replay-verification.json')
        validate_verification(out,cfg,verification)
        if any(sha(out/name)!=digest for name,digest in identities.items()): raise ValueError('Independent identity changed')
        receipt=dict(passed=True,supervision=result,verification_sha256=sha(out/'independent-replay-verification.json'),
            freeze_sha256=sha(out/'freeze.json'),identities=identities,
            same_official_engine=True,independent_rules_implementation=False,retry_allowed=False)
        write_new(out/'independent-runner-result.json',receipt)
        return receipt
    except BaseException as exc:
        error=type(exc).__name__
        if started:
            try: write_new(out/'independent-runner-failure.json',dict(error_kind=error,marker_created=True,
                marker_complete=marker_complete,identities=identities,supervision=result,retry_allowed=False))
            except BaseException as receipt: raise RuntimeError('Independent failure receipt failed: '+error+'/'+type(receipt).__name__) from None
        raise


def finalize(out=OUT):
    from .evaluation import evaluate
    from .campaign import validate_panel
    out=Path(out).resolve(); cfg=ready(out); verify_commit(out)
    if any((out/name).exists() for name in ('evaluation-started.json','evaluation-failure.json','evaluation-result.json')):
        raise ValueError('Evaluation attempt consumed')
    panel=read(out/'games-result.json'); independent=read(out/'independent-runner-result.json')
    verification=read(out/'independent-replay-verification.json')
    if (not panel['passed'] or not independent['passed'] or not validate_job(independent.get('supervision'))
        or independent['freeze_sha256']!=sha(out/'freeze.json')
        or independent['verification_sha256']!=sha(out/'independent-replay-verification.json')
        or panel['game_ledger_sha256']!=sha(out/'game-ledger.json')
        or panel['audit_ledger_sha256']!=sha(out/'game-audit-ledger.json')):
        raise ValueError('Panel and independent replay required')
    identities=independent.get('identities')
    required={'freeze.json','registration.json','pre-execution-commit.json','games-result.json',
              'game-ledger.json','game-audit-ledger.json','independent-runner-started.json'}
    if not isinstance(identities,dict) or set(identities)!=required or any(sha(out/name)!=digest for name,digest in identities.items()):
        raise ValueError('Independent runner identity changed')
    validate_verification(out,cfg,verification)
    validate_panel(out,cfg)
    write_new(out/'evaluation-started.json',dict(attempt=1,freeze_sha256=sha(out/'freeze.json'),retry_allowed=False))
    try:
        result=evaluate(read(out/'game-ledger.json'),cfg['test_seeds'])
        write_new(out/'evaluation-result.json',dict(passed=True,**result,independent_verification_sha256=sha(out/'independent-replay-verification.json'),
            no_independent_rules_engine=True,retry_allowed=False))
        return result
    except BaseException as exc:
        write_new(out/'evaluation-failure.json',dict(error_kind=type(exc).__name__,retry_allowed=False))
        raise
