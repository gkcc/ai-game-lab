"""Run one frozen official worker under a stage-bound context.

Python stdout/stderr are discarded; native output handles are not claimed to
be independently controlled. The caller supplies no seed on the command line.
"""
import os


def claim_smoke_game(context, cfg, key):
    """Consume the sole game only after its canonical and launch receipts."""
    from learning.panel_eval_v3.experiment import SMOKE, SMOKE_KEY, _same
    from learning.panel_safety_v1.seed_inventory import _json, _read_regular
    from learning.task_teacher_merged_eval_v1 import games
    if key != SMOKE_KEY:
        raise ValueError('Unexpected development smoke game')
    out=context.OUT
    def read(path): return _json(_read_regular(path,out))
    start=read(out/'results'/'coordinator-started.json')
    item=read(out/'job-inputs'/(key+'.json'))
    if not isinstance(item,dict):
        raise ValueError('Development smoke game input missing')
    games.validate_plan(item,cfg,job=True,out=out)
    digest=item.get('bot_sha256')
    canonical=read(out/'results'/'canonical'/(key+'.json'))
    launched=read(out/'results'/'job-start'/(key+'.json'))
    if (not _same(start,dict(attempt=1,prereg_commit=context.prereg_commit,run_kind=SMOKE,retry_allowed=False))
        or not _same(canonical,dict(game=key,wrapper_sha256=digest,passed=True,launcher_reaped=True,
                                    job_members_zero=True,isolated_cache_prefix_absent=True,retry_allowed=False))
        or not _same(launched,dict(game=key,wrapper_sha256=digest,retry_allowed=False))
        or games.core.sha(out/'agents'/(key+'.py')) != digest):
        raise ValueError('Development smoke canonical launch evidence changed')
    context.write_new(out/'results'/'smoke-worker-claim.json',
        dict(run_kind=SMOKE,game=SMOKE_KEY,wrapper_sha256=digest,attempt=1,retry_allowed=False))


def guarded_official_entry(context, entry, available):
    """Fail before the official runner starts if the 4 GiB floor is lost.

    The parent owns all bounded waiting; this child never waits or retries.
    The check is a sample, not a guarantee that memory stays available.
    """
    from learning.panel_safety_v1.memory_gate import MIN_AVAILABLE_BYTES
    context.deadline_check()
    current = available()
    if type(current) is not int or current < MIN_AVAILABLE_BYTES:
        raise RuntimeError('Official entry memory check failed')
    return entry()


def main():
    import argparse
    from pathlib import Path
    import sys
    from learning.panel_eval_v3.isolation import validate_runtime

    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--key', required=True)
    parser.add_argument('--prereg', required=True)
    parser.add_argument('--scratch', required=True)
    parser.add_argument('--mode', choices=('game', 'canonical'), required=True)
    args = parser.parse_args()
    validate_runtime(args.scratch, sys.pycache_prefix, Path(__file__).resolve().parents[2])

    from learning.panel_eval_v3.adapter import bind_context
    from learning.panel_eval_v3.epoch import EpochContext
    from learning.panel_eval_v3.experiment import Context, FORMAL, SMOKE, SMOKE_KEY, require_run_kind
    from learning.task_teacher_merged_eval_v1 import games
    context = EpochContext(Context(args.out, args.prereg))
    cfg=context.refresh()
    kind=cfg.get('run_kind')
    require_run_kind(cfg,kind)
    if kind == SMOKE and args.key != SMOKE_KEY:
        raise ValueError('Unexpected development smoke game')
    with bind_context(context):
        if args.mode == 'game':
            if kind == SMOKE:
                claim_smoke_game(context,cfg,args.key)
            import psutil
            original_entry = games.core.official_entry
            try:
                games.core.official_entry = lambda: guarded_official_entry(
                    context, original_entry, lambda: psutil.virtual_memory().available)
                games.job(context.OUT, args.key)
            finally:
                games.core.official_entry = original_entry
        else:
            from learning.panel_safety_v1.seed_inventory import _json, _read_regular
            from learning.panel_eval_v3.wrapper_verify import verify_wrapper
            if Path(args.key).name != args.key:
                raise ValueError('Invalid game key')
            cfg = context.ready()
            item = _json(_read_regular(context.OUT/'job-inputs'/(args.key+'.json'), context.OUT))
            if not isinstance(item, dict) or item.get('game') != args.key:
                raise ValueError('Game input identity changed')
            verify_wrapper(context.OUT, item, cfg, args.scratch)
    validate_runtime(args.scratch, sys.pycache_prefix, Path(__file__).resolve().parents[2])


if __name__ == '__main__':
    try:
        with open(os.devnull, 'w', encoding='utf-8') as sink:
            from contextlib import redirect_stdout, redirect_stderr
            with redirect_stdout(sink), redirect_stderr(sink):
                main()
    except BaseException:
        raise SystemExit(1) from None
    raise SystemExit(0)
