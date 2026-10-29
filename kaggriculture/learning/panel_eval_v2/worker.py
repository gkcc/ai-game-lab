"""Run one frozen official worker under a stage-bound context.

Python stdout/stderr are discarded; native output handles are not claimed to
be independently controlled. The caller supplies no seed on the command line.
"""
import os


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
    from learning.panel_eval_v2.isolation import validate_runtime

    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--key', required=True)
    parser.add_argument('--prereg', required=True)
    parser.add_argument('--scratch', required=True)
    parser.add_argument('--mode', choices=('game', 'canonical'), required=True)
    args = parser.parse_args()
    validate_runtime(args.scratch, sys.pycache_prefix, Path(__file__).resolve().parents[2])

    from learning.panel_eval_v2.adapter import bind_context
    from learning.panel_eval_v2.epoch import EpochContext
    from learning.panel_eval_v2.experiment import Context
    from learning.task_teacher_merged_eval_v1 import games
    context = EpochContext(Context(args.out, args.prereg))
    context.refresh()
    with bind_context(context):
        if args.mode == 'game':
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
            from learning.panel_eval_v2.wrapper_verify import verify_wrapper
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
