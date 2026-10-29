"""Run the selected same-engine replay in an isolated, bounded child."""
import os


def main():
    import argparse
    from pathlib import Path
    import sys
    from learning.panel_eval_v3.isolation import validate_runtime

    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--prereg', required=True)
    parser.add_argument('--scratch', required=True)
    args = parser.parse_args()
    validate_runtime(args.scratch, sys.pycache_prefix, Path(__file__).resolve().parents[2])
    from learning.panel_eval_v3.adapter import bind_context
    from learning.panel_eval_v3.epoch import EpochContext
    from learning.panel_eval_v3.experiment import Context, FORMAL, require_run_kind
    from learning.panel_eval_v3.runner import accept_persistent_job
    from learning.task_teacher_merged_eval_v1 import independent_method

    context = EpochContext(Context(args.out, args.prereg))
    cfg=context.refresh()
    require_run_kind(cfg,FORMAL)
    with bind_context(context, accept_job=accept_persistent_job):
        independent_method.run(context.OUT)
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
