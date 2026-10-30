"""Run exactly one official game in a caller-owned private experiment folder."""
import argparse
from contextlib import nullcontext, redirect_stderr, redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
import types

TRUSTED_IMPORTS_SHA = '329d3b6415e3fac1cabea5d27571db3510281cbabd9c5118c55cec9aa43b115a'


class QuietParser(argparse.ArgumentParser):
    def error(self, _message):
        raise ValueError('Invalid command arguments')


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_new(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    if not __debug__:
        raise RuntimeError('Optimized Python is not allowed')
    parser = QuietParser(description=__doc__)
    parser.add_argument('--stage', required=True, type=Path)
    parser.add_argument('--game-root', required=True, type=Path)
    parser.add_argument('--mode', required=True, choices=('control', 'intervene'))
    args = parser.parse_args()
    stage, root = args.stage.resolve(strict=True), args.game_root.resolve(strict=True)
    if not stage.is_relative_to((root/'artifacts').resolve(strict=True)):
        raise ValueError('Experiment must be inside the game artifacts directory')
    config = json.loads((stage/'input-private.json').read_text(encoding='utf-8'))
    if (config.get('schema') != 'single-task-input-v1' or type(config.get('seed')) is not int
        or config.get('side') != 0 or _sha(Path(__file__)) != config['source_sha256']['worker.py']
        or _sha(stage/'agents'/(args.mode+'.py')) != config['wrapper_sha256'][args.mode]
        or _sha(stage/'opponents/market.py') != config['opponent_sha256']):
        raise ValueError('Frozen worker inputs differ')
    trusted_raw = (stage/'sources/trusted_imports.py').read_bytes()
    if (hashlib.sha256(trusted_raw).hexdigest() != TRUSTED_IMPORTS_SHA
        or config['source_sha256'].get('trusted_imports.py') != TRUSTED_IMPORTS_SHA):
        raise ValueError('Experiment import guard changed')
    trusted = types.ModuleType('frozen_intervention_imports')
    exec(compile(trusted_raw, 'frozen-intervention-imports', 'exec', dont_inherit=True), trusted.__dict__)
    trusted.bind(root)
    import run_match
    if Path(run_match.__file__).resolve() != (root/'run_match.py').resolve():
        raise ValueError('Unexpected official entry')
    old_argv, old_cwd, native_output = sys.argv, Path.cwd(), run_match.native_output
    try:
        # The launcher supplies NUL native handles; Python output stays in local logs.
        run_match.native_output = lambda _log: nullcontext()
        sys.argv = [str(root/'run_match.py'), '--bot', str(stage/'agents'/(args.mode+'.py')),
                    '--opponent', str(stage/'opponents/market.py'), '--seed', str(config['seed']),
                    '--side', '0', '--name', args.mode, '--output-root', str(stage/'games')]
        os.chdir(root.parent)
        with (stage/('worker-'+args.mode+'.local.log')).open('x', encoding='utf-8') as log:
            with redirect_stdout(log), redirect_stderr(log):
                code = run_match.main()
        if code != 0:
            raise RuntimeError('Official game failed')
    finally:
        sys.argv, run_match.native_output = old_argv, native_output
        os.chdir(old_cwd)
    result = json.loads((stage/'games'/args.mode/'result.json').read_text(encoding='utf-8'))
    complete = json.loads((stage/'events'/(args.mode+'.complete.json')).read_text(encoding='utf-8'))
    if (result.get('passed') is not True or result.get('done') is not True
        or result.get('statuses') != ['DONE', 'DONE'] or result.get('recorded_states') != 720
        or result.get('action_turns') != 719 or result.get('agent_errors')
        or result.get('seed') != config['seed'] or result.get('bot_side') != 0
        or result.get('engine_sha256') != config['engine_sha256']
        or result.get('engine_matches_official_commit') is not True
        or result.get('upstream_commit') != config['engine_commit']
        or complete != {'passed': True, 'action_turns': 719, 'target_verified': True}):
        raise ValueError('Official game or gate did not complete')
    _write_new(stage/('worker-'+args.mode+'.done.json'),
               dict(passed=True, mode=args.mode, states=720, action_turns=719, agent_errors=0))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        # This private traceback may contain machine paths; never publish it.
        try:
            args = sys.argv
            if '--stage' in args and '--mode' in args:
                path = Path(args[args.index('--stage')+1]) / ('worker-'+args[args.index('--mode')+1]+'.failure.local.log')
                with path.open('x', encoding='utf-8') as stream:
                    stream.write(traceback.format_exc())
        except BaseException:
            pass
        raise SystemExit(1) from None
