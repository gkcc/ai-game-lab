"""Pre-import guard for the three fixed panel entry points.

Run this file directly with -B -X pycache_prefix=<fresh> -I.  The trusted
parent must bind this file's bytes before creating the process.  This static
check covers the frozen project package initializers, not arbitrary custom
loaders or concurrent same-user changes after the check.
"""
import hashlib
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import runpy
import stat
import sys


ROOT = Path(__file__).resolve().parents[2]
MODULES = frozenset(('learning.panel_eval_v2.runner', 'learning.panel_eval_v2.worker',
                     'learning.panel_eval_v2.independent_worker'))
NAMESPACES = (
    'learning', 'learning/panel_eval_v2', 'learning/panel_safety_v1', 'learning/panel_safety_v2',
    'learning/task_completion_bc_reentry_v1', 'learning/task_completion_bc_v1',
    'learning/task_completion_economy_v1', 'learning/task_completion_parity_v1',
    'learning/task_completion_parity_v2', 'learning/task_completion_precision_v1',
    'learning/task_completion_score_probe_v1', 'learning/task_completion_smoke_v1',
    'learning/task_completion_training_coverage_v1', 'learning/task_completion_worst_pair_v1',
    'learning/task_job_empty_exit_v1', 'learning/task_job_sampling_race_v1',
    'learning/task_probe_failure_reporting_v1', 'learning/task_teacher_choice_bc_v1',
    'learning/task_teacher_choice_eval_v2', 'learning/task_teacher_dev_deviation_v1',
    'learning/task_teacher_merge_bc_v1', 'learning/task_teacher_merge_real_v1',
    'learning/task_teacher_merge_v1', 'learning/task_teacher_merged_eval_v1',
    'learning/task_teacher_weighted_eval_v2', 'learning/task_training_job_supervision_v1',
)
INITIALIZERS = {
    'decision/__init__.py':'292c33c078490e76cf6576d8922f527acf23d59a01b8f523f7a1b595cbe15d0e',
    'learning/bc_crop/__init__.py':'62631c0302958d70277a9fb7f1886e53e3cad4d1bb516331699ed1da787dc0b7',
    'learning/task_action_impact_diagnostic_v1/__init__.py':'2049e94f1fe2891136fc09817309da1b3bc054bba42b23bd6f5cf06ac176921d',
    'learning/task_action_observation_v1/__init__.py':'f52902b08dde4a3380a4384d4c42c21128b1d8fb4fa46235ff24d14bf7f48ffe',
    'learning/task_anchor_once_v1/__init__.py':'4978d6f44ce420931cddf54f31bcff0d06ce9fd8a036c92b590a69cad07e6c94',
    'learning/task_anchor_reentry_v1/__init__.py':'a79592f8af8ad152512cd44ef07da06dca8ceae3eaa5baf07a78a55adc8e6f33',
    'learning/task_anchor_spawn_repair_v1/__init__.py':'ac363e2aa864a0ec85647bf7f60e016484ea546c171254782188fe052d7d6718',
    'learning/task_bc_error_diagnostics_v1/__init__.py':'7fe73d92a6f435034e2de4fc239e202b38b383508f2c426bd389dd7565f4a381',
    'learning/task_bc_holdout_v1/__init__.py':'928cb0cf10b2bed3c1ceb88bb0052483963936d923824d5b2815abb5cce472ea',
    'learning/task_bc_v1/__init__.py':'53acf46748abd7c1696186b447505c1a42486621027fc52149dbc153e1e10b0f',
    'learning/task_choice_v1/__init__.py':'7645246b2ff833ffabb1388ffeaffda87008a658d4085b2dc970cf67fcb5b5c6',
    'learning/task_completion_evidence_v1/__init__.py':'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
    'learning/task_dagger_once_v1/__init__.py':'796151b10daeb85b0de82b2aeb1fff5630488138a91ae6907b05769abf499c5e',
    'learning/task_entry_repair_v1/__init__.py':'3d5bde4b71974a465bbba5a0d79a4fb7974ab234974573dffc52b52cc9e27a4b',
    'learning/task_launcher_contract_repair_v1/__init__.py':'632b0299dbdc537cac994926556852d38a6d6655c36b82998f7c4ca834b6266f',
    'learning/task_request_gate_v1/__init__.py':'60a8af5fc6b0ed1c0aff962b20a1efd63b8a9da70b54bbcd54e0302c5326e03c',
    'learning/task_reward_gate_v1/__init__.py':'ab997b1e38b77e48e3e5999257fdfc60d53b0da97aa404cc36f1b8da08f61e32',
}


def _info(path):
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
        raise ValueError('Initializer path contains a link')
    return info


def verify_initializers(root=ROOT):
    """Require the recorded namespace and initializer layout before import."""
    root = Path(root)
    suffixes = ('.pyc', *importlib.machinery.EXTENSION_SUFFIXES)
    directories = set(NAMESPACES) | {str(Path(name).parent).replace('\\','/') for name in INITIALIZERS}
    for name in directories:
        folder = root/name
        info = _info(folder)
        if info is None or not stat.S_ISDIR(info.st_mode):
            raise ValueError('Frozen package directory missing')
        for suffix in suffixes:
            if _info(folder/('__init__'+suffix)) is not None:
                raise ValueError('Unexpected sourceless initializer')
    for name in NAMESPACES:
        if _info(root/name/'__init__.py') is not None:
            raise ValueError('Unexpected namespace initializer')
    for name, digest in INITIALIZERS.items():
        path = root/name
        info = _info(path)
        if info is None or not stat.S_ISREG(info.st_mode) or info.st_size > 1024*1024:
            raise ValueError('Frozen initializer missing')
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Frozen initializer changed')


def main():
    argv = getattr(sys, 'orig_argv', ())
    if (not isinstance(argv, list) or len(argv) < 6 or argv[1:3] != ['-B','-X']
        or not argv[3].startswith('pycache_prefix=') or argv[4] != '-I'
        or sys.flags.dont_write_bytecode != 1 or sys.flags.isolated != 1
        or sys.flags.optimize != 0 or not isinstance(sys.pycache_prefix, str)
        or len(sys.argv) < 2 or sys.argv[1] not in MODULES):
        raise ValueError('Untrusted Python entry')
    verify_initializers()
    module = sys.argv[1]
    sys.argv = sys.argv[1:]
    # A namespace in the first sys.path entry otherwise loses to a regular
    # package with the same name in a later entry.  Bind it to this root only.
    spec = importlib.machinery.PathFinder.find_spec('learning', [str(ROOT)])
    if ('learning' in sys.modules or spec is None or spec.loader is not None
        or spec.submodule_search_locations is None
        or [Path(p).resolve() for p in spec.submodule_search_locations] != [(ROOT/'learning').resolve()]):
        raise ValueError('Unexpected learning namespace')
    # NamespacePath can recompute when sys.path changes, so pin a plain list.
    spec.submodule_search_locations = [str(ROOT/'learning')]
    package = importlib.util.module_from_spec(spec)
    package.__path__ = [str(ROOT/'learning')]
    sys.modules['learning'] = package
    sys.path.insert(0, str(ROOT))
    runpy.run_module(module, run_name='__main__', alter_sys=True)


if __name__ == '__main__':
    try:
        main()
    except SystemExit as exc:
        raise SystemExit(0 if type(exc.code) is int and exc.code == 0 else 1) from None
    except BaseException:
        raise SystemExit(1) from None
