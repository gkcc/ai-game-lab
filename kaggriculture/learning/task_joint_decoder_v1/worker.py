"""Thin adaptation of the authenticated official Worker for joint decoding.

Only a dedicated, registered child may call execute(). The outer coordinator
owns source/import authentication, readiness, Job supervision and acceptance.
The official job, audit, single main replay, storage budget and metrics consumer
are reused unchanged. This file is neither a standalone runner nor a new Server.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path

from learning.task_pair_dev_campaign_v1.child import MAX_WRAPPER_BYTES
from learning.task_pair_dev_games_v1 import capture
from learning.task_pair_dev_metrics_v1.consumer import GameConsumer
from learning.task_teacher_fresh_panel_v1.runner import hidden_official_entry
from learning.task_teacher_fresh_panel_v1 import contract as argv_contract
from learning.task_teacher_counterfactual_pair_v1 import audit as replay_audit
from .contract import JointPanel
from .policy import JointAgent


GAME = Path(__file__).resolve().parents[3].parent.parent / 'kaggriculture'


def _require(value):
    if not value:
        raise ValueError('Registered joint-decoder worker context differs')


def _wrapper_code(root, stage, plan, cfg, task, digest):
    """Private, seed-independent filename entry; bootstrap owns import paths."""
    trace = stage / 'traces' / (plan['game'] + '.jsonl.gz')
    code = ('from pathlib import Path\nimport sys\n'
        '_p=globals().get("__file__",sys._getframe().f_code.co_filename)\n'
        'ROOT=next(p for p in Path(_p).resolve().parents if '
        '(p/"learning/task_dagger_once_v1/policy.py").is_file())\n'
        'from learning.task_joint_decoder_v1.policy import JointAgent\n'
        '_runner=JointAgent(ROOT/'+repr(cfg['crop_checkpoint'])+','+repr(cfg['crop_checkpoint_sha256'])+
        ',ROOT/'+repr(task.relative_to(root).as_posix())+','+repr(digest)+
        ',trace=ROOT/'+repr(trace.relative_to(root).as_posix())+',limits='+repr(cfg['game_limits'])+')\n'
        'def agent(observation,configuration):\n    return _runner(observation,configuration)\n'
        'submission_agent=agent\n')
    _require(0 < len(code.encode('utf-8')) <= MAX_WRAPPER_BYTES)
    return code


@contextmanager
def joint_hooks(module, panel, root, stage, check):
    """Replace actor/wrapper only on the caller's isolated, SHA-checked module."""
    old_actor, old_wrapper = module.actor, module.wrapper
    _require(callable(old_actor) and callable(old_wrapper))

    def configuration(out, plan, cfg):
        _require(Path(out).resolve(strict=False) == stage and cfg == panel.worker_config()
                 and check() is True)
        # The enclosing original _bound_worker checks the exact expected plan,
        # including the baseline task identity even for an enriched audit row.
        task, digest = module.task_identity(out, plan, cfg)
        relative, expected = panel.task_model(plan)
        _require(task == root / relative and digest == expected)
        return task, digest

    def actor(out, plan, cfg):
        task, digest = configuration(out, plan, cfg)
        return JointAgent(root / cfg['crop_checkpoint'], cfg['crop_checkpoint_sha256'],
                          task, digest, trace=None, limits=cfg['game_limits'])

    def wrapper(out, plan, cfg):
        task, digest = configuration(out, plan, cfg)
        key = plan['game'];path = stage / 'agents' / (key + '.py')
        _require(Path(key).name == key and not path.exists()
                 and not (stage / 'traces' / (key + '.jsonl.gz')).exists())
        code = _wrapper_code(root, stage, plan, cfg, task, digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8', newline='\n') as stream:
            _require(stream.write(code) == len(code))
        return path

    try:
        module.actor, module.wrapper = actor, wrapper
        yield module
    finally:
        module.actor, module.wrapper = old_actor, old_wrapper


@contextmanager
def ordered_main_replay(module, check):
    """Enhance the same single main replay with the proven shed-order helper."""
    previous = module.reexecute
    def replay(value, *, trace=False, label_sink=None):
        _require(trace is True and callable(label_sink))
        result = replay_audit.reexecute(value, guard=check, sink=label_sink)
        _require(result['shed_order_frames_checked'] == 720)
        return dict(result['official'], shed_order_frames_checked=720)
    try:
        module.reexecute = replay
        yield
    finally:
        module.reexecute = previous


def execute(modules, manifest, stage, plan, check):
    """One live complete game and the old audit's one main official replay.

    A return is not parent acceptance: acceptance.accept must still validate
    the linked receipts, Job cleanup and verify_files() independently.
    """
    dm, original = modules['driver'], modules['worker']
    original._single_python_thread()
    original._registered(check)
    with original._quiet_output():
        root, stage = original._root_and_stage(GAME, stage)
        panel = JointPanel(manifest.panel)
        plan = panel.validate_plan(plan)
        cfg = panel.worker_config()
        source = dm._read(root, root / original.WORKER_RELATIVE, original.MAX_WORKER_BYTES)
        _require(hashlib.sha256(source).hexdigest() == panel.worker_sha)
        with original._bound_worker(source, panel, root, stage, plan, check) as module:
            with joint_hooks(module, panel, root, stage, check):
                bot = module.wrapper(stage, plan, cfg)
            _require(bot == stage / 'agents' / (plan['game'] + '.py'))
            rival = dm._hash(stage, stage / 'sources' / (plan['opponent'] + '.py'), dm.MAX_SOURCE)
            _require(rival == manifest.opponent(plan['opponent'])[1])
            job = panel.validate_plan(dict(plan,
                bot_sha256=dm._hash(stage, bot, MAX_WRAPPER_BYTES), rival_sha256=rival,
                task_checkpoint_sha256=panel.task_model(plan)[1]), job=True)
            dm._write_new(stage, stage / 'job-inputs' / (plan['game'] + '.json'), job,
                          limit=original.MAX_JOB_BYTES)
        # The immutable expected object in _bound_worker now contains the whole
        # job, including the just-written wrapper hash. No mutable identity hook.
        with original._bound_worker(source, panel, root, stage, job, check) as module:
            with joint_hooks(module, panel, root, stage, check), ordered_main_replay(module, check):
                with capture.collect_one(module, stage=stage, panel=panel, expected=job,
                        check=check, consumer_factory=GameConsumer) as results:
                    with hidden_official_entry(module, argv_contract):
                        module.job(stage, plan['game'])
                _require(len(results) == 1 and check() is True)
                capture.store_metrics(module, stage, results[0])
                _require(check() is True)


def verify_files(modules, manifest, stage, plan, row):
    """Acceptance callback binding private files to this exact decoded job."""
    dm, original, acceptance = modules['driver'], modules['worker'], modules['acceptance']
    root, stage = original._root_and_stage(GAME, stage)
    panel = JointPanel(manifest.panel)
    plan = panel.validate_plan(plan)
    job = panel.validate_plan(acceptance._json(dm._read(stage,
        stage / 'job-inputs' / (plan['game'] + '.json'), original.MAX_JOB_BYTES),
        original.MAX_JOB_BYTES), job=True)
    _require(all(type(row.get(k)) is type(v) and row[k] == v for k, v in job.items())
             and all(job[k] == v for k, v in plan.items()))
    task, task_sha = panel.task_model(plan)
    rival, rival_sha = manifest.opponent(plan['opponent'])
    checks = ((root, root / task, dm.MAX_MODEL, task_sha),
        (root, root / panel.crop[0], dm.MAX_MODEL, panel.crop[1]),
        (root, root / rival, dm.MAX_SOURCE, rival_sha),
        (stage, stage / 'sources' / (plan['opponent'] + '.py'), dm.MAX_SOURCE, rival_sha),
        (root, root / original.WORKER_RELATIVE, original.MAX_WORKER_BYTES, panel.worker_sha),
        (stage, stage / 'agents' / (plan['game'] + '.py'), MAX_WRAPPER_BYTES, job['bot_sha256']),
        (stage, stage / 'traces' / (plan['game'] + '.jsonl.gz'), capture.MAX_COMPRESSED_BYTES, row['trace_sha256']),
        (stage, stage / 'games' / plan['game'] / 'replay.json.gz', capture.MAX_COMPRESSED_BYTES, row['replay_gzip_sha256']))
    _require(job['rival_sha256'] == rival_sha
             and all(dm._hash(boundary, path, maximum) == digest for boundary, path, maximum, digest in checks))
    return True
