"""Thin full-game worker for frozen-model HARVEST candidate development.

The parent owns source/model authentication, one-attempt permission and Windows
Job supervision. This module reuses the frozen filename-game worker, its Actor
reconstruction, one official replay, ordered shed checks and bounded outputs.
No checkpoint, private source or official environment is opened on import.
"""
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib
import math
from pathlib import Path
from types import ModuleType

from learning.task_pair_dev_campaign_v1.child import MAX_WRAPPER_BYTES
from learning.task_pair_dev_games_v1 import capture
from learning.task_pair_dev_metrics_v1.consumer import GameConsumer, _bytes


BASE_RELATIVE = 'learning/task_joint_decoder_v1/worker.py'
BASE_SHA256 = 'fc0d06d6b8c2d6f77b374538d67ee0c8ccdde2b88b5adf554973cb34d30c882b'
GAME = Path(__file__).resolve().parents[3].parent.parent/'kaggriculture'
PHASE_CAPS = {
    'live':dict(environment_constructions=1, resets=2, official_steps=719,
                actor_calls=719, rival_calls=719),
    'reconstruction':dict(actor_calls=719, rival_calls=719),
    'replay':dict(environment_constructions=1, resets=1, official_steps=719),
}


def require(value, message='Harvest model worker contract differs'):
    if not value:
        raise ValueError(message)


class Operations:
    """Count actual entries/returns, with phases and pre-call hard caps."""
    def __init__(self, counts, check):
        require(type(counts) is dict and not counts and callable(check))
        self.counts, self.check, self.phase = counts, check, 'live'
        counts.update({phase:{name+'_'+suffix:0 for name in caps for suffix in ('attempted','returned')}
                       for phase,caps in PHASE_CAPS.items()})

    def call(self, name, function):
        require(self.check() is True)
        caps, current = PHASE_CAPS[self.phase], self.counts[self.phase]
        require(name in caps and current[name+'_attempted'] < caps[name], 'Operation budget exhausted')
        current[name+'_attempted'] += 1
        result = function()
        current[name+'_returned'] += 1
        return result

    @contextmanager
    def at(self, phase):
        require(phase in PHASE_CAPS)
        previous, self.phase = self.phase, phase
        try:
            yield
        finally:
            self.phase = previous

    def complete(self):
        for phase,caps in PHASE_CAPS.items():
            require(all(self.counts[phase][name+'_attempted'] == self.counts[phase][name+'_returned'] == cap
                        for name,cap in caps.items()), 'Operation counts did not close')


class HarvestConsumer(GameConsumer):
    """Join same-forward candidate scores to actual official unit effects."""
    def __init__(self, player):
        super().__init__(player)
        self.choice_counts = Counter(callbacks=0, callbacks_with_added_candidates=0,
            added_candidates=0, selected_virtual_harvest=0)
        self.virtual = {}
        self.virtual_effects = {}
        self.live_model_counts = None

    def trace_turn(self, trace):
        super().trace_turn(trace)
        try:
            for ordinal,record in enumerate(trace['records']):
                candidates = record['inputs']['candidates']
                ids = [c['id'] for c in candidates]
                model, extension = record.get('task_model'), record.get('harvest_choice_extension')
                require(type(model) is dict and set(model) == {
                    'checkpoint_sha256','logits','argmax_index','chosen','fallbacks'})
                logits, index = model['logits'], model['argmax_index']
                require(type(logits) is list and len(logits) == len(ids)
                    and all(type(v) is float and math.isfinite(v) for v in logits)
                    and type(index) is int and index == max(range(len(logits)),key=logits.__getitem__)
                    and ids[index] == model['chosen'] == record['chosen']
                    and type(model['fallbacks']) is int and model['fallbacks'] == 0
                    and model['checkpoint_sha256'] == trace['task_checkpoint_sha256'],
                    'Recorded same-forward argmax differs')
                require(type(extension) is dict and set(extension) == {
                    'added_candidate_ids','selected_virtual_harvest','source_water_id'})
                added = extension['added_candidate_ids']
                require(type(added) is list and len(set(added)) == len(added)
                    and len(added) < len(ids) and (not added or ids[-len(added):] == added))
                for identifier in added:
                    candidate = candidates[ids.index(identifier)]
                    require(candidate['op'] == 'HARVEST'
                        and any(c['id'] not in added and c['op'] == 'WATER'
                            and c['unit'] == candidate['unit'] and c['target'] == candidate['target']
                            for c in candidates), 'Virtual candidate lacks its original WATER choice')
                selected = record['chosen'] in added
                require(type(extension['selected_virtual_harvest']) is bool
                    and extension['selected_virtual_harvest'] == selected)
                self.choice_counts.update(callbacks=1,callbacks_with_added_candidates=int(bool(added)),
                                          added_candidates=len(added),selected_virtual_harvest=int(selected))
                if selected:
                    water = [c for c in candidates if c['id'] == extension['source_water_id']]
                    require(len(water) == 1 and water[0]['op'] == 'WATER'
                        and water[0]['unit'] == record['unit'] and water[0]['target'] == record['target']
                        and record['unit_request'] == ['HARVEST'])
                    key = (trace['step'],self.player,record['unit'])
                    require(key not in self.virtual)
                    self.virtual[key] = dict(step=trace['step'],ordinal=ordinal,unit=record['unit'],
                        target=deepcopy(record['target']),chosen=record['chosen'],
                        selected_logit=logits[index],source_water_logit=logits[ids.index(water[0]['id'])])
                else:
                    require(extension['source_water_id'] is None)
            self.live_model_counts = deepcopy(trace['model_counts'])
        except BaseException:
            self.failed = True
            raise

    def sink(self, event):
        super().sink(event)
        if self.failed or event['phase'] != 'unit':
            return
        key = (event['step'],event['player'],event['unit'])
        if key not in self.virtual:
            return
        try:
            require(key not in self.virtual_effects)
            selected = self.virtual[key]
            require(_bytes(event['requested']) == _bytes(['HARVEST']))
            unit = event['unit']
            prior = dict(event['before']['inventories'][unit]).get('MELON',0)
            after = dict(event['after']['inventories'][unit]).get('MELON',0)
            require(type(prior) is int and type(after) is int and 0 <= prior <= after)
            same_target = event['position'] == selected['target']
            executed = (same_target and event['effective'] == ['HARVEST']
                        and event['status'] in ('executed','partial'))
            self.virtual_effects[key] = dict(selected,
                requested=deepcopy(event['requested']),effective=deepcopy(event['effective']),
                execution_status=event['status'],same_target_atomic_effect=executed,
                melon_inventory_before=prior,melon_inventory_after=after,
                harvested_melon_units=after-prior,
                tile_before=deepcopy(event['tile_before']),tile_after=deepcopy(event['tile_after']))
        except BaseException:
            self.failed = True
            raise

    def finish(self, initial_observations, final_observations, statuses, audit):
        result = super().finish(initial_observations, final_observations, statuses, audit)
        require(set(self.virtual_effects) == set(self.virtual), 'Selected HARVEST lacks official execution evidence')
        effects = [self.virtual_effects[k] for k in sorted(self.virtual_effects)]
        require(type(self.live_model_counts) is dict
            and self.live_model_counts.get('controller_calls') == 719
            and self.live_model_counts.get('task_forwards') == self.choice_counts['callbacks'])
        result['harvest_choice'] = dict(counts=dict(self.choice_counts),
            selected_virtual_actual_executions=sum(r['same_target_atomic_effect'] for r in effects),
            selected_virtual_harvested_melon_units=sum(r['harvested_melon_units'] for r in effects),
            selected_virtual_execution_status_counts=dict(Counter(r['execution_status'] for r in effects)),
            selected_virtual_effects=effects,live_model_counts=self.live_model_counts,
            interpretation='Same-forward scores are preferences, not cash values; requests and actual effects are separate.')
        return result


def _wrapper_code(root, stage, plan, cfg, task, digest):
    # Identical filename entry structure; only the new actor import changes.
    trace = stage/'traces'/(plan['game']+'.jsonl.gz')
    code = ('from pathlib import Path\nimport sys\n'
        '_p=globals().get("__file__",sys._getframe().f_code.co_filename)\n'
        'ROOT=next(p for p in Path(_p).resolve().parents if '
        '(p/"learning/task_dagger_once_v1/policy.py").is_file())\n'
        'from learning.task_harvest_model_dev_v1.policy import BoundedHarvestAgent\n'
        '_runner=BoundedHarvestAgent(ROOT/'+repr(cfg['crop_checkpoint'])+','+repr(cfg['crop_checkpoint_sha256'])+
        ',ROOT/'+repr(task.relative_to(root).as_posix())+','+repr(digest)+
        ',trace=ROOT/'+repr(trace.relative_to(root).as_posix())+',limits='+repr(cfg['game_limits'])+')\n'
        'def agent(observation,configuration):\n    return _runner(observation,configuration)\n'
        'submission_agent=agent\n')
    require(0 < len(code.encode('utf-8')) <= MAX_WRAPPER_BYTES)
    return code


def _base(panel):
    """Compile an immutable old adapter in a private namespace, not its module."""
    path = Path(__file__).resolve().parents[2]/BASE_RELATIVE
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == BASE_SHA256)
    module = ModuleType('learning.task_joint_decoder_v1._harvest_model_worker')
    module.__file__, module.__package__ = str(path), 'learning.task_joint_decoder_v1'
    exec(compile(raw,str(path),'exec'),module.__dict__)
    from .policy import BoundedHarvestAgent
    def panel_view(base):
        require(panel.base is base)
        return panel
    module.JointPanel, module.JointAgent = panel_view, BoundedHarvestAgent
    module._wrapper_code, module.GameConsumer = _wrapper_code, HarvestConsumer
    return module


@contextmanager
def _official_counts(core, official_agent, operations, stage, plan):
    cls, agent_cls = core.Environment, official_agent.Agent
    require(core.Agent is agent_cls)
    originals = cls.__init__, cls.reset, cls.step, agent_cls.act
    def construct(env,*args,**kwargs):
        return operations.call('environment_constructions',lambda:originals[0](env,*args,**kwargs))
    def reset(env,*args,**kwargs):
        return operations.call('resets',lambda:originals[1](env,*args,**kwargs))
    def step(env,*args,**kwargs):
        return operations.call('official_steps',lambda:originals[2](env,*args,**kwargs))
    def act(agent,obs):
        require(operations.phase == 'live' and type(obs['player']) is int and obs['player'] in (0,1))
        own = obs['player'] == plan['side']
        path = stage/('agents/'+plan['game']+'.py' if own else 'sources/'+plan['opponent']+'.py')
        require(type(agent.raw) is str and Path(agent.raw).resolve(strict=True) == path)
        return operations.call('actor_calls' if own else 'rival_calls',lambda:originals[3](agent,obs))
    try:
        cls.__init__, cls.reset, cls.step, agent_cls.act = construct, reset, step, act
        yield
    finally:
        cls.__init__, cls.reset, cls.step, agent_cls.act = originals


class _Actor:
    def __init__(self, actor, operations):
        self.actor, self.operations = actor, operations
    def __getattr__(self, name):
        return getattr(self.actor,name)
    def __call__(self,*args,**kwargs):
        return self.operations.call('actor_calls',lambda:self.actor(*args,**kwargs))


@contextmanager
def _instrument(base, operations, core):
    from learning.task_harvest_struct_compat_v1 import compat
    original_hooks, original_replay = base.joint_hooks, base.ordered_main_replay
    @contextmanager
    def hooks(module,panel,root,stage,check):
        with original_hooks(module,panel,root,stage,check):
            old_actor, old_load, old_audit, old_compare = module.actor,module.load,module.audit,module.same_request
            cls, identity = compat.authenticate(core,old_compare)
            comparator = compat.Comparator(cls,old_compare)
            def actor(*args,**kwargs):
                require(operations.phase == 'reconstruction')
                return _Actor(old_actor(*args,**kwargs),operations)
            def load(path):
                require(operations.phase == 'reconstruction')
                rival = old_load(path)
                return lambda *args,**kwargs:operations.call('rival_calls',lambda:rival(*args,**kwargs))
            def audit(*args,**kwargs):
                with operations.at('reconstruction'):
                    result = old_audit(*args,**kwargs)
                operations.complete()
                result.update(harvest_model_operation_counts=deepcopy(operations.counts),
                    request_container_binding=identity,request_container_comparisons=comparator.snapshot())
                return result
            try:
                module.actor,module.load,module.audit,module.same_request = actor,load,audit,comparator
                yield module
            finally:
                module.actor,module.load,module.audit,module.same_request = old_actor,old_load,old_audit,old_compare
    @contextmanager
    def replay(module,check):
        with original_replay(module,check):
            previous = module.reexecute
            def invoke(*args,**kwargs):
                with operations.at('replay'):
                    return previous(*args,**kwargs)
            module.reexecute = invoke
            try:
                yield
            finally:
                module.reexecute = previous
    try:
        base.joint_hooks,base.ordered_main_replay = hooks,replay
        yield
    finally:
        base.joint_hooks,base.ordered_main_replay = original_hooks,original_replay


def execute_job(*, modules, manifest, panel, stage, plan, check, counters):
    """One game + Actor reconstruction + one official replay; no auto-retry."""
    require(__debug__ and plan['parity_kind'] is None and plan['policy'] == 'harvest_model')
    plan = panel.validate_plan(plan)
    base = _base(panel)
    core = importlib.import_module('kaggle_environments.core')
    official_agent = importlib.import_module('kaggle_environments.agent')
    operations = Operations(counters,check)
    stage = Path(stage).resolve(strict=True)
    with _official_counts(core,official_agent,operations,stage,plan), _instrument(base,operations,core):
        base.execute(modules,manifest,stage,plan,check)
    operations.complete()
    require(check() is True)
    return dict(passed=True,counts=deepcopy(counters),official_games=1,official_replays=1)


def verify_files(*, modules, manifest, panel, stage, plan, row):
    """Same source/file bindings as the old adapter; no official import needed."""
    require(plan['parity_kind'] is None and plan['policy'] == 'harvest_model'
            and panel.base is manifest.panel)
    dm, original, acceptance = modules['driver'],modules['worker'],modules['acceptance']
    root, stage = original._root_and_stage(GAME,stage)
    plan = panel.validate_plan(plan)
    job = panel.validate_plan(acceptance._json(dm._read(stage,
        stage/'job-inputs'/(plan['game']+'.json'),original.MAX_JOB_BYTES),original.MAX_JOB_BYTES),job=True)
    require(all(type(row.get(k)) is type(v) and row[k] == v for k,v in job.items())
            and all(job[k] == v for k,v in plan.items()))
    task, task_sha = panel.task_model(plan)
    rival, rival_sha = manifest.opponent(plan['opponent'])
    checks = ((root,root/task,dm.MAX_MODEL,task_sha),
        (root,root/panel.crop[0],dm.MAX_MODEL,panel.crop[1]),
        (root,root/rival,dm.MAX_SOURCE,rival_sha),
        (stage,stage/'sources'/(plan['opponent']+'.py'),dm.MAX_SOURCE,rival_sha),
        (root,root/original.WORKER_RELATIVE,original.MAX_WORKER_BYTES,panel.worker_sha),
        (stage,stage/'agents'/(plan['game']+'.py'),MAX_WRAPPER_BYTES,job['bot_sha256']),
        (stage,stage/'traces'/(plan['game']+'.jsonl.gz'),capture.MAX_COMPRESSED_BYTES,row['trace_sha256']),
        (stage,stage/'games'/plan['game']/'replay.json.gz',capture.MAX_COMPRESSED_BYTES,row['replay_gzip_sha256']))
    require(job['rival_sha256'] == rival_sha
        and all(dm._hash(boundary,path,maximum) == digest for boundary,path,maximum,digest in checks))
    return True
