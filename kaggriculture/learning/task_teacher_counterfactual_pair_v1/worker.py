"""One private, externally supervised phase of an official paired experiment.

No seed, source identity or observation is printed. The caller authenticates
the private case and preceding phase receipts, owns the Windows Job, file-write
budget and deadline, and publishes only independently checked aggregates.
"""
from contextlib import nullcontext,redirect_stderr,redirect_stdout
import gzip
import hashlib
import importlib.util
import argparse
import json
import os
from pathlib import Path
import sys
import threading
import types

from decision.api import observe
from decision.audit_market_teacher import observation
from learning.task_bc_error_diagnostics_v1.shadow import first_difference,ordered_hash
from learning.task_student_anchor_collection_v1 import sources
from learning.task_student_continuation_v1 import samples
from learning.task_smoke_weight_probe_v1 import core as trace_budget
from learning.task_teacher_counterfactual_screen_v1 import experiment as screen_experiment,screen
from . import actor,audit,checks,gate


PHASES=('prepare_case','control','control_audit','control_verify',
        'intervene','intervene_audit','pair_verify')
GAME=sources.old.fresh.GAME.resolve()
MAX_REPLAY=128*1024**2
MAX_SMALL=8*1024**2
CASE_KEYS=frozenset(('role','slot','opponent','side','window','rank','step','ordinal',
    'input_sha256','original_choice','teacher_choice','seed','task_sha256','crop_sha256'))


def _require(condition):
    if not condition: raise ValueError('Private paired-game contract failed')


def _digest(raw): return hashlib.sha256(raw).hexdigest()


def _sha_file(path,guard,maximum=MAX_REPLAY):
    guard(); h=hashlib.sha256();size=0
    with path.open('rb') as stream:
        while True:
            block=stream.read(1024**2)
            if not block: break
            size+=len(block);_require(size<=maximum);h.update(block);guard()
    return h.hexdigest()


def _json(path,maximum=MAX_SMALL):
    with path.open('rb') as stream: raw=stream.read(maximum+1)
    _require(0<len(raw)<=maximum)
    def pairs(items):
        result={}
        for k,v in items:
            _require(k not in result);result[k]=v
        return result
    return json.loads(raw,object_pairs_hook=pairs,
        parse_constant=lambda _value: (_ for _ in ()).throw(ValueError('Invalid numeric value')))


def _write(path,raw):
    with path.open('xb') as stream: stream.write(raw)


def _json_bytes(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode('utf-8')


def _case(value):
    _require(type(value) is dict and set(value)==CASE_KEYS
        and value['role'] in screen.ROLES and value['slot']==screen.SLOTS[value['role']]
        and value['opponent'] in screen.OPPONENTS and type(value['side']) is int and value['side'] in (0,1)
        and type(value['window']) is int and 0<=value['window']<len(screen.WINDOWS)
        and type(value['rank']) is int and 0<=value['rank']<2
        and type(value['step']) is int and screen.WINDOWS[value['window']][0]<=value['step']<=screen.WINDOWS[value['window']][1]
        and type(value['ordinal']) is int and 0<=value['ordinal']<32
        and type(value['seed']) is int and 0<=value['seed']<2**32
        and all(type(value[k]) is str and len(value[k])==64 and all(c in '0123456789abcdef' for c in value[k])
                for k in ('input_sha256','task_sha256','crop_sha256'))
        and all(type(value[k]) is str and value[k] for k in ('original_choice','teacher_choice'))
        and value['original_choice']!=value['teacher_choice'])
    return value


def _source_context(case,guard):
    context=sources.open_context(guard=guard)
    _require(context.private_role_binding()[case['role']]==case['seed']
        and context.source_role(batch=2,role=case['slot'])==case['role']
        and context.task_sha256==case['task_sha256'] and context.crop_sha256==case['crop_sha256'])
    return context


def _game(context,case,guard):
    return sources.read_game(context,batch=2,role=case['slot'],opponent=case['opponent'],
                             side=case['side'],guard=guard)


def _scan_target(game,case,guard):
    identity=game.identity
    _require(all(identity.get(k)==v for k,v in dict(source_role=case['role'],batch=2,
        role=case['slot'],opponent=case['opponent'],side=case['side']).items()))
    accounting,windows=game.scan_candidates(guard=guard)
    _require(accounting.get('lines')==719)
    selected,_outside=screen._select(windows,slot=case['slot'],opponent=case['opponent'],side=case['side'])
    _require(case['rank']<len(selected[case['window']]))
    chosen=selected[case['window']][case['rank']]
    _require(chosen['step']==case['step'] and chosen['ordinal']==case['ordinal'])
    records=chosen['document']['records'];_require(case['ordinal']<len(records))
    record=records[case['ordinal']]; inputs=record['inputs']
    view=observe(observation(game.replay['steps'][case['step']],case['side']),game.replay['configuration'])
    _require(first_difference(inputs.get('observation'),view.observation) is None
        and first_difference(inputs.get('configuration'),view.configuration) is None
        and ordered_hash(inputs)==case['input_sha256'] and record.get('source')=='callback'
        and record.get('chosen')==case['original_choice'] and record.get('teacher_label')==case['teacher_choice']
        and game.replay['info']['seed']==case['seed'])
    return record


def _opponent(game,case,guard):
    item=game._item
    root=sources.old.fresh.ARTIFACTS
    anchor=root/sources.STAGE_NAME
    row_raw=sources.dagger._assert_blob(root,anchor/'job-results'/(item.game+'.row.json'),
                                       item.hashes['row'],guard)
    row=sources._json(row_raw)
    return sources.dagger._assert_blob(root,anchor/'sources'/(case['opponent']+'.py'),
                                      row['rival_sha256'],guard)


def _relative_model(path):
    actual=Path(path).resolve(strict=True)
    _require(actual.is_relative_to(GAME) and actual.is_file())
    return actual.relative_to(GAME).as_posix()


def _wrapper(mode):
    _require(mode in ('control','intervene'))
    return ("from pathlib import Path\nimport hashlib,json,sys,types\n"
        "_source=Path(globals().get('__file__',sys._getframe().f_code.co_filename)).resolve()\n"
        "_stage=_source.parent.parent\n"
        "_root=next(p for p in _stage.parents if (p/'run_match.py').is_file())\n"
        "sys.path.insert(0,str(_root))\n"
        "_config=json.loads((_stage/'gate.local.json').read_text(encoding='utf-8'))\n"
        "_raw=(_stage/'sources/actor.py').read_bytes()\n"
        "if hashlib.sha256(_raw).hexdigest()!=_config['actor_source_sha256']: raise ValueError('Actor changed')\n"
        "_mod=types.ModuleType('frozen_teacher_pair_actor')\n"
        "exec(compile(_raw,'<frozen-teacher-pair-actor>','exec',dont_inherit=True),_mod.__dict__)\n"
        + f"_agent=_mod.build(_stage,_root,{mode!r})\n"
        + "def agent(observation,configuration):\n    return _agent(observation,configuration)\nsubmission_agent=agent\n").encode()


def _prepare(stage,case,guard):
    context=_source_context(case,guard)
    with _game(context,case,guard) as game:
        _scan_target(game,case,guard)
        opponent=_opponent(game,case,guard)
        limits=game.game_limits
    task,crop=_relative_model(context._task_path),_relative_model(context._crop_path)
    actor_raw=Path(actor.__file__).read_bytes();gate_raw=Path(gate.__file__).read_bytes()
    configuration=dict(schema='teacher-pair-gate-v1',
        expected={k:case[k] for k in ('input_sha256','original_choice','teacher_choice')},
        step=case['step'],ordinal=case['ordinal'],task_checkpoint=task,task_sha256=case['task_sha256'],
        crop_checkpoint=crop,crop_sha256=case['crop_sha256'],limits=limits,
        gate_source_sha256=_digest(gate_raw),actor_source_sha256=_digest(actor_raw))
    for name in ('sources','agents','opponents','events','traces','games'):
        (stage/name).mkdir(exist_ok=False)
    files={'sources/actor.py':actor_raw,'sources/gate.py':gate_raw,
           'agents/control.py':_wrapper('control'),'agents/intervene.py':_wrapper('intervene'),
           'opponents/opponent.py':opponent,'gate.local.json':_json_bytes(configuration)}
    for name,raw in files.items(): _write(stage/name,raw)
    guard()
    return {'prepared':True},tuple(files)


def _verify_bundle(stage,case,guard):
    value=_json(stage/'gate.local.json')
    _require(type(value) is dict and value.get('schema')=='teacher-pair-gate-v1'
        and value.get('expected')=={k:case[k] for k in ('input_sha256','original_choice','teacher_choice')}
        and value.get('step')==case['step'] and value.get('ordinal')==case['ordinal']
        and value.get('task_sha256')==case['task_sha256'] and value.get('crop_sha256')==case['crop_sha256']
        and value.get('limits')==sources.dagger.GAME_LIMITS)
    for name,key in (('actor','actor_source_sha256'),('gate','gate_source_sha256')):
        _require(_sha_file(stage/'sources'/(name+'.py'),guard,MAX_SMALL)==value[key])
    for mode in ('control','intervene'):
        _require((stage/'agents'/(mode+'.py')).read_bytes()==_wrapper(mode))
    for name in ('task','crop'):
        model=actor._relative(GAME,value[name+'_checkpoint'])
        _require(_sha_file(model,guard,16*1024**2)==case[name+'_sha256'])
    return value


def _engine(guard):
    from learning.task_student_continuation_v1 import official_step
    screen_experiment._engine_identity(official_step,guard)
    local=sources.old.fresh.REPO/'kaggriculture'/'run_match.py'
    _require(_sha_file(GAME/'run_match.py',guard,MAX_SMALL)==_sha_file(local,guard,MAX_SMALL))


def _run_match(stage,case,mode,guard):
    _verify_bundle(stage,case,guard)
    folder=stage/'games'/mode
    _require(not folder.exists())
    path=GAME/'run_match.py'
    spec=importlib.util.spec_from_file_location('private_official_entry',path)
    _require(spec is not None and spec.loader is not None)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    _require(Path(module.ROOT).resolve()==GAME)
    saved_argv,saved_cwd,saved_path=sys.argv,Path.cwd(),list(sys.path)
    saved_native,saved_argparse=module.native_output,module.argparse
    try:
        module.native_output=lambda _log:nullcontext()
        sys.path.insert(0,str(GAME))
        sys.argv=[str(path),'--bot',str(stage/'agents'/(mode+'.py')),
            '--opponent',str(stage/'opponents/opponent.py'),'--side',str(case['side']),
            '--seed',str(case['seed']),'--name',mode,'--output-root',str(stage/'games')]
        class PrivateParser(argparse.ArgumentParser):
            def parse_args(self,*args,**kwargs):
                parsed=super().parse_args(*args,**kwargs)
                # The official CLI already parsed the seed; do not leave it
                # in sys.argv while untrusted agent callbacks are running.
                sys.argv=[str(path)]
                return parsed
        module.argparse=types.SimpleNamespace(ArgumentParser=PrivateParser)
        os.chdir(GAME.parent)
        class Quiet:
            def write(self,value): return len(value)
            def flush(self): pass
        # The CLI prints private paths and the seed at completion. Suppress
        # both Python streams; inherited native handles remain supervisor-owned.
        with redirect_stdout(Quiet()),redirect_stderr(Quiet()):
            guard();code=module.main();guard()
        _require(code==0)
    finally:
        sys.argv=saved_argv;sys.path[:]=saved_path;module.native_output=saved_native
        module.argparse=saved_argparse;os.chdir(saved_cwd)
    result=_json(folder/'result.json')
    complete=_json(stage/'events'/(mode+'.complete.local.json'))
    _require(result.get('passed') is True and result.get('done') is True
        and result.get('statuses')==['DONE','DONE'] and result.get('recorded_states')==720
        and result.get('action_turns')==719 and result.get('agent_errors')==[]
        and result.get('seed')==case['seed'] and result.get('bot_side')==case['side']
        and result.get('engine_sha256')==screen_experiment.ENGINE_SHA
        and result.get('engine_matches_official_commit') is True
        and result.get('upstream_commit')==screen_experiment.ENGINE_COMMIT
        and complete==dict(passed=True,action_turns=719,target_verified=True))
    with gzip.open(stage/'traces'/(mode+'.jsonl.gz'),'rb') as stream:
        trace=samples.scan(stream,task_sha256=case['task_sha256'],crop_sha256=case['crop_sha256'],guard=guard)
        _require(trace['lines']==719)
    del trace
    # The historic scanner checks continuity, not all absolute/per-turn limits.
    # Reuse the frozen verifier on a second explicitly bounded, guarded pass.
    with gzip.open(stage/'traces'/(mode+'.jsonl.gz'),'rb') as stream:
        _validate_trace_budget(stream,case,guard)
    names=[path.relative_to(stage).as_posix() for path in folder.rglob('*') if path.is_file()]
    names.extend(('traces/'+mode+'.jsonl.gz','events/'+mode+'.target.local.json',
                  'events/'+mode+'.complete.local.json'))
    _require('games/'+mode+'/replay.json' in names and 'games/'+mode+'/result.json' in names)
    return dict(states=720,action_turns=719,both_done=True,agent_errors=0),tuple(sorted(names))


def _trace_lines(stream,guard):
    """Bounded, single-pass decoder for the independent budget verifier."""
    decoded=0
    for _step in range(719):
        guard();raw=stream.readline(samples.MAX_LINE+1)
        _require(type(raw) is bytes and 0<len(raw)<=samples.MAX_LINE)
        decoded+=len(raw);_require(decoded<=samples.MAX_DECODED)
        yield json.loads(raw,parse_constant=lambda _value: (_ for _ in ()).throw(ValueError('Invalid numeric value')))
    guard();_require(stream.read(1)==b'')


def _validate_trace_budget(stream,case,guard):
    """Run the historic budget checks without retaining its diagnostic rows."""
    _require(threading.active_count()==1)
    saved=trace_budget.SAMPLE_TURNS
    try:
        # In this frozen verifier SAMPLE_TURNS only determines which already
        # validated rows to deepcopy; an empty set changes no budget checks.
        trace_budget.SAMPLE_TURNS=()
        retained=trace_budget.validate_trace(_trace_lines(stream,guard),sources.dagger.GAME_LIMITS,
                                             case['task_sha256'],case['crop_sha256'])
        _require(retained=={})
    finally:
        trace_budget.SAMPLE_TURNS=saved


def _replay(stage,mode):
    value=_json(stage/'games'/mode/'replay.json',MAX_REPLAY)
    _require(type(value) is dict and type(value.get('steps')) is list and len(value['steps'])==720)
    return value


def _audit(stage,case,mode,guard):
    value=_replay(stage,mode)
    receipt=audit.reexecute(value,guard=guard)
    result=_json(stage/'games'/mode/'result.json')
    official=receipt['official']
    _require(official['semantic_sha256']==result['semantic_replay_sha256']
        and official['semantic_sha256_v2']==result['semantic_replay_sha256_v2']
        and official['final_rewards']==result['rewards'])
    return dict(action_turns=719,shed_order_frames_checked=720,
                semantic_sha256=official['semantic_sha256'],semantic_sha256_v2=official['semantic_sha256_v2']),()


def _control_verify(stage,case,guard):
    context=_source_context(case,guard)
    with _game(context,case,guard) as game:
        _scan_target(game,case,guard)
        historical=checks.compact(game.replay,target_step=case['step'],guard=guard)
    replay=_replay(stage,'control')
    control=checks.compact(replay,target_step=case['step'],guard=guard)
    del replay
    checks.control_matches_history(historical,control)
    expected={k:case[k] for k in ('input_sha256','original_choice','teacher_choice')}
    with _game(context,case,guard) as game:
        with game.open_trace(guard=guard) as old, gzip.open(stage/'traces/control.jsonl.gz','rt',encoding='utf-8') as current:
            checks.control_trace(old,current,target_step=case['step'],ordinal=case['ordinal'],expected=expected,guard=guard)
    return {'control_verified':True},()


def _pair_verify(stage,case,guard):
    a=_replay(stage,'control');control=checks.compact(a,target_step=case['step'],guard=guard);del a
    b=_replay(stage,'intervene');intervention=checks.compact(b,target_step=case['step'],guard=guard);del b
    checks.intervention_matches_prefix(control,intervention,target_step=case['step'],side=case['side'])
    expected={k:case[k] for k in ('input_sha256','original_choice','teacher_choice')}
    c=_json(stage/'events/control.target.local.json');t=_json(stage/'events/intervene.target.local.json')
    crop_before=checks.proof_pair(c,t,target_step=case['step'],ordinal=case['ordinal'],expected=expected)
    with gzip.open(stage/'traces/control.jsonl.gz','rt',encoding='utf-8') as left, \
         gzip.open(stage/'traces/intervene.jsonl.gz','rt',encoding='utf-8') as right:
        checks.intervention_trace(left,right,target_step=case['step'],ordinal=case['ordinal'],
                                  expected=expected,crop_before=crop_before,guard=guard)
    return {'pair_verified':True,'account':checks.account(control,intervention,side=case['side'])},()


def run_phase(*,stage,phase,case,guard):
    _require(__debug__ and phase in PHASES and callable(guard))
    case=_case(case);stage=Path(stage).resolve(strict=True)
    _require(stage.is_dir() and stage.is_relative_to((GAME/'artifacts').resolve(strict=True)))
    _engine(guard)
    # open_context rechecks GAME and repository copies of the historically
    # authenticated runtime components, plus both checkpoints and roles.
    _source_context(case,guard)
    if phase!='prepare_case': _verify_bundle(stage,case,guard)
    if phase=='prepare_case': details,names=_prepare(stage,case,guard)
    elif phase in ('control','intervene'): details,names=_run_match(stage,case,phase,guard)
    elif phase in ('control_audit','intervene_audit'):
        details,names=_audit(stage,case,phase.removesuffix('_audit'),guard)
    elif phase=='control_verify': details,names=_control_verify(stage,case,guard)
    else: details,names=_pair_verify(stage,case,guard)
    if phase!='prepare_case': _verify_bundle(stage,case,guard)
    _source_context(case,guard)
    _engine(guard)
    return dict(passed=True,phase=phase,outputs={name:_sha_file(stage/name,guard) for name in names},details=details)
