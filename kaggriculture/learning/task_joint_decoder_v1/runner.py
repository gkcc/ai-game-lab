"""Registered four-game development check; no training or fresh seed search.

Use the game venv with -I -B. Preparation starts no game. --run closes the
four-job campaign; --results must independently finish all trace/replay
comparisons before a complete technical result exists. Raw data remain local.
"""
import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import time
from types import ModuleType


REPO = Path(__file__).resolve().parents[3]
GAME = REPO.parent.parent / 'kaggriculture'
BASE = 'a68c818ca4f73d7584e7ca8d1900fbe3f89cab2d'
PAIR_COMMIT = '501136ca371b2fd3a7ff4d60997a010401f1d6dc'
PAIR_REG = 'review/evidence/task-teacher-f-pair-v1-20261106/registration.json'
PAIR_RESULTS = 'review/evidence/task-teacher-f-pair-v1-result-20261106'
PREFIX = 'kaggriculture/learning/task_joint_decoder_v1/'
EVIDENCE = 'review/evidence/task-joint-decoder-v1-20261106'
REGISTRATION = EVIDENCE + '/registration.json'
NOTE = EVIDENCE + '/REGISTRATION.zh-CN.md'
PRIVATE = REPO.parent / 'private/task-joint-decoder-v1-20261106'
PROBE = REPO.parent / 'private/task-joint-decoder-micro-v1-20261106'
STAGE = GAME / 'artifacts/task-joint-decoder-v1-20261106'
INFRA = 'kaggriculture/learning/task_teacher_fresh_panel_v1/runner.py'
READY = 'kaggriculture/learning/task_teacher_fresh_panel_v1/contract.py'
PROBE_FILES = tuple(name + '.local.json' for name in ('registration', 'started', 'result', 'closed')) + tuple(
    arm + '-' + name + '.local.json' for arm in ('greedy', 'joint')
    for name in ('initial', 'plan-0', 'plan-1', 'step-0', 'step-1'))
MAX_SMALL = 512 * 1024
SCHEMA = 'task-joint-decoder-prereg-v1'


def require(value):
    if not value: raise ValueError('Joint decoder registered development check differs')


def sha(raw): return hashlib.sha256(raw).hexdigest()


def git(*args):
    result = subprocess.run(['git', '-C', str(REPO), *args], stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=45,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    require(result.returncode == 0 and len(result.stdout) <= 16*1024**2)
    return result.stdout


def infrastructure():
    raw = (REPO/INFRA).read_bytes();require(raw == git('show', BASE+':'+INFRA))
    module = ModuleType('_joint_decoder_infrastructure');module.__file__ = str(REPO/INFRA)
    exec(compile(raw, module.__file__, 'exec', dont_inherit=True), module.__dict__)
    return module


def required_sources(infra):
    old = infra.decode(git('show', PAIR_COMMIT+':'+PAIR_REG))
    require(len(old['public_sources']) == 386)
    paths = set(old['public_sources']) | {PAIR_REG, NOTE, PREFIX+'README.zh-CN.md',
        PAIR_RESULTS+'/summary.json', PAIR_RESULTS+'/certificate.json'}
    paths.update(p.relative_to(REPO).as_posix() for p in (REPO/PREFIX).glob('*.py'))
    for path, expected in old['public_sources'].items():
        raw = infra.read(REPO/path)
        if path == 'review/PROGRESS.md':
            historical = git('show', PAIR_COMMIT+':'+path)
            published = git('show', BASE+':'+path)
            require(sha(historical) == expected and published.startswith(historical) and raw.startswith(published))
        else: require(sha(raw) == expected)
    for path, commit in ((PAIR_REG, PAIR_COMMIT), (PAIR_RESULTS+'/summary.json', BASE),
                         (PAIR_RESULTS+'/certificate.json', BASE)):
        require(infra.read(REPO/path) == git('show', commit+':'+path))
    return paths


def specification(contract):
    return dict(cases=contract.sources(), policy=contract.POLICY, games=4, official_replays=4,
        source_completed_games=32, task_model_role='baseline', scope='consumed_development',
        seed_opponent_combinations=2, comparison='all_cases_then_case_and_combination_equal_means',
        acceptance='own_positive_margin_score_and_both_combinations_nonnegative_with_request_and_state_change')


def validate_registration(reg, contract):
    from_expected = json.dumps(contract.LIMITS, sort_keys=True, allow_nan=False)
    require(type(reg) is dict and reg.get('schema') == SCHEMA and reg.get('base_commit') == BASE
        and reg.get('engine_commit') == 'f994f2050d58d84f649329c1e35ce09ada7007c5'
        and reg.get('engine_sha256') == 'bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e'
        and json.dumps(reg.get('limits'), sort_keys=True, allow_nan=False) == from_expected
        and reg.get('panel') == specification(contract)
        and reg.get('evaluation') == dict(training=False,gpu=False,new_seeds=False,
            independent_evaluation=False,statistical_significance=False,promotion=False,retries=False)
        and type(reg.get('binding_commitment_sha256')) is str
        and re.fullmatch(r'[0-9a-f]{64}', reg['binding_commitment_sha256']) is not None)


def authenticate(commit):
    infra = infrastructure()
    infra.REGISTRATION, infra.PREFIX, infra.BASE = REGISTRATION, PREFIX, BASE
    infra._required_sources = lambda _old: required_sources(infra)
    infra.validate_registration = validate_registration
    reg, contract, identity, bootstrap, base = infra.authenticate(commit)
    require(not os.path.lexists(GAME/'learning/task_joint_decoder_v1'))
    return infra, reg, contract, identity, bootstrap, base


def runtime(infra, bootstrap, reg):
    modules = infra._runtime(bootstrap, reg)
    module = importlib.import_module('learning.task_joint_decoder_v1.worker')
    require(Path(module.__file__).resolve(strict=True) == (REPO/PREFIX/'worker.py').resolve(strict=True))
    modules['joint_worker'] = module
    return modules


def micro_evidence(infra, identities):
    """Hash the fixed original 14 files; decode only registration/result/close."""
    probe = infra.snapshot(PREFIX+'official_probe.py',identities[PREFIX+'official_probe.py'])
    originals = {name: infra.read(PROBE/name, MAX_SMALL) for name in PROBE_FILES}
    registration, result, closed = (infra.decode(originals[name+'.local.json'])
                                    for name in ('registration', 'result', 'closed'))
    counts = dict(initializations_attempted=2, initializations_returned=2,
                  action_transitions_attempted=4, action_transitions_returned=4)
    require(registration.get('schema') == 'joint-decoder-official-micro-registration-v1'
        and registration.get('sources') == {name:identities['kaggriculture/'+name] for name in probe.SOURCE_NAMES}
        and registration.get('specification') == probe.SPECIFICATION
        and all(registration.get(name) == expected for name,expected in
            (('engine_commit',probe.ENGINE_COMMIT),('engine_sha256',probe.ENGINE_SHA),
             ('core_sha256',probe.CORE_SHA),('spec_sha256',probe.SPEC_SHA)))
        and result.get('schema') == 'joint-decoder-official-micro-result-v1' and result.get('passed') is True
        and result.get('registration_sha256') == sha(originals['registration.local.json'])
        and result.get('fixture_scores_not_model') is True and result.get('counts') == counts
        and all(type(result.get(k)) is int and result[k] == 0 for k in ('complete_games','model_forwards'))
        and all(result.get(k) is False for k in ('training','gpu','future_return_evaluated'))
        and closed.get('schema') == 'joint-decoder-official-micro-closed-v1'
        and closed.get('outcome') == 'completed' and closed.get('error_kind') is None
        and closed.get('registration_sha256') == result['registration_sha256']
        and closed.get('result_sha256') == sha(originals['result.local.json'])
        and closed.get('counts') == counts and closed.get('checked_transitions') == 4
        and closed.get('evidence_write_errors') == [] and closed.get('retry_allowed') is False)
    return {name: sha(raw) for name, raw in originals.items()}


def prepare():
    require(os.name == 'nt' and sys.flags.isolated and sys.flags.dont_write_bytecode and __debug__)
    infra = infrastructure();head = git('rev-parse','HEAD').decode().strip()
    require(head == BASE and all(not os.path.lexists(p) for p in (PRIVATE,STAGE,REPO/REGISTRATION)))
    identities = {p:sha(infra.read(REPO/p)) for p in sorted(required_sources(infra))}
    contract = infra.snapshot(PREFIX+'contract.py', identities[PREFIX+'contract.py'])
    ready = infra.snapshot(READY, identities[READY]);base = infra.snapshot(infra.BASE_ENTRY,identities[infra.BASE_ENTRY])
    ready.await_memory(base.physical_available)
    deadline = time.monotonic()+900
    def check():
        if base.physical_available()<contract.LIMITS['running_available_bytes']: raise MemoryError('Preparation memory floor')
        if time.monotonic()>=deadline: raise TimeoutError('Preparation deadline')
        return True
    identity = infra.snapshot(infra.IDENTITY, identities[infra.IDENTITY]);infra.preparation_runtime(identity,head)
    reg = dict(schema=SCHEMA,base_commit=BASE,engine_commit=identity.ENGINE_COMMIT,engine_sha256=identity.ENGINE_SHA,
        public_sources=identities,versions=infra._versions(),limits=contract.LIMITS,panel=specification(contract),
        evaluation=dict(training=False,gpu=False,new_seeds=False,independent_evaluation=False,
                        statistical_significance=False,promotion=False,retries=False))
    modules = runtime(infra,infra.snapshot(infra.BOOTSTRAP,identities[infra.BOOTSTRAP]),reg)
    context = modules['sources'].load(guard=check)
    require(base.validate_assets(context.manifests['C01'],modules['driver']))
    panel = contract.JointPanel(context.manifests['C01'].panel)
    binding = dict(schema='task-joint-decoder-binding-v1',base_commit=BASE,salt=secrets.token_hex(32),
        source_parent_commit=head,completed_sources=context.identities,micro_files=micro_evidence(infra,identities),
        default_sha256=sha(infra.read(GAME/'main.py')),plans=list(panel.plans()))
    require(len(binding['plans']) == 4 and all(sha(infra.read(REPO/p)) == h for p,h in identities.items()) and check())
    PRIVATE.mkdir(exist_ok=False)
    reg['binding_commitment_sha256'] = infra.write(PRIVATE/'binding.local.json',binding)
    validate_registration(reg,contract);(REPO/EVIDENCE).mkdir(parents=True,exist_ok=True)
    infra.write(REPO/REGISTRATION,reg)
    return dict(status='prepared_no_games',frozen_games=4,official_replays=4,old_games_authenticated=32,training=False)


def marker(kind,commit,binding_sha,game=None):
    require(kind in ('attempt','permit','started','consumed'))
    result = dict(schema='task-joint-decoder-marker-v1',kind=kind,registration_commit=commit,
                  binding_sha256=binding_sha,retry_allowed=False)
    if kind != 'attempt': require(type(game) is str);result['game'] = game
    return result


def untouched(exists=os.path.lexists):
    require(all(not exists(p) for p in (STAGE,PRIVATE/'attempt.local.json',PRIVATE/'owner.local.json',PRIVATE/'closed.local.json')))


def binding_read(infra,reg):
    raw = infra.read(PRIVATE/'binding.local.json');binding = infra.decode(raw)
    require(sha(raw) == reg['binding_commitment_sha256'] and binding.get('schema') == 'task-joint-decoder-binding-v1'
        and binding.get('base_commit') == BASE and binding.get('source_parent_commit') == BASE)
    return raw,binding


def source_context(infra,reg,modules,base,contract,binding,check,*,child=False):
    context = modules['sources'].load(guard=check,verify_game_files=not child)
    require(context.identities == binding['completed_sources']
        and sha(infra.read(GAME/'main.py')) == binding['default_sha256']
        and micro_evidence(infra,reg['public_sources']) == binding['micro_files']
        and base.validate_assets(context.manifests['C01'],modules['driver']))
    panel = contract.JointPanel(context.manifests['C01'].panel)
    require(len(panel.plans()) == 4 and list(panel.plans()) == binding['plans'])
    return context,panel


def budget_check(dm,contract,*,reserve=0):
    limits = contract.LIMITS
    require(dm._owned_size(PRIVATE) <= limits['private_result_bytes']
        and (not STAGE.exists() or dm._owned_size(STAGE)+reserve <= limits['total_output_bytes'])
        and shutil.disk_usage(GAME/'artifacts').free >= limits['disk_floor_bytes']+reserve)


def bundle_read(dm,acceptance,plan):
    return {k:dm._read(STAGE,STAGE/'job-results'/(plan['game']+suffix),
        8*1024**2 if k=='metrics' else acceptance.MAX_RECEIPT_BYTES) for k,suffix in dm.RECEIPTS.items()}


def accept_one(modules,manifest,panel,plan,supervision,check):
    dm,acceptance = modules['driver'],modules['acceptance'];check()
    bundle = bundle_read(dm,acceptance,plan)
    metrics = acceptance.accept(panel,plan,bundle,supervision,
        lambda p,r:modules['joint_worker'].verify_files(modules,manifest,STAGE,p,r))
    row,audit = (acceptance._json(bundle[k],acceptance.MAX_RECEIPT_BYTES) for k in ('row','audit'))
    require(audit['official'].get('shed_order_frames_checked') == 720)
    return row,metrics,audit,{k:sha(v) for k,v in bundle.items()}


def close_preserving_primary(writer,value,error):
    """A close failure cannot replace an existing failure or count as success."""
    try: writer(value)
    except BaseException:
        if error is None: raise
        return False
    return True


def trace_summary(turns,task_sha,primary):
    """Count actual current callback records, not checkpoint/weight changes."""
    from learning.task_joint_decoder_v1.decoder import candidate_key,choose
    count,rows,changed,seen = 0,0,0,0
    for step,turn in enumerate(turns):
        require(step<719 and type(turn) is dict and turn.get('step') == step
            and turn.get('decoder_schema_version') == 'task-joint-decoder-v1'
            and turn.get('task_checkpoint_sha256') == task_sha
            and all(turn.get(k)==0 for k in ('load_errors','inference_errors','fallbacks')))
        records,decoded = turn.get('records'),turn.get('decoder_records')
        require(type(records) is list and type(decoded) is list and len(records)==len(decoded))
        turn_rows = 0
        for ordinal,(record,decision) in enumerate(zip(records,decoded)):
            candidates = record['inputs']['candidates'];logits = decision['logits']
            require([candidate_key(c) for c in candidates] == sorted(candidate_key(c) for c in candidates))
            selected = choose(candidates,logits)
            original = candidates[max(range(len(logits)),key=lambda i:logits[i])]['id']
            require(type(decision.get('ordinal')) is int and decision['ordinal'] == ordinal and record.get('source') == 'callback'
                and decision.get('original_network_choice') == original
                and decision.get('decoder_choice') == record.get('chosen') == selected.choice_id
                and decision.get('matching_ids') == list(selected.matching_ids)
                and type(decision.get('matching_cardinality')) is int and decision['matching_cardinality'] == selected.cardinality
                and decision.get('matching_logit_sum') == selected.total_logit)
            changed += original != selected.choice_id;turn_rows += len(candidates)
        count += len(records);rows += turn_rows;seen += 1
        cumulative,local = turn['model_counts'],turn['turn_model_counts']
        require(all(type(v) is int and v>=0 for counts in (cumulative,local) for v in counts.values())
            and cumulative['task_forwards']==count and cumulative['task_rows']==rows
            and cumulative['controller_calls']==seen and local['task_forwards']==len(records)
            and local['task_rows']==turn_rows and local['controller_calls']==1)
    require(seen==719 and count>0 and cumulative==primary and primary['crop_forwards']>0)
    return dict(trace_turns=seen,primary_model_counts=primary,decoder_callbacks=count,
        decoder_candidate_rows=rows,decoder_changed_callbacks=changed,
        original_network_forward_calls=count,additional_network_forwards=0)


def compact_replay(replay,check):
    """One already bounded replay resident; emit only compact per-frame hashes."""
    from decision.semantics import canonical
    from learning.task_request_gate_v1.policy import request_digest
    from learning.task_teacher_counterfactual_pair_v1.audit import _shed
    from learning.task_bc_error_diagnostics_v1.shadow import ordered_hash
    def digest(value):
        return sha(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
    require(type(replay.get('steps')) is list and len(replay['steps'])==720
        and replay.get('configuration',{}).get('seed') is None
        and type(replay.get('info',{}).get('seed')) is int)
    states,requests = [],[]
    for index,frame in enumerate(replay['steps']):
        check();require(type(frame) is list and len(frame)==2
            and all(type(p) is dict and type(p.get('observation')) is dict for p in frame)
            and frame[0]['observation']['step']==index)
        # Logs and actions are not physical effects. v2 keeps backpack order;
        # the established extra helper keeps both shed mapping orders too.
        physical = [{k:p[k] for k in ('observation','reward','status')} for p in frame]
        states.append((digest(canonical(physical)),ordered_hash([_shed(frame,s) for s in (0,1)])))
        if index: requests.append(tuple(request_digest(p['action']) for p in frame))
    initial = [f['money'] for f in replay['steps'][0][0]['observation']['farms']]
    final = [f['money'] for f in replay['steps'][-1][0]['observation']['farms']]
    require(len(initial)==len(final)==2 and all(p['status']=='DONE' for p in replay['steps'][-1])
        and [p['reward'] for p in replay['steps'][-1]]==final)
    return dict(configuration=digest(canonical(replay['configuration'])),seed=replay['info']['seed'],
        states=states,requests=requests,initial_banks=initial,banks=final)


def compare_compacts(before,after,side):
    require(type(side) is int and side in (0,1) and before['configuration']==after['configuration']
        and before['seed']==after['seed'] and before['initial_banks']==after['initial_banks']
        and len(before['states'])==len(after['states'])==720
        and len(before['requests'])==len(after['requests'])==719)
    request = next((i for i,(a,b) in enumerate(zip(before['requests'],after['requests'])) if a[side]!=b[side]),None)
    state = next((i for i,(a,b) in enumerate(zip(before['states'],after['states'])) if a!=b),None)
    end = 719 if request is None else request+1
    require(all(a[1-side]==b[1-side] for a,b in zip(before['requests'][:end],after['requests'][:end])))
    require(state is None or request is not None and state>request)
    return dict(first_request_difference=request,first_state_difference=state)


def run(commit,*,preflight=False,game=None):
    infra,reg,contract,identity,bootstrap,base = authenticate(commit)
    binding_raw,binding = binding_read(infra,reg);commitment = sha(binding_raw)
    child = game is not None;require(not (child and preflight))
    started = time.monotonic();latest = dict(available_bytes=None,sample_monotonic=None)
    def available():
        value=base.physical_available();latest.update(available_bytes=value,sample_monotonic=time.monotonic());return value
    import psutil
    owner = infra.decode(infra.read(PRIVATE/'owner.local.json')) if child else dict(
        schema='task-pair-fresh-multi-owner-v1',registration_commit=commit,binding_sha256=commitment,
        pid=os.getpid(),created_us=int(psutil.Process().create_time()*1_000_000),
        deadline=started+contract.LIMITS['total_wall_seconds'])
    owned=False;accepted=[];supervisions=[];games_sha=None;owner_sha=None;attempt_sha=None
    if child: owner_sha=sha(infra.read(PRIVATE/'owner.local.json'))
    def check():
        if available()<contract.LIMITS['running_available_bytes']: raise MemoryError('Runtime memory floor')
        if time.monotonic()>=owner['deadline']: raise TimeoutError('Campaign deadline')
        require(infra.read(PRIVATE/'binding.local.json')==binding_raw and not os.path.lexists(PRIVATE/'closed.local.json'))
        if owner_sha is not None: require(sha(infra.read(PRIVATE/'owner.local.json'))==owner_sha)
        return True
    ready=infra.snapshot(READY,reg['public_sources'][READY],commit)
    def gate():
        return ready.await_memory(available,timeout=min(contract.LIMITS['readiness_timeout_seconds'],
                                  owner['deadline']-time.monotonic()))
    try:
        if not child:
            untouched()
            if not preflight:
                attempt_sha=infra.write(PRIVATE/'attempt.local.json',marker('attempt',commit,commitment));owned=True
                owner_sha=infra.write(PRIVATE/'owner.local.json',owner)
            gate()
        else: check()
        modules=runtime(infra,bootstrap,reg);dm,acceptance=modules['driver'],modules['acceptance']
        if child:
            modules['fresh'].verify_live_owner(owner,commit=commit,binding_sha=commitment,current_pid=os.getpid(),
                process=psutil.Process,executable=sys.executable,samefile=os.path.samefile,now=time.monotonic())
            require(infra.decode(infra.read(PRIVATE/'attempt.local.json'))==marker('attempt',commit,commitment))
        context,panel=source_context(infra,reg,modules,base,contract,binding,check,child=child)
        manifest=context.manifests['C01'];budget_check(dm,contract,reserve=contract.LIMITS['per_game_reserve_bytes'])
        if preflight: return dict(status='ready',games_started=0,frozen_games=4,old_games_authenticated=32)
        if child:
            plans=[p for p in panel.plans() if p['game']==game];require(len(plans)==1)
            require(infra.read(STAGE/'manifest.local.json')==binding_raw and infra.decode(infra.read(
                STAGE/'job-results'/(game+'.permit.local.json')))==marker('permit',commit,commitment,game))
            for kind in ('started','consumed'):
                infra.write(STAGE/'job-results'/(game+'.'+kind+'.local.json'),marker(kind,commit,commitment,game),MAX_SMALL)
            modules['joint_worker'].execute(modules,manifest,STAGE,plans[0],check)
            require(check() and sha(infra.read(GAME/'main.py'))==binding['default_sha256'])
            identity.verify_public(repo=REPO,game_root=GAME,registration_commit=commit)
            return None
        driver=dm.LocalDriver(manifest=manifest,game_root=GAME,stage=STAGE,commit=commit,command_for=lambda _name:[],check=check)
        require(driver.claim());infra.write(STAGE/'manifest.local.json',binding)
        for plan in panel.plans():
            readiness=gate();require(check() and len(accepted)<4
                and owner['deadline']-time.monotonic()>contract.LIMITS['job_wall_seconds']+contract.LIMITS['cleanup_seconds'])
            budget_check(dm,contract,reserve=contract.LIMITS['per_game_reserve_bytes']);name=plan['game']
            readiness_sha=infra.write(STAGE/'job-results'/(name+'.readiness.local.json'),readiness,MAX_SMALL)
            infra.write(STAGE/'job-results'/(name+'.permit.local.json'),marker('permit',commit,commitment,name),MAX_SMALL)
            command=[sys.executable,'-I','-B',str(Path(__file__).resolve()),'--registration-commit',commit,'--game',name]
            limits=contract.LIMITS
            supervision=infra.supervise_recorded(modules['campaign'],command,GAME,dict(
                max_rss_bytes=limits['max_rss_bytes'],max_commit_bytes=limits['max_commit_bytes'],
                min_available_bytes=limits['running_available_bytes'],wall_seconds=limits['job_wall_seconds'],
                cleanup_seconds=limits['cleanup_seconds'],poll_seconds=limits['poll_seconds']))
            supervisions.append(supervision)
            spath=STAGE/'job-results'/(name+'.supervision.local.json')
            supervision_sha=infra.write(spath,dict(game=name,supervision=supervision,retry_allowed=False),MAX_SMALL)
            require(acceptance.supervision_ok(supervision) and check())
            row,_metrics,_audit,hashes=accept_one(modules,manifest,panel,plan,supervision,check)
            require(row['initial_banks']==modules['sources'].initial_cash_for(context,'C01',plan))
            markers={kind:infra.read(STAGE/'job-results'/(name+'.'+kind+'.local.json'),MAX_SMALL)
                     for kind in ('permit','started','consumed')}
            require(all(infra.decode(raw)==marker(k,commit,commitment,name) for k,raw in markers.items()))
            receipt=dict(schema='task-joint-decoder-accepted-v1',game=name,registration_commit=commit,
                binding_sha256=commitment,receipt_sha256=hashes,supervision_sha256=supervision_sha,
                readiness_sha256=readiness_sha,
                job_input_sha256=sha(infra.read(STAGE/'job-inputs'/(name+'.json'),16384)),
                marker_sha256={k:sha(raw) for k,raw in markers.items()},passed=True,retry_allowed=False)
            digest=infra.write(STAGE/'job-results'/(name+'.accepted.local.json'),receipt,MAX_SMALL)
            accepted.append(dict(game=name,sha256=digest));budget_check(dm,contract)
        require(len(accepted)==4 and check())
        again,_=source_context(infra,reg,modules,base,contract,binding,check)
        require(again.identities==context.identities)
        identity.verify_public(repo=REPO,game_root=GAME,registration_commit=commit)
        games_sha=infra.write(PRIVATE/'games.local.json',dict(schema='task-joint-decoder-games-v1',
            registration_commit=commit,binding_sha256=commitment,accepted=accepted,games=4,official_replays=4))
        return dict(status='games_completed_pending_comparison',games=4,official_replays=4,technical_acceptance=False)
    finally:
        if owned:
            error=sys.exc_info()[1];memory=infra.closing_memory(error,latest,available)
            closed=dict(schema='task-joint-decoder-closed-v1',registration_commit=commit,binding_sha256=commitment,
                outcome='games_completed_pending_comparison' if error is None else 'failed',
                error_kind=None if error is None else type(error).__name__,accepted_games=len(accepted),
                owner_sha256=owner_sha,attempt_sha256=attempt_sha,
                games_sha256=games_sha,child_cleanup_receipts=supervisions,elapsed_seconds=time.monotonic()-started,
                retry_allowed=False,**memory)
            close_preserving_primary(lambda value:infra.write(PRIVATE/'closed.local.json',value),closed,error)


def read_accepted(infra,modules,manifest,panel,binding,commit,commitment,check):
    closed_raw=infra.read(PRIVATE/'closed.local.json');closed=infra.decode(closed_raw)
    games_raw=infra.read(PRIVATE/'games.local.json');games=infra.decode(games_raw)
    require(closed.get('schema')=='task-joint-decoder-closed-v1'
        and closed.get('outcome')=='games_completed_pending_comparison' and closed.get('error_kind') is None
        and closed.get('registration_commit')==commit and closed.get('binding_sha256')==commitment
        and closed.get('owner_sha256')==sha(infra.read(PRIVATE/'owner.local.json'))
        and closed.get('attempt_sha256')==sha(infra.read(PRIVATE/'attempt.local.json'))
        and infra.decode(infra.read(PRIVATE/'attempt.local.json'))==marker('attempt',commit,commitment)
        and closed.get('retry_allowed') is False
        and closed.get('accepted_games')==4 and closed.get('games_sha256')==sha(games_raw)
        and type(closed.get('child_cleanup_receipts')) is list and len(closed['child_cleanup_receipts'])==4
        and all(modules['acceptance'].supervision_ok(s) for s in closed['child_cleanup_receipts'])
        and games.get('schema')=='task-joint-decoder-games-v1' and games.get('registration_commit')==commit
        and games.get('binding_sha256')==commitment and games.get('games')==games.get('official_replays')==4
        and type(games.get('accepted')) is list and len(games['accepted'])==4
        and [r.get('game') for r in games['accepted']]==[p['game'] for p in panel.plans()])
    rows=[]
    for plan,accepted,supervised in zip(panel.plans(),games['accepted'],closed['child_cleanup_receipts']):
        name=plan['game'];check()
        accepted_raw=infra.read(STAGE/'job-results'/(name+'.accepted.local.json'),MAX_SMALL)
        receipt=infra.decode(accepted_raw)
        raw=infra.read(STAGE/'job-results'/(name+'.supervision.local.json'),MAX_SMALL);job=infra.decode(raw)
        require(sha(accepted_raw)==accepted['sha256'] and receipt.get('schema')=='task-joint-decoder-accepted-v1'
            and receipt.get('game')==name and receipt.get('registration_commit')==commit
            and receipt.get('binding_sha256')==commitment and receipt.get('passed') is True
            and receipt.get('retry_allowed') is False and receipt.get('supervision_sha256')==sha(raw)
            and job.get('game')==name and job.get('retry_allowed') is False and job.get('supervision')==supervised
            and receipt.get('job_input_sha256')==sha(infra.read(STAGE/'job-inputs'/(name+'.json'),16384)))
        readiness_raw=infra.read(STAGE/'job-results'/(name+'.readiness.local.json'),MAX_SMALL)
        readiness=infra.decode(readiness_raw)
        require(receipt.get('readiness_sha256')==sha(readiness_raw) and readiness.get('consecutive_samples')==3
            and type(readiness.get('last_available_bytes')) is int and readiness['last_available_bytes']>=6*1024**3
            and type(readiness.get('elapsed_seconds')) in (int,float) and 20<=readiness['elapsed_seconds']<=300)
        markers={k:infra.read(STAGE/'job-results'/(name+'.'+k+'.local.json'),MAX_SMALL)
                 for k in ('permit','started','consumed')}
        require(receipt.get('marker_sha256')=={k:sha(v) for k,v in markers.items()}
            and all(infra.decode(v)==marker(k,commit,commitment,name) for k,v in markers.items()))
        row,metrics,audit,hashes=accept_one(modules,manifest,panel,plan,supervised,check)
        require(hashes==receipt['receipt_sha256'])
        rows.append((plan,row,metrics,audit))
    return rows,dict(games=sha(games_raw),closed=sha(closed_raw),accepted=games['accepted'])


def metrics(row):
    return dict(own=row['net_profit'],opponent=row['opponent_net_profit'],margin=row['margin'],score=row['score'])


def verify_seal(infra,contract,commit,commitment,inputs):
    seal=infra.decode(infra.read(PRIVATE/'certificate.local.json'))
    require(seal.get('schema')=='task-joint-decoder-certificate-v1' and seal.get('registration_commit')==commit
        and seal.get('binding_sha256')==commitment and seal.get('game_inputs')==inputs
        and seal.get('technical_passed') is True and seal.get('games')==seal.get('official_replays')==4)
    files={name:infra.read(PRIVATE/(name+'.local.json')) for name in ('ledger','projection','results-closed','results-attempt')}
    require(seal.get('files')=={k:sha(v) for k,v in files.items()})
    closed=infra.decode(files['results-closed']);ledger=infra.decode(files['ledger'])
    projected=infra.decode(files['projection'])
    attempt=infra.decode(files['results-attempt'])
    require(closed.get('schema')=='task-joint-decoder-results-closed-v1'
        and closed.get('outcome')=='completed' and closed.get('error_kind') is None
        and closed.get('registration_commit')==commit and closed.get('binding_sha256')==commitment
        and closed.get('completed_comparisons')==4 and closed.get('retry_allowed') is False
        and attempt==dict(registration_commit=commit,binding_sha256=commitment,retry_allowed=False)
        and ledger.get('registration_commit')==commit and ledger.get('binding_sha256')==commitment
        and contract.project(ledger['rows'])==projected)
    return dict(status='passed',technical_acceptance=True,games=4,official_replays=4,training=False,gpu=False,**projected)


def results(commit):
    infra,reg,contract,identity,bootstrap,base=authenticate(commit)
    binding_raw,binding=binding_read(infra,reg);commitment=sha(binding_raw)
    owner=infra.decode(infra.read(PRIVATE/'owner.local.json'))
    require(owner.get('registration_commit')==commit and owner.get('binding_sha256')==commitment
        and type(owner.get('deadline')) in (int,float) and math.isfinite(owner['deadline']))
    sealed=os.path.lexists(PRIVATE/'certificate.local.json')
    latest=dict(available_bytes=None,sample_monotonic=None);started=time.monotonic()
    def available():
        value=base.physical_available();latest.update(available_bytes=value,sample_monotonic=time.monotonic());return value
    def check():
        if available()<contract.LIMITS['running_available_bytes']: raise MemoryError('Comparison memory floor')
        if not sealed and time.monotonic()>=owner['deadline']: raise TimeoutError('Campaign comparison deadline')
        require(infra.read(PRIVATE/'binding.local.json')==binding_raw)
        return True
    ready=infra.snapshot(READY,reg['public_sources'][READY],commit)
    ready.await_memory(available,timeout=300 if sealed else min(300,owner['deadline']-time.monotonic()))
    modules=runtime(infra,bootstrap,reg);dm=modules['driver']
    context,panel=source_context(infra,reg,modules,base,contract,binding,check)
    accepted,inputs=read_accepted(infra,modules,context.manifests['C01'],panel,binding,commit,commitment,check)
    if sealed:
        value=verify_seal(infra,contract,commit,commitment,inputs);check();return value
    require(all(not os.path.lexists(PRIVATE/(name+'.local.json')) for name in
        ('results-attempt','results-closed','ledger','projection')))
    infra.write(PRIVATE/'results-attempt.local.json',dict(registration_commit=commit,binding_sha256=commitment,retry_allowed=False))
    error=None;rows=[]
    try:
        old_stage=GAME/'artifacts'/modules['fresh'].STAGES['C01']
        for (role,_alias,_opponent,_side),(plan,row,_metric,audit) in zip(contract.CASES,accepted):
            check();budget_check(dm,contract)
            def turns():
                for value in dm._trace(STAGE,STAGE/'traces'/(plan['game']+'.jsonl.gz'),row['trace_sha256']):
                    check();yield value
            trace=trace_summary(turns(),panel.task_model(plan)[1],audit['primary_model_counts'])
            old_name=f"dev-{plan['seed_alias']}-{plan['opponent']}-{plan['side']}-baseline"
            prior=context.rows[('C01',old_name)]
            require(row['initial_banks']==prior['initial_banks'] and prior['task_checkpoint_sha256']==row['task_checkpoint_sha256'])
            before=compact_replay(dm._replay(old_stage,old_stage/'games'/old_name/'replay.json.gz',prior['replay_gzip_sha256']),check)
            # compact_replay keeps no source Replay reference: the old large
            # object is released before the second bounded decode begins.
            after=compact_replay(dm._replay(STAGE,STAGE/'games'/plan['game']/'replay.json.gz',row['replay_gzip_sha256']),check)
            require(before['seed']==after['seed']==plan['seed'] and before['banks']==prior['banks']
                and after['banks']==row['banks'] and before['initial_banks']==prior['initial_banks'])
            comparison=compare_compacts(before,after,plan['side'])
            rows.append(dict(role=role,opponent=plan['opponent'],side=plan['side'],baseline=metrics(prior),candidate=metrics(row),
                decoder_changed_callbacks=trace['decoder_changed_callbacks'],trace_summary=trace,
                baseline_replay_sha256=prior['replay_gzip_sha256'],candidate_replay_sha256=row['replay_gzip_sha256'],
                **comparison))
            del before,after
        require(len(rows)==4 and check());projection=contract.project(rows)
        again,_=source_context(infra,reg,modules,base,contract,binding,check)
        require(again.identities==context.identities)
        identity.verify_public(repo=REPO,game_root=GAME,registration_commit=commit)
        infra.write(PRIVATE/'ledger.local.json',dict(registration_commit=commit,binding_sha256=commitment,rows=rows))
        infra.write(PRIVATE/'projection.local.json',projection);budget_check(dm,contract);check()
    except BaseException as exc:
        error=exc;raise
    finally:
        closed=dict(schema='task-joint-decoder-results-closed-v1',registration_commit=commit,binding_sha256=commitment,
            outcome='completed' if error is None else 'failed',error_kind=None if error is None else type(error).__name__,
            completed_comparisons=len(rows),retry_allowed=False,elapsed_seconds=time.monotonic()-started,
            **infra.closing_memory(error,latest,available))
        close_preserving_primary(lambda value:infra.write(PRIVATE/'results-closed.local.json',value),closed,error)
    files={name:sha(infra.read(PRIVATE/(name+'.local.json'))) for name in ('ledger','projection','results-closed','results-attempt')}
    infra.write(PRIVATE/'certificate.local.json',dict(schema='task-joint-decoder-certificate-v1',
        registration_commit=commit,binding_sha256=commitment,game_inputs=inputs,files=files,
        technical_passed=True,games=4,official_replays=4,training=False,gpu=False,new_seeds=False,
        independent_evaluation=False,statistical_significance=False,promotion=False))
    return verify_seal(infra,contract,commit,commitment,inputs)


class Parser(argparse.ArgumentParser):
    def error(self,_message): raise ValueError('Invalid command line')


def main(argv=None):
    parser=Parser(description=__doc__)
    modes=parser.add_mutually_exclusive_group()
    for name in ('prepare','preflight','run','results'): modes.add_argument('--'+name,action='store_true')
    parser.add_argument('--registration-commit');parser.add_argument('--game')
    try:
        try: args=parser.parse_args(argv)
        except SystemExit as exc:
            if exc.code in (None,0): return 0
            raise ValueError('Invalid command line') from None
        require(not args.prepare or not (args.registration_commit or args.game))
        require(not args.game or not (args.results or args.preflight))
        value=(prepare() if args.prepare else results(args.registration_commit) if args.results else
               run(args.registration_commit,preflight=args.preflight,game=args.game))
        if value is not None: print(json.dumps(value,allow_nan=False))
        return 0
    except BaseException as exc:
        kind=('memory_gate' if isinstance(exc,MemoryError) else 'time_gate' if isinstance(exc,TimeoutError)
              else 'io_error' if isinstance(exc,OSError) else 'invalid_or_runtime')
        print(json.dumps(dict(status='failed',category=kind,retry_allowed=False)),file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
