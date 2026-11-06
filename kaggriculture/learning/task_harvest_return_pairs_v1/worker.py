"""Reuse owned storage, official restoration and streaming trace verification."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import sys
import traceback

from decision.api import observe
from learning.task_anchor_once_v1.storage import owned_writes
from learning.task_harvest_window_v1.worker import _events, _load_rival, _stable, trace_factory
from learning.task_pair_dev_games_v1.capture import MAX_LINE_BYTES
from learning.task_pair_dev_games_v1.panel import GAME_LIMITS
from . import contract, core
from .policy import BoundedPairAgent
from .sources import GAME, canonical, digest


CASE_FILE, EVENT_FILE, SAMPLE_FILE = 'case.local.json','events.jsonl.gz','sample.local.json'
CASE_RESERVE=contract.LIMITS['max_receipt_bytes']
FIELDS=('own_net_profit','opponent_net_profit','margin','score')


def aggregate_terminal(value):return {k:value[k] for k in FIELDS}


def sample_for(source,target,result):
    """Future trainer input is explicit; outcomes/provenance stay separate."""
    h,w=result['target_pair']['H'],result['target_pair']['W']
    core.same_target_proofs(h,w)
    inputs=deepcopy(h['inputs'])
    view=observe(inputs['observation'],inputs['configuration'])
    contract.require(core.ordered_hash(view.observation)==core.ordered_hash(inputs['observation'])
        and core.ordered_hash(view.configuration)==core.ordered_hash(inputs['configuration'])
        and not any(k in inputs for k in ('teacher_label','chosen','teacher_priority','seed','opponent'))
        and core.ordered_hash(inputs)==target['input_sha256'])
    returns=contract.terminal_delta(*(aggregate_terminal(result['arms'][a]['terminal']) for a in contract.ARMS))
    return dict(schema='task-harvest-return-pair-private-sample-v1',inputs=inputs,
        choices={a:target['harvest_choice' if a=='H' else 'water_choice'] for a in contract.ARMS},
        targets=returns,
        provenance=dict(source=deepcopy(source),target=deepcopy(target),
            continuation='same frozen task and crop weights',scope='consumed development pilot',
            training_admission=False),training_admission=False)


class ReplayWriter:
    """Serialize copied source prefix plus new states without another engine."""
    def __init__(self,path,replay,start):
        self.context=_events(path);self.stream=self.context.__enter__()
        self.closed=False;self.frames=0;self.decoded_bytes=0
        try:
            metadata={k:deepcopy(v) for k,v in replay.items() if k not in ('steps','rewards','statuses')}
            self.write(json.dumps(metadata,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()[:-1]
                       +b',"steps":[')
            for frame in replay['steps'][:start+1]:self.frame(frame)
        except BaseException as primary:
            try:self.close(sys.exc_info())
            except BaseException as secondary:raise primary from secondary
            raise

    def write(self,raw):
        contract.require(self.decoded_bytes+len(raw)<=contract.LIMITS['reconstructed_replay_decoded_per_file'])
        contract.require(self.stream.write(raw)==len(raw))
        self.decoded_bytes+=len(raw)

    def frame(self,value):
        contract.require(self.frames<720)
        raw=json.dumps(value,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
        contract.require(len(raw)<=MAX_LINE_BYTES)
        self.write((b',' if self.frames else b'')+raw);self.frames+=1

    def finish(self,frame):
        contract.require(self.frames==720 and [p['status'] for p in frame]==['DONE','DONE'])
        tail=dict(rewards=[p['reward'] for p in frame],statuses=[p['status'] for p in frame])
        self.write(b'],'+json.dumps(tail,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()[1:])
        self.close((None,None,None))

    def close(self,active):
        if not self.closed:
            self.context.__exit__(*active);self.closed=True


class EventSink:
    def __init__(self,stream,guard,out,replay,start):
        self.stream,self.guard,self.out,self.replay,self.start=stream,guard,out,replay,start
        self.events=self.decoded_bytes=0;self.replays={}
        self.turns={a:dict(prefix=0,suffix=0) for a in contract.ARMS}

    def __call__(self,event):
        self.guard();kind,arm=event['kind'],event['arm']
        contract.require(arm in contract.ARMS and kind in ('arm_start','turn','arm_end'))
        if kind=='arm_start':
            contract.require(arm not in self.replays and event['step']==self.start)
            self.replays[arm]=ReplayWriter(self.out/(arm+'.replay.json.gz'),self.replay,self.start)
        elif kind=='turn':
            scope=event['scope'];contract.require(arm in self.replays and scope in self.turns[arm])
            self.turns[arm][scope]+=1
            if scope=='prefix':
                event=dict(kind=kind,arm=arm,scope=scope,step=event['step'],
                    request_sha256=core.request_digest(event['request']),
                    rival_request_sha256=core.request_digest(event['rival_request']),
                    stable_trace_sha256=event['stable_trace_sha256'],
                    model_counts=deepcopy(event['trace']['model_counts']),
                    turn_model_counts=deepcopy(event['trace']['turn_model_counts']))
            elif event['transition_returned'] and event['error_kind'] is None:
                self.replays[arm].frame(event['frame'])
        else:
            contract.require(self.turns[arm]==dict(prefix=self.start,suffix=719-self.start))
            self.replays[arm].finish(event['frame'])
        raw=(json.dumps(event,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode()
        contract.require(len(raw)<=MAX_LINE_BYTES and self.events<1442
            and self.decoded_bytes+len(raw)<=contract.LIMITS['per_case_decoded_bytes'])
        contract.require(self.stream.write(raw)==len(raw))
        self.events+=1;self.decoded_bytes+=len(raw)
        if kind=='arm_end':self.stream.flush()
        self.guard()

    def close_open(self,active):
        failure=None
        for writer in self.replays.values():
            try:writer.close(active)
            except BaseException as error:failure=failure or error
        if failure is not None and active[1] is None:raise failure


def _assets(dm,source,guard):
    for key in ('task','crop','rival'):
        path,expected=source[key];guard()
        contract.require(dm._hash(GAME,GAME/path,dm.MAX_MODEL if key!='rival' else dm.MAX_SOURCE)==expected)


def _source_replay(dm,source,guard):
    guard();replay=dm._replay(GAME,GAME/source['replay'][0],source['replay'][1])
    row=source['row']
    contract.require(replay['name']=='kaggriculture' and len(replay['steps'])==720
        and replay['info']['seed']==row['seed']
        and [f['money'] for f in replay['steps'][0][0]['observation']['farms']]==row['initial_banks'])
    facts=core.terminal(replay['steps'][-1],replay,source['side'])
    contract.require(facts['final_cash']==row['banks'] and facts['rewards']==row['rewards']
        and facts['own_net_profit']==row['net_profit'] and facts['opponent_net_profit']==row['opponent_net_profit']
        and facts['margin']==row['margin'] and facts['score']==row['score'])
    guard();return replay


def execute_case(modules,source,selected,out,guard):
    dm=modules['driver'];out=Path(out);target=selected['selection']['target']
    contract.require(out.is_dir() and canonical(selected['source'])==canonical(source)
        and not any((out/p).exists() for p in (CASE_FILE,EVENT_FILE,SAMPLE_FILE,'H.replay.json.gz','W.replay.json.gz')))
    header=dict(schema='task-harvest-return-pair-case-v1',source=deepcopy(source),
        selection_sha256=core.ordered_hash(selected),target=deepcopy(target),retry_allowed=False,training=False)
    meter=budget=sink=result=document=None;scans=[];phase='source';primary=None
    row={k:source[k] for k in ('role','cohort','opponent','side')}
    try:
        if target is None:
            document=dict(header,passed=True,row=dict(row,status='no_eligible_target'),events=None,sample=None,replays=None)
        else:
            _assets(dm,source,guard);replay=_source_replay(dm,source,guard)
            phase='runtime_preparation';runtime=core.official_runtime()  # Before protected output writes.
            meter=core.Meter(target['step'])
            def actor(arm):
                return BoundedPairAgent(GAME/source['crop'][0],source['crop'][1],
                    GAME/source['task'][0],source['task'][1],trace=None,limits=dict(GAME_LIMITS),target=target,arm=arm)
            def rival():return _load_rival(GAME/source['rival'][0])
            budget=owned_writes(out,contract.LIMITS['per_case_output_bytes']-CASE_RESERVE,
                allowed_paths=[EVENT_FILE,SAMPLE_FILE,'H.replay.json.gz','W.replay.json.gz'])
            phase='terminal_pair'
            with budget:
                with _events(out/EVENT_FILE) as stream:
                    sink=EventSink(stream,guard,out,replay,target['step'])
                    try:
                        result=core.run_pair(replay,
                            trace_factory(dm,GAME,GAME/source['trace'][0],source['trace'][1],guard,scans),
                            target=target,actor_factory=actor,rival_factory=rival,runtime=runtime,
                            meter=meter,guard=guard,emit=sink)
                    finally:sink.close_open(sys.exc_info())
                contract.require(sink.events==1442 and all(w.closed and w.frames==720 for w in sink.replays.values()))
                sample=sample_for(source,target,result)
                dm._write_new(out,out/SAMPLE_FILE,sample,CASE_RESERVE)
            phase='result_identity'
            contract.require(result['passed'] is True and scans==[dict(lines=719,complete=True,closed=True)]*2)
            h=result['arms']['H']['terminal'];old=source['row']
            contract.require(h['final_cash']==old['banks'] and h['own_net_profit']==old['net_profit']
                and h['opponent_net_profit']==old['opponent_net_profit'] and h['score']==old['score'])
            _assets(dm,source,guard)
            def receipt(name,limit):
                return dict(path=name,sha256=dm._hash(out,out/name,limit),bytes=(out/name).stat().st_size)
            events=dict(receipt(EVENT_FILE,contract.LIMITS['per_case_output_bytes']),
                lines=sink.events,decoded_bytes=sink.decoded_bytes,closed=True)
            replays={a:dict(receipt(a+'.replay.json.gz',contract.LIMITS['per_case_output_bytes']),
                states=720,source_prefix_states=target['step']+1,official_suffix_actions=719-target['step'],
                decoded_bytes=sink.replays[a].decoded_bytes,closed=True) for a in contract.ARMS}
            row.update(status='paired',technical_passed=True,**{a:aggregate_terminal(result['arms'][a]['terminal']) for a in contract.ARMS})
            document=dict(header,passed=True,row=row,core=result,events=events,
                sample=receipt(SAMPLE_FILE,CASE_RESERVE),replays=replays)
        guard()
    except BaseException as error:
        primary=error
        document=dict(header,passed=False,status='failed',error_kind=type(error).__name__,phase=phase,
            private_traceback=''.join(traceback.format_exception(error))[-8192:])
        raise
    finally:
        document=document or dict(header,passed=False,status='failed',phase=phase)
        document.update(meter=None if meter is None else meter.snapshot(),trace_scans=scans,
            output_budget=None if budget is None else budget.snapshot(),emitted_events=0 if sink is None else sink.events,
            case_record_reserved_bytes=CASE_RESERVE)
        try:dm._write_new(out,out/CASE_FILE,document,CASE_RESERVE)
        except BaseException:
            if primary is None:raise
    return document


def meter_check(value,result,start):
    expected=core.Meter(start)
    contract.require(type(value) is dict and canonical(value['caps'])==canonical(expected.caps)
        and value['target_step']==start and value['unknown_budget_observations']==0
        and canonical(value)==canonical(result['meter']))
    calls=value['calls']
    contract.require(all(type(v) is int and v>=0 for v in calls.values()))
    for name,cap in expected.caps.items():
        contract.require(calls.get(name+'_attempts')==calls.get(name+'_returns')==cap)
    contract.require(not any(v for k,v in calls.items() if k.endswith(('_denials','_failures'))))
    for scope,n in (('prefix',start),('suffix',719-start)):
        for who in ('actor','rival'):
            contract.require(calls.get(scope+'_'+who+'_attempts',0)==calls.get(scope+'_'+who+'_returns',0)==2*n)
        expected_cost={k:sum(result['arms'][a][scope+'_budget'][k] for a in contract.ARMS) for k in core.BUDGET_KEYS}
        contract.require(canonical(value['budget_by_scope'][scope])==canonical(expected_cost))
    cost={k:sum(value['budget_by_scope'][s][k] for s in ('prefix','suffix')) for k in core.BUDGET_KEYS}
    contract.require(cost['controller_calls']==1438 and cost['teacher_proposals']==cost['gate_evaluations']==0
        and cost['task_forwards']<=8000 and cost['crop_forwards']<=16000
        and cost['task_rows']+cost['crop_rows']<=320000)
    return cost


def verify_case(modules,source,selected,out,guard):
    """One bounded event scan and source/constructed replay decode; no inference."""
    dm,parse=modules['driver'],modules['acceptance']._json;out=Path(out)
    raw=dm._read(out,out/CASE_FILE,CASE_RESERVE);doc=parse(raw,CASE_RESERVE)
    target=selected['selection']['target']
    contract.require(doc['schema']=='task-harvest-return-pair-case-v1' and doc['passed'] is True
        and canonical(doc['source'])==canonical(source) and core.ordered_hash(doc['target'])==core.ordered_hash(target)
        and doc['selection_sha256']==core.ordered_hash(selected) and doc['retry_allowed'] is False
        and doc['training'] is False and doc['case_record_reserved_bytes']==CASE_RESERVE)
    if target is None:
        contract.require(doc['row']['status']=='no_eligible_target' and doc['meter'] is None and doc['trace_scans']==[]
            and doc['events'] is doc['sample'] is doc['replays'] is None and doc['emitted_events']==0)
        return dict(row=doc['row'],document_sha256=digest(raw),model_costs={k:0 for k in core.BUDGET_KEYS},official_actions=0)
    result=doc['core'];start,side=target['step'],source['side']
    contract.require(result['passed'] is True and result['complete_games']==result['extra_official_replays']==0
        and result['terminal_suffixes']==2 and result['training'] is False
        and doc['trace_scans']==[dict(lines=719,complete=True,closed=True)]*2)
    cost=meter_check(doc['meter'],result,start)
    core.same_target_proofs(result['target_pair']['H'],result['target_pair']['W'])
    for a in contract.ARMS:
        arm=result['arms'][a];core.verify_proof(arm['target_proof'],target,a)
        contract.require(arm['prefix_turns']==start and arm['suffix_turns']==719-start and arm['policy_turns']==719
            and arm['trace_rows_verified']==719 and arm['episode_budget']['controller_calls']==719
            and len(arm['request_sha256'])==719-start and len(arm['physical_state_sha256'])==720-start
            and arm['override_count']==int(a=='W'))
    _assets(dm,source,guard);replay=_source_replay(dm,source,guard)
    from learning.task_harvest_unexpressed_runtime_v2.runtime import validate_receipt
    from learning.task_harvest_struct_compat_v1.compat import identity as comparator_identity
    validate_receipt(result['runtime_identity']['runtime_preparation'],production=True)
    contract.require(canonical(result['runtime_identity']['request_container_binding'])==canonical(comparator_identity()))
    contract.require(doc['events']['path']==EVENT_FILE and doc['sample']['path']==SAMPLE_FILE
        and all(doc['replays'][a]['path']==a+'.replay.json.gz' for a in contract.ARMS))
    for item in [doc['events'],doc['sample'],*doc['replays'].values()]:
        contract.require(dm._hash(out,out/item['path'],contract.LIMITS['per_case_output_bytes'])==item['sha256']
            and (out/item['path']).stat().st_size==item['bytes'])
    sample_raw=dm._read(out,out/SAMPLE_FILE,CASE_RESERVE)
    contract.require(core.ordered_hash(parse(sample_raw,CASE_RESERVE))==core.ordered_hash(sample_for(source,target,result)))
    counts={a:dict(prefix=0,suffix=0) for a in contract.ARMS};ended=[];active=None;lines=decoded=0
    with gzip.open(out/EVENT_FILE,'rb') as stream:
        while True:
            guard();line=stream.readline(MAX_LINE_BYTES+1)
            if not line:break
            lines+=1;decoded+=len(line)
            contract.require(len(line)<=MAX_LINE_BYTES and lines<=1442 and decoded<=contract.LIMITS['per_case_decoded_bytes'])
            event=parse(line,MAX_LINE_BYTES);a=event['arm'];kind=event['kind'];branch=result['arms'][a]
            if kind=='arm_start':
                contract.require(active is None and len(ended)<2 and a==contract.ARMS[len(ended)] and event['step']==start
                    and core.frame_digest(event['frame'])==core.frame_digest(replay['steps'][start]))
                active=a
            elif kind=='arm_end':
                contract.require(active==a and counts[a]==dict(prefix=start,suffix=719-start)
                    and canonical(event['result'])==canonical(branch)
                    and canonical(core.terminal(event['frame'],replay,side))==canonical(branch['terminal']))
                ended.append(a);active=None
            else:
                contract.require(kind=='turn' and active==a)
                scope,step=event['scope'],event['step']
                contract.require(scope in counts[a] and type(step) is int
                    and step==counts[a][scope]+(start if scope=='suffix' else 0))
                counts[a][scope]+=1
                old=selected['trace_index'][step]
                if scope=='prefix':
                    contract.require(all(core.ordered_hash(event[k])==core.ordered_hash(old[k])
                                         for k in ('stable_trace_sha256','model_counts','turn_model_counts'))
                        and event['request_sha256']==core.request_digest(replay['steps'][step+1][side]['action'])
                        and event['rival_request_sha256']==core.request_digest(replay['steps'][step+1][1-side]['action']))
                else:
                    offset=step-start;trace=event['trace'];joint=[None,None]
                    joint[side],joint[1-side]=event['request'],event['rival_request']
                    contract.require(event['transition_returned'] is True and event['error_kind'] is None
                        and core.ordered_hash(_stable(trace))==event['stable_trace_sha256']
                        and [core.request_digest(x) for x in joint]==branch['request_sha256'][offset]
                        and core.frame_digest(event['frame'],physical=True)==branch['physical_state_sha256'][offset+1])
                    if a=='H':
                        contract.require(event['stable_trace_sha256']==old['stable_trace_sha256']
                            and core.frame_digest(event['frame'])==core.frame_digest(replay['steps'][step+1]))
                    if step==start:
                        contract.require(core.ordered_hash(event['target_execution'])==core.ordered_hash(branch['target_execution'])
                            and event['target_execution']['status']=='executed'
                            and trace['records'][target['ordinal']]['chosen']==target['harvest_choice' if a=='H' else 'water_choice'])
    contract.require(ended==list(contract.ARMS) and active is None and lines==doc['events']['lines']==doc['emitted_events']==1442
        and decoded==doc['events']['decoded_bytes'] and doc['events']['closed'] is True)
    # Reconstructed Replay JSON is an export, not an additional simulation.
    for a in contract.ARMS:
        item=doc['replays'][a];export=dm._replay(out,out/item['path'],item['sha256']);branch=result['arms'][a]
        contract.require(len(export['steps'])==720 and export['statuses']==['DONE','DONE']
            and export['rewards']==branch['terminal']['rewards']
            and canonical(export['configuration'])==canonical(replay['configuration'])
            and canonical(export['info'])==canonical(replay['info']))
        for step,frame in enumerate(export['steps']):
            expected=(core.frame_digest(replay['steps'][step],physical=True) if step<=start
                      else branch['physical_state_sha256'][step-start])
            contract.require(core.frame_digest(frame,physical=True)==expected)
            if step>start:contract.require([core.request_digest(x['action']) for x in frame]==branch['request_sha256'][step-start-1])
        del export
    values={a:aggregate_terminal(result['arms'][a]['terminal']) for a in contract.ARMS}
    delta=contract.terminal_delta(values['H'],values['W'])
    contract.require(canonical(delta['H_minus_W'])==canonical(result['terminal_delta_H_minus_W'])
        and delta['objective_conflict'] is result['target_conflict'])
    expected_row=dict({k:source[k] for k in ('role','cohort','opponent','side')},status='paired',technical_passed=True,**values)
    contract.require(canonical(doc['row'])==canonical(expected_row))
    payload=sum(doc[k]['bytes'] for k in ('events','sample'))+sum(x['bytes'] for x in doc['replays'].values())
    budget=doc['output_budget']
    contract.require(budget['limit_bytes']==contract.LIMITS['per_case_output_bytes']-CASE_RESERVE
        and budget['used_bytes']==payload and budget['refused']==0 and budget['created_count']==4
        and payload+len(raw)<=contract.LIMITS['per_case_output_bytes'])
    guard();contract.require(dm._read(out,out/CASE_FILE,CASE_RESERVE)==raw)
    return dict(row=expected_row,document_sha256=digest(raw),model_costs=cost,
        official_actions=2*(719-start),meter=doc['meter'])
