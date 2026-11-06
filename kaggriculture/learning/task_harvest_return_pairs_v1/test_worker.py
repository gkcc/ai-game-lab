"""Memory-only export, event, sample and failure-contract regressions.

Streams are BytesIO objects. There are no model forwards, official transitions,
filesystem writes or historical-data reads; fixtures are not real game states.
"""
from contextlib import contextmanager
from copy import deepcopy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from learning.task_harvest_choice_v1.fixtures import observation
from . import core, worker


def frame(step, *, label='source'):
    return [dict(observation=dict(step=step,label=label,side=p),action=None,
        status='DONE' if step==719 else 'ACTIVE',reward=20+p if step==719 else 0) for p in (0,1)]


def replay():
    return dict(name='kaggriculture',configuration=dict(episodeSteps=720,seed=None),
        info=dict(seed=17),steps=[frame(i) for i in range(720)],
        rewards=['outdated'],statuses=['outdated'])


class MemoryStreams:
    def __init__(self):self.buffers,self.closed={},[]
    @contextmanager
    def open(self,path):
        key=str(path)
        if key in self.buffers:raise FileExistsError(key)
        stream=io.BytesIO();self.buffers[key]=stream
        try:yield stream
        finally:self.closed.append(key)


def costs(controllers):
    result={key:0 for key in core.BUDGET_KEYS}
    result.update(controller_calls=controllers,task_forwards=controllers,task_rows=2*controllers)
    return result


def successful_meter(start=716):
    meter=core.Meter(start)
    for name,cap in meter.caps.items():
        meter.calls[name+'_attempts']=meter.calls[name+'_returns']=cap
    for scope,turns in (('prefix',start),('suffix',719-start)):
        for who in ('actor','rival'):
            meter.calls[scope+'_'+who+'_attempts']=meter.calls[scope+'_'+who+'_returns']=2*turns
        meter.costs[scope].update(costs(2*turns))
    result=dict(meter=meter.snapshot(),arms={arm:dict(prefix_budget=costs(start),
        suffix_budget=costs(719-start)) for arm in ('H','W')})
    return deepcopy(result['meter']),result


def sample_fixture(mutator=None):
    obs,cfg=observation('mature5dry')
    inputs=dict(observation=obs,configuration=cfg,
        candidates=[dict(id='u0:x4:y4:'+op,unit=0,target=[4,4],op=op)
                    for op in ('HARVEST','WATER')],history={})
    if mutator is not None:mutator(inputs)
    target=dict(input_sha256=core.ordered_hash(inputs),harvest_choice='u0:x4:y4:HARVEST',
        water_choice='u0:x4:y4:WATER')
    common=dict(target=target,inputs=inputs,input_sha256=target['input_sha256'],
        original_proposal=dict(chosen=target['harvest_choice']),prefix_records_sha256='0'*64,budget_before={})
    result=dict(target_pair={a:dict(arm=a,**deepcopy(common)) for a in ('H','W')},
        arms=dict(H=dict(terminal=dict(own_net_profit=20,opponent_net_profit=30,margin=-10,score=0.)),
                  W=dict(terminal=dict(own_net_profit=10,opponent_net_profit=5,margin=5,score=1.))))
    return dict(role='synthetic',seed=17,opponent='synthetic'),target,result


class WorkerTests(unittest.TestCase):
    def test_export_has_exact_720_states_source_prefix_and_new_suffix(self):
        memory=MemoryStreams();source=replay();original=deepcopy(source)
        path=Path('synthetic/H.replay.json.gz')
        with patch.object(worker,'_events',memory.open):
            writer=worker.ReplayWriter(path,source,716)
            for step in range(717,720):writer.frame(frame(step,label='new'))
            writer.finish(frame(719,label='new'))
        exported=json.loads(memory.buffers[str(path)].getvalue())
        self.assertEqual(len(exported['steps']),720)
        self.assertEqual(exported['steps'][:717],source['steps'][:717])
        self.assertEqual(exported['steps'][717:],[frame(i,label='new') for i in range(717,720)])
        self.assertEqual(exported['info'],source['info'])
        self.assertEqual(exported['configuration'],source['configuration'])
        self.assertEqual(exported['statuses'],['DONE','DONE'])
        self.assertEqual(exported['rewards'],[20,21])
        self.assertEqual(source,original)
        self.assertTrue(writer.closed)
        self.assertEqual(writer.decoded_bytes,len(memory.buffers[str(path)].getvalue()))

    def test_partial_or_overlong_export_cannot_finish_successfully(self):
        for length in (718,720):
            with self.subTest(length=length):
                memory=MemoryStreams()
                with patch.object(worker,'_events',memory.open):
                    writer=worker.ReplayWriter(Path('synthetic/partial.gz'),replay(),length-1)
                    try:
                        with self.assertRaises(ValueError):
                            if length==718:writer.finish(frame(719))
                            else:writer.frame(frame(719))
                    finally:writer.close((None,None,None))
                with self.assertRaises(json.JSONDecodeError):
                    json.loads(next(iter(memory.buffers.values())).getvalue())

    def test_event_sink_compacts_prefix_and_exports_both_complete_suffixes(self):
        memory=MemoryStreams();stream=io.BytesIO();source=replay();start=718
        sink=worker.EventSink(stream,lambda:None,Path('synthetic'),source,start)
        request=dict(farmer=['PASS'],hands=[],market=[])
        with patch.object(worker,'_events',memory.open):
            for arm in ('H','W'):
                sink(dict(kind='arm_start',arm=arm,step=start,frame=source['steps'][start]))
                for step in range(start):
                    sink(dict(kind='turn',arm=arm,scope='prefix',step=step,request=request,rival_request=request,
                        trace=dict(model_counts=costs(step+1),turn_model_counts=costs(1),
                                   raw_private_sentinel='not part of compact prefix'),
                        stable_trace_sha256='a'*64))
                sink(dict(kind='turn',arm=arm,scope='suffix',step=start,request=request,rival_request=request,
                    trace=dict(retained_suffix_marker=True),frame=frame(719,label=arm),
                    transition_returned=True,error_kind=None,target_execution=dict(status='executed')))
                sink(dict(kind='arm_end',arm=arm,frame=frame(719,label=arm),result=dict(arm=arm)))
        rows=[json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertEqual((sink.events,len(rows)),(1442,1442))
        prefix=rows[1]
        self.assertEqual(set(prefix),{'kind','arm','scope','step','request_sha256','rival_request_sha256',
            'stable_trace_sha256','model_counts','turn_model_counts'})
        self.assertNotIn(b'raw_private_sentinel',stream.getvalue())
        self.assertEqual(rows[719]['trace'],dict(retained_suffix_marker=True))
        self.assertEqual(rows[719]['target_execution'],dict(status='executed'))
        self.assertEqual(sink.decoded_bytes,len(stream.getvalue()))
        self.assertEqual(len(memory.closed),2)
        for arm,writer in sink.replays.items():
            self.assertTrue(writer.closed)
            exported=json.loads(memory.buffers[str(Path('synthetic')/(arm+'.replay.json.gz'))].getvalue())
            self.assertEqual(len(exported['steps']),720)
            self.assertEqual(exported['steps'][-1],frame(719,label=arm))

    def test_failed_transition_does_not_become_a_complete_replay(self):
        memory=MemoryStreams();sink=worker.EventSink(io.BytesIO(),lambda:None,Path('synthetic'),replay(),718)
        with patch.object(worker,'_events',memory.open):
            sink(dict(kind='arm_start',arm='H',step=718,frame=frame(718)))
            request=dict(farmer=['PASS'],hands=[],market=[])
            for step in range(718):
                sink(dict(kind='turn',arm='H',scope='prefix',step=step,request=request,rival_request=request,
                    trace=dict(model_counts=costs(step+1),turn_model_counts=costs(1)),
                    stable_trace_sha256='a'*64))
            sink(dict(kind='turn',arm='H',scope='suffix',step=718,
                transition_returned=False,error_kind='OSError',frame=frame(718)))
            with self.assertRaises(ValueError):sink(dict(kind='arm_end',arm='H',frame=frame(719)))
            sink.close_open((None,None,None))
        self.assertEqual(sink.replays['H'].frames,719)
        self.assertEqual(sink.turns['W'],dict(prefix=0,suffix=0))
        with self.assertRaises(json.JSONDecodeError):json.loads(next(iter(memory.buffers.values())).getvalue())

    def test_constructor_write_error_is_not_masked_by_stream_close(self):
        primary=OSError('synthetic replay write failure')
        secondary=RuntimeError('synthetic stream close failure')
        class FailedContext:
            def __enter__(self):return SimpleNamespace(write=lambda _raw: (_ for _ in ()).throw(primary))
            def __exit__(self,*_active):raise secondary
        with patch.object(worker,'_events',return_value=FailedContext()):
            with self.assertRaises(OSError) as caught:
                worker.ReplayWriter(Path('synthetic/fail.gz'),replay(),0)
        self.assertIs(caught.exception,primary)

    def test_execute_failure_does_not_claim_success_or_mask_primary(self):
        primary=OSError('synthetic terminal pair failure');written=[]
        source=dict(role='synthetic',cohort='C',opponent='synthetic',side=0,trace=('unused.gz','a'*64))
        selected=dict(source=deepcopy(source),selection=dict(target=dict(step=718)))
        def write(_root,_path,value,_limit):
            written.append(deepcopy(value));raise BufferError('synthetic closure write failure')
        class Budget:
            def __enter__(self):return self
            def __exit__(self,*_active):return False
            def snapshot(self):return dict(synthetic=True)
        memory=MemoryStreams()
        with patch.object(Path,'is_dir',return_value=True),patch.object(Path,'exists',return_value=False),\
                patch.object(worker,'_assets',return_value=None),\
                patch.object(worker,'_source_replay',return_value=replay()),\
                patch.object(worker.core,'official_runtime',return_value=object()),\
                patch.object(worker,'owned_writes',return_value=Budget()),patch.object(worker,'_events',memory.open),\
                patch.object(worker,'trace_factory',return_value=lambda:None),\
                patch.object(worker.core,'run_pair',side_effect=primary):
            with self.assertRaises(OSError) as caught:
                worker.execute_case(dict(driver=SimpleNamespace(_write_new=write)),source,selected,
                    Path('synthetic'),lambda:None)
        self.assertIs(caught.exception,primary)
        self.assertEqual(len(written),1)
        self.assertFalse(written[0]['passed'])
        self.assertEqual(written[0]['status'],'failed')
        self.assertEqual(written[0]['phase'],'terminal_pair')
        self.assertEqual(written[0]['emitted_events'],0)
        self.assertNotIn('core',written[0])

    def test_sample_separates_future_targets_and_provenance_from_legal_inputs(self):
        source,target,result=sample_fixture();before=deepcopy(result)
        sample=worker.sample_for(source,target,result)
        self.assertEqual(set(sample),{'schema','inputs','choices','targets','provenance','training_admission'})
        self.assertEqual(sample['inputs'],result['target_pair']['H']['inputs'])
        self.assertEqual(sample['targets']['H_minus_W']['own_net_profit'],10)
        self.assertEqual(sample['targets']['H_minus_W']['margin'],-15)
        self.assertTrue(sample['targets']['objective_conflict'])
        self.assertFalse(sample['training_admission'])
        self.assertFalse(sample['provenance']['training_admission'])
        self.assertEqual(sample['provenance']['source'],source)
        self.assertNotIn('seed',sample['inputs'])
        self.assertNotIn('targets',sample['inputs'])
        sample['inputs']['observation']['farms'][0]['money']=999
        self.assertEqual(result,before)
        for mutator in (lambda x:x.update(seed=17),
                        lambda x:x['observation'].update(info=dict(seed=17)),
                        lambda x:x['observation']['farms'][1].update(private=dict(secret=1)),
                        lambda x:x['configuration'].update(seed=17)):
            with self.subTest(mutator=mutator):
                with self.assertRaises(ValueError):worker.sample_for(*sample_fixture(mutator))

    def test_meter_rejects_wrong_caps_missing_calls_failure_and_budget_drift(self):
        value,result=successful_meter()
        self.assertEqual(worker.meter_check(value,result,716)['controller_calls'],1438)
        for defect in ('cap','return','write_failure','unknown','scope_budget'):
            with self.subTest(defect=defect):
                value,result=successful_meter()
                if defect=='cap':value['caps']['official_step']+=1
                elif defect=='return':value['calls']['official_step_returns']-=1
                elif defect=='write_failure':value['calls']['evidence_write_failures']=1
                elif defect=='unknown':value['unknown_budget_observations']=1
                else:value['budget_by_scope']['suffix']['task_rows']+=1
                result['meter']=deepcopy(value)
                with self.assertRaises(ValueError):worker.meter_check(value,result,716)


if __name__=='__main__':unittest.main()
