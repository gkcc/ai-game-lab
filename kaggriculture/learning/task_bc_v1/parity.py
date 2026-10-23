"""Observation-only hybrid wiring parity on four consumed development trajectories."""
import argparse
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import time

from learning.bc_crop.agent import ROOT, CropAgent, sha
from learning.bc_crop.experiment import read, save
from learning.bc_crop.audit import stable
from learning.task_choice_v1.agent import TaskAgent
from learning.task_choice_v1.dataset import canonical
from decision.audit_market_teacher import observation


class ReservationProbe(CropAgent):
    """Observe the old selector's actual finish context without changing actions."""
    def finish(self, context, result):
        value = super().finish(context,result)
        self.complete_reservations = deepcopy(self.last['final_reservations'])
        self.complete_reservations.update(projected_shed=deepcopy(context['projected_shed']),
                                          unit_actions=deepcopy(context['actions']))
        return value


def run(out):
    from .experiment import ready, verify
    out=Path(out).resolve(); cfg=ready(out); tick=time.perf_counter()
    assert not (out/'composition-parity-started.json').exists()
    assert not (out/'composition-parity.json').exists()
    source=ROOT/cfg['demonstration_root']; model=ROOT/cfg['crop_checkpoint']
    assert sha(model)==cfg['crop_checkpoint_sha256']
    original=read(source/'registration.json')
    assert cfg['validation_seeds']==original['validation_seeds'] and len(cfg['validation_seeds'])==2
    rows=[r for name in ('collect-ledger.json','collect-smoke-ledger.json') for r in read(source/name)
          if r['seed'] in cfg['validation_seeds'] and r['opponent']=='round2']
    assert len(rows)==4
    assert {(r['seed'],r['side']) for r in rows}=={(s,i) for s in cfg['validation_seeds'] for i in (0,1)}
    save(out/'composition-parity-started.json',dict(registration_sha256=sha(out/'registration.json'),
        original_observation_trajectories=4, paired_observations=2876,new_games=0,official_transitions=0))
    certificates=[]
    for row in sorted(rows,key=lambda r:r['game']):
        path=source/'games'/row['game']/'replay.json.gz'
        assert sha(path)==row['replay_gzip_sha256']
        with gzip.open(path,'rt',encoding='utf-8') as stream: replay=json.load(stream)
        assert len(replay['steps'])==720
        hybrid=TaskAgent(crop_mode='model',checkpoint=model,expected_sha=cfg['crop_checkpoint_sha256'])
        crop=ReservationProbe(mode='model',checkpoint=model,expected_sha=cfg['crop_checkpoint_sha256'])
        transcript=hashlib.sha256(); task_calls=crop_calls=raw_differences=0
        for before,after in zip(replay['steps'],replay['steps'][1:]):
            obs=observation(before,row['side'])
            a=hybrid(deepcopy(obs),replay['configuration']); b=crop(deepcopy(obs),replay['configuration'])
            assert a==b,'Hybrid selector wiring changed complete actions'
            assert stable(hybrid.last['crop_records'])==stable(crop.last['records'])
            assert hybrid.last['final_reservations']==crop.complete_reservations
            assert hybrid.last['checkpoint_sha256']==crop.last['checkpoint_sha256']==cfg['crop_checkpoint_sha256']
            assert all(hybrid.last[k]==crop.last[k]==0 for k in ('fallbacks','load_errors','inference_errors'))
            assert all(r['chosen']==r['teacher_label'] for r in hybrid.last['records'])
            task_calls+=len(hybrid.last['records']); crop_calls+=len(hybrid.last['crop_records'])
            raw_differences+=a!=after[row['side']]['action']
            transcript.update(canonical(dict(action=a,crop_records=stable(crop.last['records']),
                reservations=crop.complete_reservations)).encode('utf-8'))
        certificates.append(dict(game=row['game'],seed=row['seed'],side=row['side'],
            replay_gzip_sha256=row['replay_gzip_sha256'],paired_observations=719,
            actions_equal=719,crop_records_equal=True,complete_reservations_equal=True,
            hybrid_vs_original_teacher_raw_action_differences=raw_differences,
            task_calls=task_calls,crop_calls=crop_calls,comparison_transcript_sha256=transcript.hexdigest()))
    result=dict(passed=True,rows=certificates,paired_observations=2876,adapter_calls=5752,
        wall_seconds=time.perf_counter()-tick,device='CPU observation-only parity; no training',
        new_games=0,official_transitions=0,training_runs=0,crop_checkpoint_sha256=cfg['crop_checkpoint_sha256'],
        boundary='Wiring parity on original teacher-reached validation observations only. The hybrid uses epoch6 crop choices, so it need not match teacher raw actions. This is neither a hybrid closed-loop trajectory nor proof of distribution robustness.')
    save(out/'composition-parity.json',result); verify(out)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True,type=Path)
    run(parser.parse_args().out)
