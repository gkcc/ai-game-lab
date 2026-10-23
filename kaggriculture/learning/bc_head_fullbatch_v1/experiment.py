"""Register one20-update fit; reuse existing guarded developer game pipeline."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import shutil
import subprocess
from learning.bc_crop.agent import ROOT,sha
from learning.bc_crop.experiment import read,save
from learning.bc_head_v1 import experiment as previous
from learning.bc_head_diagnostics.experiment import verify as verify_geometry


def verify(out):
    cfg=previous.verify(out)
    verify_geometry(ROOT/cfg['geometry_experiment'])
    for name,digest in read(out/'protection-before.json')['runtime'].items():assert sha(ROOT.parent/name)==digest,name
    return cfg


def prepare(out,public):
    head=ROOT/'artifacts/bc-head-v1-20261023';geometry=ROOT/'artifacts/bc-head-diagnostics-v1-20261023'
    cfg=previous.verify(head);verify_geometry(geometry)
    assert read(geometry/'acceptance.json')['stage_execution_passed']
    assert not out.exists() and not subprocess.check_output(['git','-C',str(public),'status','--porcelain'])
    cfg=dict(cfg);cfg['training']=dict(cfg['training'])
    cfg['training'].update(epochs=20,batch_size=9127,sampling='all9127 original train calls every step, no padding or sampling',
        updates_per_epoch=1,checkpoint_retention='all21 output heads and ordered row predictions; all epoch metrics retained')
    cfg.update(experiment='bc-output-head-fullbatch-v1',created_utc=datetime.now(timezone.utc).isoformat(),
        baseline_head=subprocess.check_output(['git','-C',str(public),'rev-parse','HEAD'],text=True).strip(),
        geometry_experiment=geometry.relative_to(ROOT).as_posix(),optimizer_update_budget=20,
        comparison='same20 complete data traversals as prior minibatch fit,20 rather than720 optimizer steps; not an equal-step comparison',
        selection='strict minimum deployment float64 unweighted developer validation CE over0..20, earliest tie; no error-based reselection',
        scope='Only output w2/b2; existing crop controller, labels, class weights, hidden layers, normalizer, data and default unchanged',
        acceptance_A='Onlyw2/b2 change;20 real finite nonzero GPU updates; all21 heads retained and independently recomputed; exact export identity; all permitted games720/719 DONE,0 fallback and full official ordered replay plus2 file entries',
        acceptance_B='Development only: both complete seeds profit increment>0; pooled margin/score>=0; market profit/margin/score>=0; no independent improvement or promotion claim',
        stop='One fit, maximum20 updates; offline failure stops before games. No extension or hyperparameter retry, no new demonstrations, DAgger/PPO or independent-test reuse.')
    cfg.pop('test_seeds',None)
    cfg['input_sha256']=dict(cfg['input_sha256'])
    for name in ('results-corrected.json','curvature-repair-certificate.json','acceptance.json','next-stage-advice.json','final-review.json'):
        p=geometry/name;cfg['input_sha256'][p.relative_to(ROOT).as_posix()]=sha(p)
    protected=read(geometry/'protection-before.json')['runtime'].copy()
    for base in (geometry,ROOT/'learning/bc_head_diagnostics'):
        for p in base.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:protected[p.relative_to(ROOT.parent).as_posix()]=sha(p)
    names=subprocess.check_output(['git','-C',str(public),'ls-files','-z']).decode().split('\0')
    out.mkdir(parents=True,exist_ok=False);(out/'sources').mkdir()
    for p in (head/'sources').glob('*.py'):shutil.copyfile(p,out/'sources'/p.name)
    shutil.copyfile(head/'protection.json',out/'protection.json')
    save(out/'protection-before.json',dict(runtime=protected,public={n:sha(public/n) for n in names if n}))
    save(out/'registration.json',cfg)
    files=dict(read(head/'freeze.json')['runtime_files'])
    files.update(read(geometry/'freeze.json')['code'])
    for p in [*(Path(__file__).parent.glob('*.py')),ROOT/'learning/bc_head_v1/metrics.py',out/'registration.json',out/'protection.json',out/'protection-before.json']:
        files[p.relative_to(ROOT).as_posix()]=sha(p)
    save(out/'freeze.json',dict(runtime_files=files,registration_sha256=sha(out/'registration.json'),
        training_and_game_results_absent=True,local_freeze_not_external_timestamp=True))
    print('Registered one fullbatch fit:20 updates, at most14 gated games')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','freeze-candidate','games','entries','analyze']);p.add_argument('--out',type=Path,required=True);p.add_argument('--public',type=Path)
    a=p.parse_args();out=a.out.resolve()
    if a.mode=='prepare':prepare(out,a.public.resolve())
    else:
        verify(out)
        {'freeze-candidate':previous.freeze_candidate,'games':previous.games,'entries':previous.entries,'analyze':previous.analyze}[a.mode](out)
