"""Actual official filename loader, from an unrelated working directory."""
import argparse
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
from .agent import ROOT, sha
from .experiment import read, save, integrity, wrapper


def run(out, source_stage='test'):
    cfg=integrity(out);ledger=read(out/(source_stage+'-ledger.json'));rows=[]
    seed=cfg['test_seeds'][0] if source_stage=='test' else cfg['train_seeds'][0]
    for side in (0,1):
        old=next(r for r in ledger if r['seed']==seed and r['side']==side and r['opponent']=='market' and r['policy']=='trained')
        key=f'entry-{source_stage}-{side}'
        bot=ROOT/'bots/bc_crop_student.py'
        trace=out/'traces'/(key+'.jsonl.gz');assert not trace.exists()
        cmd=[sys.executable,'-B','-X','utf8',str(ROOT/'run_match.py'),'--bot',str(bot),
             '--opponent',str(out/'sources/market.py'),'--side',str(side),'--seed',str(seed),
             '--name',key,'--output-root',str(out/'games')]
        completed=subprocess.run(cmd,cwd=ROOT.parent,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                                 env={**os.environ,'BC_CROP_TRACE':str(trace),
                                      'BC_CROP_MODEL_CARD':str(ROOT/'models/bc_crop_v1/checkpoint.json')},
                                 creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
        (out/(key+'-launcher.log')).write_bytes(completed.stdout+completed.stderr)
        if completed.returncode:raise RuntimeError('Official file entry failed; inspect local launcher log')
        folder=out/'games'/key;r=read(folder/'result.json')
        assert r['passed'] and r['statuses']==['DONE','DONE'] and r['recorded_states']==720
        assert r['bot_farming_preconditions_passed']
        assert r['semantic_replay_sha256_v2']==old['semantic_replay_sha256_v2'] and r['final_money']==old['banks']
        with gzip.open(folder/'replay.json.gz','wb',compresslevel=1) as f:f.write((folder/'replay.json').read_bytes())
        row={**old,'game':key,'bot_sha256':sha(bot),'agent_relative':bot.relative_to(ROOT).as_posix(),
             'trace_sha256':sha(out/'traces'/(key+'.jsonl.gz')),'replay_gzip_sha256':sha(folder/'replay.json.gz'),
             'entry':'official file loader','same_checkpoint_sha256':sha(out/'model/trained.npz')}
        rows.append(row);print(f'file entry seat {side}: complete, same trajectory and checkpoint',flush=True)
    save(out/('entry-'+source_stage+'-ledger.json'),rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source-stage',default='test',choices=['test','trained-smoke'])
    a=p.parse_args();run(a.out.resolve(),a.source_stage)
