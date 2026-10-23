"""Real observation control/isolation evidence before gated development games."""
import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path

from learning.bc_crop.agent import ROOT, sha
from learning.bc_crop.experiment import read, save
from learning.bc_crop.audit import stable
from decision.audit_market_teacher import observation
from .deployment import JointAgent
from .experiment import ready


def run(out):
    out=Path(out);cfg=ready(out,phase='train')
    # An independent reviewer must check all epoch metrics and the unique choice.
    review=read(out/'post-training-review.json')
    assert review['passed'] and review['offline_entry_passed']
    for rel,digest in review['evidence_sha256'].items(): assert sha(out/rel)==digest,rel
    reloaded=read(out/'reload.json')
    assert reloaded['passed'],'Full standalone deployment export check is required'
    (out/'admission-started.json').open('x').write('{}\n')
    initial,trained=out/'model/initial.npz',out/'model/trained.npz'
    assert sha(initial)!=sha(trained)
    assert sha(trained)==reloaded['checkpoint_sha256']
    assert sha(out/'model/report.json')==reloaded['training_report_sha256']
    assert sha(out/'model/selected-export.json')==reloaded['selected_export_sha256']
    source=ROOT/cfg['demonstration_root']
    rows=read(source/'collect-ledger.json')+read(source/'collect-smoke-ledger.json')
    rows=sorted((r for r in rows if r['seed'] in cfg['validation_seeds'] and r['opponent']=='round2'),key=lambda r:r['game'])
    assert len(rows)==4
    evidence=[];calls=0
    def actor(path): return JointAgent(ROOT/cfg['crop_checkpoint'],cfg['crop_checkpoint_sha256'],path,sha(path))
    for row in rows:
        a,b=actor(initial),actor(trained)
        rp=source/'games'/row['game']/'replay.json.gz'
        assert sha(rp)==row['replay_gzip_sha256']
        with gzip.open(rp,'rt',encoding='utf-8') as f: replay=json.load(f)
        for frame in replay['steps'][:-1]:
            o=observation(frame,row['side']);c=replay['configuration']
            aa,bb=a(o,c),b(o,c);calls+=1
            ids_a=[r['chosen'] for r in a.last['records']];ids_b=[r['chosen'] for r in b.last['records']]
            changed=aa!=bb or a.last['final_reservations']!=b.last['final_reservations']
            if ids_a==ids_b or not changed: continue
            # Save the genuine source privately; public receipt only identifies it.
            safe=stable(b.last);altered=deepcopy(o);altered_cfg=deepcopy(c)
            altered_cfg['seed']=987654321
            altered['seed']=123456789
            altered['farms'][1-row['side']]['private']={'shed':{'MELON':999999},'seeds':{'MELON':999999}}
            altered['market']['future_prices']={'MELON':999999}
            for path in (initial,trained):
                isolated=actor(path);original=actor(path)
                assert isolated(altered,altered_cfg)==original(o,c)
                assert stable(isolated.last)==stable(original.last)
                bad=deepcopy(o);bad['info']={'seed':3}
                try: isolated(bad,c)
                except ValueError: pass
                else: raise AssertionError('Replay info was not rejected')
            private=out/'control-local'/f"{row['game']}-{o['step']}.json"
            save(private,dict(observation=o,configuration=c,initial_action=aa,trained_action=bb,
                              initial_trace=a.last,trained_trace=safe))
            evidence.append(dict(game=row['game'],step=o['step'],side=row['side'],
                raw_sha256=sha(private),initial_ids=ids_a,trained_ids=ids_b,
                action_changed=aa!=bb,reservations_changed=a.last['final_reservations']!=safe['final_reservations'],
                privacy_injection_unchanged=True,replay_info_rejected=True))
            break
        if evidence: break  # first deterministic real influence; not outcome selection
    receipt=dict(passed=bool(evidence),observation_calls=calls,official_transitions=0,new_games=0,
        initial_sha256=sha(initial),trained_sha256=sha(trained),crop_sha256=cfg['crop_checkpoint_sha256'],
        evidence=evidence,reload_sha256=sha(out/'reload.json'),post_training_review_sha256=sha(out/'post-training-review.json'))
    save(out/'admission.json',receipt)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    run(p.parse_args().out)
