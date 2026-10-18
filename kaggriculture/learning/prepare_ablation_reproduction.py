"""Prepare a new directory for reproducing the disclosed ablation panel."""
import argparse,json,hashlib,shutil
from pathlib import Path

def main():
    root=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--registration',type=Path,default=root.parent/'review/evidence/fertilizer-ablation-20261019/registration.json')
    p.add_argument('--models',type=Path,default=root/'models');a=p.parse_args()
    cfg=json.loads(a.registration.read_text(encoding='utf-8'));out=a.output.resolve()
    files={'baseline.py':a.models/'tuned_round2.py','fertilizer.py':a.models/'ablation-20261019/fertilizer.py','harvest.py':a.models/'ablation-20261019/harvest.py','both.py':root/'bots/teacher_fertilizer_terminal_v2.py','teacher.py':root/'bots/teacher_fertilizer_terminal_v2.py','v1.py':root/'bots/heuristic_v1.py','market.py':root/'bots/heuristic_v1_market.py','worker.py':root/'learning/round2_worker.py'}
    for n in ('optimize_round2.py','evaluate_holdout.py','fertilizer_ablation.py'):files[n]=root/'learning'/n
    for n in ('audit_fertilizer_ablation.py','verify_ablation.py'):files[n]=root/'decision'/n
    assert all(x.is_file() for x in files.values())
    out.mkdir(parents=True,exist_ok=False);(out/'sources').mkdir()
    for n,x in files.items():shutil.copyfile(x,out/'sources'/n)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    cfg['source_sha256']={n:sha(out/'sources'/n) for n in files}
    cfg['reproduction_of_registration_sha256']=sha(a.registration)
    cfg['scope']='Published seeds reused for reproduction only, not a new independent test'
    (out/'registration.json').write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf-8')
    (out/'registration.sha256').write_text(sha(out/'registration.json')+'\n')
    freeze={'registration_sha256':sha(out/'registration.json'),'source_sha256':cfg['source_sha256'],'runner_sha256':sha(files['fertilizer_ablation.py']),'audit_sha256':sha(files['audit_fertilizer_ablation.py']),'scope':cfg['scope']}
    (out/'freeze.json').write_text(json.dumps(freeze,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
