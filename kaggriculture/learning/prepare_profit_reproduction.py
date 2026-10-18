"""Create a new output directory; disclosed seeds are reproduction only."""
import argparse,json,hashlib,shutil
from pathlib import Path

def main():
    root=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--registration',type=Path,default=root.parent/'review/evidence/profit-attribution-20261019/registration.json');p.add_argument('--models',type=Path,default=root/'models');a=p.parse_args()
    cfg=json.loads(a.registration.read_text(encoding='utf-8'));out=a.output.resolve()
    files={'baseline.py':a.models/'tuned_round2.py','fertilizer.py':a.models/'ablation-20261019/fertilizer.py','both.py':root/'bots/teacher_fertilizer_terminal_v2.py','teacher.py':root/'bots/teacher_fertilizer_terminal_v2.py','v1.py':root/'bots/heuristic_v1.py','market.py':root/'bots/heuristic_v1_market.py','worker.py':root/'learning/round2_worker.py','optimize_round2.py':root/'learning/optimize_round2.py','evaluate_holdout.py':root/'learning/evaluate_holdout.py'}
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for n,path in files.items():assert sha(path)==cfg['source_sha256'][n],n
    out.mkdir(parents=True,exist_ok=False);(out/'sources').mkdir()
    for n,path in files.items():shutil.copyfile(path,out/'sources'/n)
    cfg['scope']='Reproduction of disclosed seeds; not new independent evidence'
    (out/'registration.json').write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf-8')
    freeze=dict(registration_sha256=sha(out/'registration.json'),runner_sha256=sha(root/'learning/profit_attribution.py'),source_sha256=cfg['source_sha256'],instrumentation_sha256={n:sha(root/'decision'/n) for n in ('profit_attribution.py','audit_profit_attribution.py','verify_profit_attribution.py')})
    (out/'freeze.json').write_text(json.dumps(freeze,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
