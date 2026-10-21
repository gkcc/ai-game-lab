"""Prepare a fresh disclosed-development reproduction; never an unseen test."""
import argparse,hashlib,json,shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--evidence',type=Path,default=ROOT.parent/'review/evidence/market-teacher-v1-20261021')
    p.add_argument('--models',type=Path,default=ROOT/'models');a=p.parse_args()
    cfg=json.loads((a.evidence/'development-plan.json').read_text(encoding='utf-8'))
    # Reproduce the actual named-worker source bytes used in development. The
    # delivered Bot adds only a last-callable alias, separately verified by the
    # real official filename entry in verify_market_entry.py.
    files={'baseline.py':ROOT/'bots/teacher_fertilizer_terminal_v2.py','teacher.py':ROOT/'bots/teacher_fertilizer_terminal_v2.py','candidate.py':a.evidence/'sources/teacher_market_v1.development.py','v1.py':ROOT/'bots/heuristic_v1.py','market.py':ROOT/'bots/heuristic_v1_market.py','tuned.py':a.models/'tuned_round2.py','worker.py':ROOT/'learning/round2_worker.py'}
    payloads={}
    for name,path in files.items():
        raw=path.read_bytes();wanted=cfg['source_sha256'][name]
        variants=[raw,raw.replace(b'\r\n',b'\n'),raw.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n')]
        payloads[name]=next((b for b in variants if hashlib.sha256(b).hexdigest()==wanted),None)
        assert payloads[name] is not None,name
    out=a.output;out.mkdir(parents=True,exist_ok=False);(out/'sources').mkdir()
    for name,raw in payloads.items():(out/'sources'/name).write_bytes(raw)
    cfg['scope']='Reproduction using already disclosed development seeds; not independent testing.'
    (out/'development-plan.json').write_bytes((json.dumps(cfg,ensure_ascii=False,indent=2)+'\n').encode())
    old=json.loads((a.evidence/'development-freeze.json').read_text(encoding='utf-8'))
    freeze=dict(registration_sha256=sha(out/'development-plan.json'),source_sha256=cfg['source_sha256'],runtime_sha256={n:sha(ROOT/n) for n in old['runtime_sha256']},
        original_freeze_sha256=sha(a.evidence/'development-freeze.json'),source_line_endings='Bind actual checkout bytes; original runtime/public mapping remains in source-map.json.',test_results_seen=True,reproduction_only=True)
    (out/'development-freeze.json').write_bytes((json.dumps(freeze,indent=2)+'\n').encode())
    print(json.dumps(dict(prepared=True,games=cfg['development_games'],candidate_sha256=cfg['source_sha256']['candidate.py'],reproduction_only=True)))

if __name__=='__main__':main()
