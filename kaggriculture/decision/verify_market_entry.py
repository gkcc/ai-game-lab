"""Regression for official file loading versus a worker's named-agent loading."""
import argparse,gzip,hashlib,json
from pathlib import Path
from decision.verify_mechanisms import make,env,obs
from kaggle_environments.agent import get_last_callable
from decision.verify_market_teacher import load
from decision.audit_market_teacher import observation,check
from decision.semantics import digest

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();root=a.root
    out=root/'file-entry-check';out.mkdir(exist_ok=False)
    old=root/'sources/candidate.py';new=ROOT/'bots/teacher_market_v1.py'
    original=load(old)['agent'];named=load(new)['agent']
    bad=get_last_callable(old.read_text(encoding='utf-8'))
    e=env()
    try:bad(obs(e),e.configuration)
    except TypeError:old_error=True
    else:raise AssertionError('expected original loader arity regression')
    entry=get_last_callable(new.read_text(encoding='utf-8'))
    assert new.read_bytes()==old.read_bytes()+b'\n# Official file loader selects the last callable; keep the named worker entry identical.\nsubmission_agent = agent\n'
    rows=json.loads((root/'development/ledger.json').read_text(encoding='utf-8'));checked=0
    for r in rows:
        replay=json.loads(gzip.decompress((root/'development'/f"{r['seed']}-{r['opponent']}-{r['side']}-{r['policy']}"/'replay.json.gz').read_bytes()))
        for frame in replay['steps'][:-1]:
            o=observation(frame,r['side']);cfg=replay['configuration']
            assert original(o,cfg)==named(o,cfg)==entry(o,cfg)
            checked+=1
    # Full real official filename entry. A disclosed game, excluded from stats.
    r=next(r for r in rows if r['policy']=='candidate' and r['opponent']=='market' and r['side']==0)
    e=make('kaggriculture',configuration={'seed':r['seed']},debug=True)
    e.run([str(new.resolve()),str((root/'sources/market.py').resolve())])
    replay=e.toJSON();assert e.done and len(e.steps)==720
    assert [s.status for s in e.state]==['DONE','DONE']
    assert [s.reward for s in e.state]==r['banks']
    assert digest(e.steps,1)==r['semantic_replay_sha256'] and digest(e.steps)==r['semantic_replay_sha256_v2']
    assert not any(x.get('stderr') for frame in e.logs for x in frame)
    raw=gzip.compress(json.dumps(replay,separators=(',',':')).encode(),mtime=0)
    (out/'replay.json.gz').write_bytes(raw)
    sources=out/'sources';sources.mkdir()
    for name in ('baseline.py','market.py'):(sources/name).write_bytes((root/'sources'/name).read_bytes())
    (sources/'candidate.py').write_bytes(new.read_bytes())
    rerow={**r,'bot_sha256':sha(new),'replay_gzip_sha256':sha(out/'replay.json.gz')}
    cert=check((rerow,str(out),str(sources),'file_entry_repeat'))
    report=dict(passed=True,original_file_loader_failed=True,fix='Trailing submission_agent alias; same agent callable, no decision logic change',original_candidate_sha256=sha(old),delivered_candidate_sha256=sha(new),observations_with_identical_old_new_named_and_file_actions=checked,
        complete_filename_entry_repeat=dict(seed=r['seed'],side=0,opponent='market',banks=r['banks'],states=720,actions=719,semantic_v1_equal=True,semantic_v2_equal=True,excluded_from_statistics=True),official_repeat_certificate=cert)
    (root/'file-entry-check.json').write_bytes((json.dumps(report,indent=2)+'\n').encode())
    print(json.dumps({k:v for k,v in report.items() if k!='official_repeat_certificate'}))

if __name__=='__main__':main()
