"""Derive selected candidate coordinates and tile features from saved observations."""
import gzip
import hashlib
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    rows={r['game']:r for r in read(OUT/'reconstruction-ledger.json')['rows']}
    decisions=candidates=values=0
    for sample in read(OUT/'independent-sample-plan.json')['existing_replay_samples']:
        row=rows[sample['game']]
        with gzip.open(OUT/row['files']['inputs']['path'],'rt',encoding='utf-8') as xf, gzip.open(OUT/row['files']['labels']['path'],'rt',encoding='utf-8') as yf:
            for line,label in zip(xf,yf):
                x,y=json.loads(line),json.loads(label);decisions+=1
                o=x['observation'];own=o['farms'][o['player']]
                assert all('private' not in f and 'inventories' not in f for f in o['farms'])
                shared=dict(zip(x['shared_feature_names'],x['shared_x']))
                for k in ('step','day','hour'):assert shared[k]==o[k]
                for index in set((0,y['index'])):
                    c=x['candidates'][index];candidates+=1
                    u=c['unit'];tx,ty=c['target'];ux,uy=([own['farmer']]+own['hands'])[u]
                    for name,value in zip(x['feature_names'],c['x']):
                        if name in ('unit.index','unit.x','unit.y','target.x','target.y'):
                            expected=dict(zip(('unit.index','unit.x','unit.y','target.x','target.y'),(u,ux,uy,tx,ty)))[name]
                        elif name.startswith('carried_unit.'):
                            expected=o['private']['inventories'][u].get(name.split('.',1)[1],0)
                        elif name.startswith('task.op.'):
                            expected=name.split('.')[-1]==c['op']
                        elif name=='task.distance':expected=abs(ux-tx)+abs(uy-ty)
                        else:
                            prefix,field=name.split('.',1)
                            tile=own['tiles'][ty][tx] if prefix=='target_tile' else own['tiles'][uy][ux]
                            d=tile if isinstance(tile,dict) else {}
                            if field.startswith('kind.'):
                                kind='EMPTY' if tile is None else 'LOCKED' if tile=='LOCKED' else d.get('kind','OTHER')
                                if kind not in ('EMPTY','LOCKED','WEED','PLANT','COOP','PASTURE'):kind='OTHER'
                                expected=kind==field.split('.',1)[1]
                            elif field.startswith(('crop.','animal.')):
                                key,v=field.split('.',1);expected=d.get(key)==v
                            elif field.endswith('.present'):expected=field.rsplit('.',1)[0] in d
                            else:expected=d.get(field,0)
                        assert value==float(expected),(sample['game'],decisions,name)
                        values+=1
    result=dict(passed=True,script_sha256=sha(Path(__file__)),source_sample_plan_sha256=sha(OUT/'independent-sample-plan.json'),
        saved_decisions_checked=decisions,candidates_checked=candidates,candidate_feature_values_checked=values,
        shared_fields_checked=['step','day','hour'],
        scope='First canonical candidate and teacher-selected candidate in every row of two preselected historical games. Direct coordinate, carried inventory, operation, distance and target/unit tile derivation without importing production encoders.',
        official_transitions_executed=0,training_runs=0,new_games=0)
    p=OUT/'independent-feature-review.json';assert not p.exists()
    p.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))

if __name__=='__main__':main()
