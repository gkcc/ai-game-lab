"""Check recorded fertilizer requests against episode and cash-realization windows.

Only joins to a completed official effect certificate. This is planning telemetry,
not an official marginal-profit label, and does not export raw private states.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from bots.teacher_fertilizer_terminal_v2 import saleable_bonus_nights


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    args=parser.parse_args(); root=args.folder
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    ledger=read(root/'test/ledger.json'); certificate=read(root/'test-official-audit.json')
    rows=[]
    for row in ledger:
        key=f"{row['seed']}-{row['opponent']}-{row['side']}-{row['policy']}"
        raw=(root/'test'/key/'replay.json.gz').read_bytes()
        assert hashlib.sha256(raw).hexdigest()==row['replay_gzip_sha256']
        replay=json.loads(gzip.decompress(raw)); requests=0; zero=[]
        for index in range(len(replay['steps'])-1):
            before,after=replay['steps'][index:index+2]
            farm=before[0]['observation']['farms'][row['side']]
            units=[farm['farmer'],*farm['hands']]
            action=after[row['side']]['action']
            for unit,a in enumerate([action.get('farmer',[]),*action.get('hands',[])]):
                if not a or a[0]!='FERTILIZE' or unit>=len(units):continue
                requests+=1; x,y=units[unit];tile=farm['tiles'][y][x]
                if not saleable_bonus_nights(tile,(x,y),index//replay['configuration']['turnsPerDay'],replay['configuration']):
                    zero.append({'step':index,'unit':unit,'crop':tile['crop']})
        cert=next(r for r in certificate['rows'] if r['replay_gzip_sha256']==row['replay_gzip_sha256'])
        assert cert['audit']['semantic_sha256_v2']==row['semantic_replay_sha256_v2']
        executed=cert['fertilizer_effects'].get('FERTILIZE:executed',0)
        assert executed==requests, 'Cannot call request-window counts successful without complete attribution'
        rows.append({k:row[k] for k in ('seed','side','opponent','policy','replay_gzip_sha256')} | {'successful_fertilizations':executed,'zero_saleable_window':zero})
    out={'passed':all(not r['zero_saleable_window'] for r in rows if r['policy']=='candidate'),
         'scope':'Shared economic helper, validated separately with official boundary regressions; not profitability attribution',
         'rows':rows}
    path=root/'terminal-window-audit.json';assert not path.exists()
    path.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'passed':out['passed'],'counts':{p:sum(len(r['zero_saleable_window']) for r in rows if r['policy']==p) for p in ('baseline','candidate')}}))
    assert out['passed']


if __name__=='__main__':main()
