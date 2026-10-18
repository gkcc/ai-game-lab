"""Exact cash/revenue identities and descriptive paired mediator summaries."""
import argparse,json
from pathlib import Path
from statistics import mean
from collections import Counter

def analyze(root):
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    cfg=read(root/'registration.json');ledger=read(root/'test/ledger.json');audits=read(root/'official-audit.json')['rows']
    rows=[]
    for r in ledger:
        cert=next(x for x in audits if all(x[k]==r[k] for k in ('seed','opponent','side','policy','replay_gzip_sha256')))
        m=cert['metrics'];assert m['cash_residual']==0
        rows.append({k:r[k] for k in ('seed','opponent','side','policy')}|dict(metrics=m))
    lookup={(r['seed'],r['opponent'],r['side'],r['policy']):r['metrics'] for r in rows}
    paired=[]
    for s in cfg['seeds']:
        for o in cfg['opponents']:
            for side in (0,1):
                for name,a,b in [('fertilizer-baseline','baseline','fertilizer'),('both-fertilizer','fertilizer','both'),('both-baseline','baseline','both')]:
                    x=lookup[s,o,side,a];y=lookup[s,o,side,b];sales=[]
                    for crop in sorted(x['sales'].keys()|y['sales'].keys()):
                        u=x['sales'].get(crop,dict(units=0,revenue=0));v=y['sales'].get(crop,dict(units=0,revenue=0))
                        q0,q1=u['units'],v['units'];r0,r1=u['revenue'],v['revenue'];volume=price=entry=0
                        if q0 and q1:
                            p0,p1=r0/q0,r1/q1;volume=(q1-q0)*(p1+p0)/2;price=(p1-p0)*(q1+q0)/2
                        else:entry=r1-r0
                        assert abs(volume+price+entry-(r1-r0))<1e-7
                        sales.append(dict(item=crop,quantity_delta=q1-q0,revenue_delta=r1-r0,volume_term=volume,price_term=price,entry_exit_term=entry))
                    costs=sum(v for k,v in y['counts'].items() if k.startswith('cash:') and not k.startswith('cash:SELL:'))-sum(v for k,v in x['counts'].items() if k.startswith('cash:') and not k.startswith('cash:SELL:'))
                    delta=y['market_cash']-x['market_cash'];revenue=sum(r['revenue_delta'] for r in sales)
                    assert delta==revenue+costs
                    paired.append(dict(seed=s,opponent=o,side=side,contrast=name,net_profit_delta=delta,sale_revenue_delta=revenue,other_cash_delta=costs,sales=sales,cash_residual=0))
    averages={}
    for p in cfg['policies']:
        rr=[r['metrics'] for r in rows if r['policy']==p];n=len(rr);counts=Counter();prod=Counter();stock=Counter();sales={}
        for m in rr:
            counts.update(m['counts']);stock.update(m['terminal_stock'])
            for crop,d in m['production'].items():
                for k,v in d.items():prod[crop+':'+k]+=v
            for crop,d in m['sales'].items():
                sales.setdefault(crop,Counter()).update(d)
        averages[p]=dict(games=n,counts={k:v/n for k,v in counts.items()},production={k:v/n for k,v in prod.items()},terminal_stock={k:v/n for k,v in stock.items()},sales={c:dict(mean_units=d['units']/n,mean_revenue=d['revenue']/n,pooled_realized_price=d['revenue']/d['units'] if d['units'] else None) for c,d in sales.items()})
    summaries={}
    for name in cfg['contrasts']:
        rr=[r for r in paired if r['contrast']==name]
        summaries[name]={k:mean(r[k] for r in rr) for k in ('net_profit_delta','sale_revenue_delta','other_cash_delta')}
        summaries[name].update({k:mean(sum(v[k] for v in r['sales']) for r in rr) for k in ('volume_term','price_term','entry_exit_term')})
    result=dict(passed=True,averages=averages,contrasts=summaries,paired=paired,scope='Descriptive mediators and accounting identities, not independent causal shares. Volume and price terms are symmetric algebra, affected by timing and endogenous shared-market response.')
    target=root/'attribution.json';assert not target.exists();target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summaries,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);analyze(p.parse_args().folder.resolve())
