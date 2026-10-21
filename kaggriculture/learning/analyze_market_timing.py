"""Cash/sales decomposition from official traced fills; no causal percentage."""
import argparse,json
from collections import Counter
from pathlib import Path
from statistics import mean

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def analyze(root,stage):
    cfg=read(root/('registration.json' if stage=='test' else 'development-plan.json'))
    ledger=read(root/stage/'ledger.json')
    audits=[r for r in read(root/(stage+'-audit.json'))['rows'] if r['kind']==stage]
    lookup={(r['seed'],r['opponent'],r['side'],r['policy']):r for r in audits}
    assert len(lookup)==len(ledger)
    paired=[]
    for row in ledger:
        if row['policy']!='baseline':continue
        key=row['seed'],row['opponent'],row['side'];a,b=[lookup[key+(p,)]['metrics'] for p in ('baseline','candidate')]
        sales=[]
        for item in sorted(a['sales'].keys()|b['sales'].keys()):
            x=a['sales'].get(item,dict(units=0,revenue=0));y=b['sales'].get(item,dict(units=0,revenue=0))
            q0,q1=x['units'],y['units'];r0,r1=x['revenue'],y['revenue'];volume=price=entry=0
            if q0 and q1:
                p0,p1=r0/q0,r1/q1;volume=(q1-q0)*(p1+p0)/2;price=(p1-p0)*(q1+q0)/2
            else:entry=r1-r0
            assert abs(volume+price+entry-(r1-r0))<1e-7
            sales.append(dict(item=item,baseline_units=q0,candidate_units=q1,baseline_revenue=r0,candidate_revenue=r1,baseline_mean_price=r0/q0 if q0 else None,candidate_mean_price=r1/q1 if q1 else None,volume_term=volume,price_term=price,entry_exit_term=entry))
        revenue=sum(x['candidate_revenue']-x['baseline_revenue'] for x in sales)
        other=lambda m:sum(v for k,v in m['counts'].items() if k.startswith('cash:') and not k.startswith('cash:SELL:'))
        cash=other(b)-other(a);delta=b['market_cash']-a['market_cash'];assert delta==revenue+cash
        paired.append(dict(seed=key[0],opponent=key[1],side=key[2],profit_delta=delta,revenue_delta=revenue,other_cash_delta=cash,sales=sales))
    panels={}
    for name,ops in [('primary',cfg['primary_opponents']),('mirror',['teacher'])]+[(o,[o]) for o in cfg['primary_opponents']]:
        pp=[r for r in paired if r['opponent'] in ops]
        delta={k:mean(r[k] for r in pp) for k in ('profit_delta','revenue_delta','other_cash_delta')}
        delta.update({k:mean(sum(x[k] for x in r['sales']) for r in pp) for k in ('volume_term','price_term','entry_exit_term')})
        agg={}
        for policy in ('baseline','candidate'):
            rr=[r for r in audits if r['policy']==policy and r['opponent'] in ops];n=len(rr)
            counts=Counter();terminal=Counter();sales={};outcomes=Counter()
            for r in rr:
                m=r['metrics'];counts.update(m['counts']);terminal.update(m['terminal_stock']);outcomes.update(r['request_outcomes'])
                for c,v in m['sales'].items():sales.setdefault(c,Counter()).update(v)
            agg[policy]=dict(games=n,mean_counts={k:v/n for k,v in counts.items()},mean_terminal_stock={k:v/n for k,v in terminal.items()},request_outcomes_total=dict(outcomes),
                mean_observed_shed_units=mean(r['storage']['mean_observed_shed_units'] for r in rr),max_shed_units=max(r['storage']['max_observed_shed_units'] for r in rr),
                sales={c:dict(mean_units=v['units']/n,mean_revenue=v['revenue']/n,pooled_mean_price=v['revenue']/v['units'] if v['units'] else None) for c,v in sales.items()})
        panels[name]=dict(accounting_delta=delta,policies=agg)
    report=dict(passed=True,games=len(ledger),panels=panels,paired=paired,
        interpretation='Pair-level symmetric quantity/price identity and other cash. Not independent causal shares. Unit policies identical on identical inputs; held stock/prices/cash can alter later planting, fertilizer or labor choices.')
    target=root/(stage+'-accounting.json');assert not target.exists();target.write_bytes((json.dumps(report,indent=2)+'\n').encode())
    print(json.dumps(panels['primary']['accounting_delta'],indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--stage',choices=['development','test'],required=True);a=p.parse_args();analyze(a.root.resolve(),a.stage)
