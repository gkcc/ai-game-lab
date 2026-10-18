"""Focused official-transition checks of offline production instrumentation."""
import json,argparse
from pathlib import Path
from decision.verify_mechanisms import env,step,rules
from decision.profit_attribution import Attribution
from decision.semantics import canonical

def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);out=p.parse_args().output
    rows=[];original=rules._daily_refresh_plants
    for crop,day,yield_units,watered,fertilized,expected in [
        ('TOMATO',7,3,True,True,(2,1,1)),('TOMATO',7,4,True,True,(2,0,2)),
        ('TOMATO',7,0,False,True,(1,1,0)),('TOMATO',6,0,True,True,(0,0,0)),
        ('STRAWBERRY',9,3,True,True,(2,1,1)),('TOMATO',11,0,True,True,(0,0,0))]:
        def fixture():
            e=env();t=rules._new_plant(crop,0,24);t.update(yield_units=yield_units,watered_today=watered,consecutive_unwatered=0,fertilized_until_day=day if fertilized else -1)
            e.state[0].observation.farms[0]['tiles'][4][4]=t
            return e
        a=fixture();b=fixture();m=Attribution(0)
        with m.installed():step(a,at=day*24+23)
        step(b,at=day*24+23);assert canonical(a.state)==canonical(b.state)
        d=m.production.get(crop,{})
        actual=tuple(d.get(k,0) for k in ('potential_units','admitted_units','clipped_units'))
        assert actual==expected,(actual,expected)
        assert rules._daily_refresh_plants is original
        rows.append(dict(crop=crop,day=day,initial_yield=yield_units,watered=watered,expected=list(expected),actual=list(actual),states_equal=True))
    try:
        with Attribution(0).installed():raise ValueError('restore check')
    except ValueError:pass
    assert rules._daily_refresh_plants is original
    assert not out.exists();out.write_text(json.dumps(dict(passed=True,cases=rows,exception_restored=True),indent=2)+'\n',encoding='utf-8')
    print('official production boundaries passed')

if __name__=='__main__':main()
