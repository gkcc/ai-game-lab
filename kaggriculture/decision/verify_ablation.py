"""Small official-state check for the exact early-harvest toggle."""
import argparse,json
from pathlib import Path
from decision.verify_mechanisms import env,obs,step,rules

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);a=p.parse_args();agents={}
    for policy in ('baseline','fertilizer','harvest','both'):
        ns={};exec(compile((a.folder/'sources'/(policy+'.py')).read_bytes(),policy,'exec'),ns);agents[policy]=ns['agent']
    rows=[]
    for fertilized in (False,True):
        actions={}
        for policy,agent in agents.items():
            e=env();e.state[0].observation.step=216
            for s in e.state:s.observation.day=9;s.observation.hour=0
            tile=rules._new_plant('TOMATO',0,24);tile['yield_units']=2;tile['fertilized_until_day']=9 if fertilized else -1
            e.state[0].observation.farms[0]['tiles'][4][4]=tile
            action=agent(obs(e),e.configuration);actions[policy]=action['farmer']
            step(e,action)
            if action['farmer']==['HARVEST']:
                assert e.state[0].observation.private['inventories'][0].get('TOMATO',0)==2
        if fertilized:
            assert actions['both']==actions['harvest']==['HARVEST']
            assert actions['baseline']==actions['fertilizer']==['WATER']
        else:assert actions['baseline']==actions['harvest'] and actions['fertilizer']==actions['both']
        rows.append({'fertilized_fixture':fertilized,'farmer_actions':actions})
    result={'passed':True,'scope':'Two constructed official states, not full-game equivalence proof or profit test','rows':rows}
    path=a.folder/'toggle-check.json';assert not path.exists();path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))

if __name__=='__main__':main()
