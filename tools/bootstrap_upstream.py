"""Fetch pinned official checkouts without altering an existing mismatched checkout."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--target', choices=['kaggriculture', 'lux', 'all'], default='all')
args = parser.parse_args()
pins = json.loads((ROOT/'upstreams.json').read_text(encoding='utf-8'))
for name, pin in pins.items():
    if args.target not in ('all', name):
        continue
    target = ROOT/pin['path']
    if (target/'.git').exists():
        actual = subprocess.check_output(['git','-C',str(target),'rev-parse','HEAD'],text=True).strip()
        if actual != pin['commit']:
            raise SystemExit(f'{name}: existing checkout differs; preserve it and choose a clean location')
    else:
        if target.exists() and any(target.iterdir()):
            raise SystemExit(f'{name}: nonempty unrecognized directory')
        target.mkdir(parents=True,exist_ok=True)
        sparse=['kaggle_environments/envs/kaggriculture'] if name=='kaggriculture' else ['src','kits/python','docs']
        for cmd in [['git','init',str(target)], ['git','-C',str(target),'config','core.autocrlf','false'],
                    ['git','-C',str(target),'remote','add','origin',pin['url']],
                    ['git','-C',str(target),'sparse-checkout','init','--cone'],
                    ['git','-C',str(target),'sparse-checkout','set',*sparse],
                    ['git','-C',str(target),'fetch','--filter=blob:none','--depth','1','origin',pin['commit']],
                    ['git','-C',str(target),'checkout','--detach','FETCH_HEAD']]:
            subprocess.run(cmd,check=True)
    print(f'{name}: {pin["commit"]}')
