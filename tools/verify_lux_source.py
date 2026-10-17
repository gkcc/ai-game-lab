"""Verify installed official Lux Python files against the pinned Git commit."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--gpu',action='store_true')
args=parser.parse_args()
pin=json.loads((ROOT/'upstreams.json').read_text(encoding='utf-8'))['lux']
upstream=ROOT/pin['path']
actual=subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()
assert actual==pin['commit']
paths=subprocess.check_output(['git','-C',str(upstream),'ls-tree','-r','--name-only','HEAD','src/luxai_runner','src/luxai_s3'],text=True).splitlines()
dist=importlib.metadata.distribution('luxai-s3')
files=[]
for path in paths:
    if not path.endswith('.py'):
        continue
    expected=subprocess.check_output(['git','-C',str(upstream),'show','HEAD:'+path])
    installed=Path(dist.locate_file(path.removeprefix('src/'))).read_bytes()
    # Git checkouts may use CRLF on Windows. Permit only this representation
    # difference; record the installed raw hash for subsequent runtime checks.
    same=expected.replace(b'\r\n',b'\n')==installed.replace(b'\r\n',b'\n')
    files.append({'file':path,'sha256':hashlib.sha256(installed).hexdigest(),
                  'official_normalized_sha256':hashlib.sha256(expected.replace(b'\r\n',b'\n')).hexdigest(),
                  'byte_identical_to_git_blob':expected==installed,'matches_official_commit':same})
assert files and all(f['matches_official_commit'] for f in files)
out=ROOT/'lux-s3/artifacts'/('setup-gpu' if args.gpu else 'setup')/'source-verification.json'
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps({'commit':actual,'files':files,'all_match':True,
    'comparison':'Official Git source; only CRLF-to-LF normalization permitted'},indent=2),encoding='utf-8')
print(f'Official Lux source verified: {len(files)} files')
