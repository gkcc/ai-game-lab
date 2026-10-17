"""Fail closed on sensitive paths, credential forms and non-source Git payloads.

This is a focused publication guard, not a guarantee that arbitrary text is safe.
Inspect the staged file list and use a human review as well.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--staged',action='store_true',help='Read the exact index bytes, not the working files')
args=parser.parse_args()
cmd=['git','-C',str(ROOT),'diff','--cached','--name-only','--diff-filter=ACMR','-z'] if args.staged else ['git','-C',str(ROOT),'ls-files','-z']
paths=[p for p in subprocess.check_output(cmd).decode().split('\0') if p]
if not paths:
    raise SystemExit('No versioned files to inspect')
forbidden_parts={'.venv','artifacts','upstream','research','.ssh','.kaggle','__pycache__'}
allowed={'.py','.ps1','.md','.json','.txt','.csv','.css','.js','.png','.gitignore','.example'}
patterns={
    'absolute_windows_path':r'(?i)\b[a-z]:[\\/]',
    'host_user_path':r'(?i)/(?:home|users|mnt/[a-z])/',
    'private_key':r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    'github_credential':r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})\b',
    'api_credential':r'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}\b',
    'aws_access_key':r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
    'url_embedded_credential':r'https?://[^\s/:@]+:[^\s/@]+@',
    'email_address':r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
    'hardware_inventory':r'(?i)\b(?:NVIDIA\s+)?(?:GeForce\s+)?RTX\s*\d{4}\b|\b(?:Driver Version|CUDA Version):',
}
failures=[]
total=0
for name in paths:
    path=Path(name)
    if forbidden_parts.intersection(path.parts) or path.name.startswith('.env') and path.name!='.env.example':
        failures.append({'file':name,'reason':'forbidden_path'})
    if path.suffix.lower() not in allowed and path.name!='.gitignore':
        failures.append({'file':name,'reason':'unapproved_file_type'})
    data=subprocess.check_output(['git','-C',str(ROOT),'show',':'+name]) if args.staged else (ROOT/name).read_bytes()
    total+=len(data)
    if len(data)>2*1024*1024:
        failures.append({'file':name,'reason':'oversized_payload'})
    if path.suffix.lower()=='.png':
        if not data.startswith(b'\x89PNG\r\n\x1a\n'):
            failures.append({'file':name,'reason':'invalid_png'})
        # PNG text metadata is uncompressed for the generated scientific plots.
        content=data.decode('latin-1')
    else:
        try:
            content=data.decode('utf-8-sig')
        except UnicodeDecodeError:
            failures.append({'file':name,'reason':'unexpected_binary'})
            continue
    for reason,pattern in patterns.items():
        match=re.search(pattern,content)
        if match:
            failures.append({'file':name,'reason':reason,'line':content[:match.start()].count('\n')+1})
print(json.dumps({'passed':not failures,'files':len(paths),'bytes':total,'findings':failures},ensure_ascii=False,indent=2))
raise SystemExit(1 if failures else 0)
