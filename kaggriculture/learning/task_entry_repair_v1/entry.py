"""Generate a filename-compatible entry without changing frozen decisions."""
from pathlib import Path

from learning.bc_crop.agent import ROOT, sha


def generate_wrapper(out, key, cfg):
    out = Path(out).resolve()
    if Path(key).name != key or key in ('', '.', '..'):
        raise ValueError('Expected a single artifact key')
    trace = out / 'traces' / (key + '.jsonl.gz')
    path = out / 'agents' / (key + '.py')
    if trace.exists() or path.exists():
        raise FileExistsError('Preserve existing wrapper and trace')
    model = out / 'model/trained.npz'
    assert sha(ROOT / cfg['crop_checkpoint']) == cfg['crop_checkpoint_sha256']
    assert sha(model) == cfg['task_checkpoint_sha256']
    code = ('from pathlib import Path\nimport sys\n'
        '_source_path=globals().get("__file__",sys._getframe().f_code.co_filename)\n'
        'ROOT=next(p for p in Path(_source_path).resolve().parents if (p/"learning/task_bc_v1/deployment.py").is_file())\n'
        'sys.path.insert(0,str(ROOT))\nfrom learning.task_bc_v1.deployment import JointAgent\n'
        '_runner=JointAgent(ROOT/' + repr(cfg['crop_checkpoint']) + ',' + repr(cfg['crop_checkpoint_sha256']) +
        ',ROOT/' + repr(model.relative_to(ROOT).as_posix()) + ',' + repr(sha(model)) +
        ',trace=ROOT/' + repr(trace.relative_to(ROOT).as_posix()) + ')\n'
        'def agent(observation,configuration):\n    return _runner(observation,configuration)\n'
        'submission_agent=agent\n')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(code)
    return path
