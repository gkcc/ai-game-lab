"""Descriptive exact-feature overlap, using the retained local decision ledger."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
from .analyze_existing import aggregate
from learning.bc_crop.agent import sha
from learning.bc_crop.experiment import save


def run(out):
    destination = out/'overlap-diagnostics.json'
    if destination.exists():
        raise FileExistsError('Preserve existing diagnostic results')
    source = out/'diagnostic-decisions-private.jsonl.gz'
    with gzip.open(source, 'rt', encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream]
    train = [r for r in rows if r['split']=='train']
    seen = {r['feature_sha256'] for r in train}
    groups = {}
    for split in ('validation', 'student_development'):
        part = [r for r in rows if r['split']==split]
        groups[split] = {label: aggregate([r for r in part if (r['feature_sha256'] in seen)==is_seen])
                         for label,is_seen in [('seen_exactly_in_training', True),('not_seen_exactly_in_training', False)]}
    repetition = Counter(r['feature_sha256'] for r in train)
    result = dict(passed=True, source_ledger_sha256=sha(source), training_calls=len(train),
                  training_unique_feature_vectors=len(seen),
                  training_calls_per_exact_vector_histogram=dict(sorted(Counter(repetition.values()).items())),
                  groups=groups,
                  interpretation='Exact observed vector recurrence across whole-seed splits; not metadata leakage or proof of causal economic harm. Novel vectors may still be close to training; seen vectors may have different unencoded world states.',
                  new_games=0, training_runs=0)
    save(destination, result)
    print(json.dumps({split:{label:{'calls':v['calls'],'accuracy':v['accuracy']} for label,v in parts.items()} for split,parts in groups.items()}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args().out.resolve())
