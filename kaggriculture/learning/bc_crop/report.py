"""CLI for the immutable predeclared analysis function.

The frozen script wrote both reports, then failed its final print because its
json import was missing. Supply that standard-library name at the caller,
preserving every byte of the frozen calculations and identity checks.
"""
import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from . import analyze
from .experiment import read, save


def interpretation(result, cfg):
    """Keep frozen metrics, but never count historical reuse as new evidence."""
    if not any(key in cfg for key in ('preparation_schema', 'evaluation_role', 'new_independent_evidence')):
        return result  # The original frozen registration keeps its exact schema.
    role = cfg.get('evaluation_role')
    independent = cfg.get('new_independent_evidence')
    if cfg.get('preparation_schema') != 2 or role not in ('new-holdout', 'historical-replay'):
        raise ValueError('Unsupported preparation interpretation')
    if type(independent) is not bool or independent != (role == 'new-holdout'):
        raise ValueError('Inconsistent independent-evidence declaration')
    result = dict(result, evaluation_role=role, new_independent_evidence=independent)
    if not independent:
        result['historical_imitation_metric_gate_passed'] = result['imitation_B_passed']
        result['imitation_B_passed'] = None
        result['historical_seed_clusters'] = result['independent_seed_clusters']
        result['independent_seed_clusters'] = 0
        result['limitations'] = [*result['limitations'], 'Historical seed reuse: no new independent test evidence; metric gate is descriptive only']
    return result


def run(out):
    cfg = read(out/'registration.json')
    # Validate role before invoking the frozen computation or writing any result.
    interpretation(dict(imitation_B_passed=False, independent_seed_clusters=0, limitations=[]), cfg)
    pending = {}
    previous_save = analyze.save
    analyze.json = json
    try:
        # The frozen routine computes unchanged statistics. Buffer its two outputs
        # until interpretation succeeds, so a historical run cannot leave behind
        # a summary or console message claiming eight NEW independent clusters.
        analyze.save = lambda path, value: pending.__setitem__(path, value)
        with redirect_stdout(io.StringIO()):
            analyze.run(out)
    finally:
        analyze.save = previous_save
    result = interpretation(pending[out/'summary.json'], cfg)
    pending[out/'summary.json'] = result
    for path, value in pending.items():
        save(path, value)
    print(json.dumps(dict(engineering_A_passed=result['engineering_A_passed'],
                         imitation_B_passed=result['imitation_B_passed'],
                         evaluation_role=result.get('evaluation_role', 'original-frozen-evaluation'),
                         new_independent_evidence=result.get('new_independent_evidence', True))), flush=True)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    run(args.out.resolve())
