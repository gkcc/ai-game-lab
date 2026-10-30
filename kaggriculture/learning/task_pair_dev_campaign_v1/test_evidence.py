"""Synthetic receipt-chain regression; no private files or game execution."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import acceptance, driver, evidence
from .manifest import build_manifest
from .test_projection import SEEDS, fixture


def encoded(value):
    return json.dumps(value, separators=(',', ':'), allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def synthetic_chain(*, swap_comparisons=False, swap_coordinates=False):
    manifest = build_manifest(seeds=SEEDS,
        models={name: {'path': 'artifacts/example/' + name + '.npz', 'sha256': digit * 64}
                for name, digit in (('baseline', '1'), ('control', '2'), ('auxiliary', '3'))},
        crop={'path': 'artifacts/example/crop.npz', 'sha256': '4' * 64},
        opponents={'market': {'path': 'artifacts/example/market.py', 'sha256': '5' * 64},
                   'round2': {'path': 'artifacts/example/round2.py', 'sha256': '6' * 64}},
        worker_sha256='7' * 64)
    stage = Path('synthetic-stage')
    campaign, rows, comparisons = fixture()
    row_by_game = {row['game']: row for row in rows}
    for row in rows:
        row['replay_sha256'] = sha(('replay:' + row['game']).encode())
        row['trace_sha256'] = sha(('trace:' + row['game']).encode())
    for item in comparisons:
        if (item['reference'], item['candidate'], item['seed_alias'], item['opponent'], item['side']) == (
                'baseline', 'control', 'S02', 'market', 0):
            item['result']['stop_reason'] = 'input_difference'
    cmp_index = {(x['reference'], x['candidate'], x['seed_alias'], x['opponent'], x['side']): x
                 for x in comparisons}
    a = ('baseline', 'control', 'S02', 'market', 0)
    b = ('baseline', 'control', 'S02', 'round2', 0)
    if swap_comparisons:
        cmp_index[a], cmp_index[b] = cmp_index[b], cmp_index[a]
    if swap_coordinates:
        cmp_index[a]['opponent'], cmp_index[b]['opponent'] = (
            cmp_index[b]['opponent'], cmp_index[a]['opponent'])

    files = {}
    bundle = {name: b'{}' for name in acceptance.FILES}
    campaign['receipts'] = []
    for plan in manifest.panel.plans():
        game = plan['game']
        supervision = encoded(dict(schema='task-pair-dev-job-supervision-v1',
                                   game=game, retry_allowed=False, supervision={}))
        input_raw = encoded({'synthetic': game})
        accepted = encoded(dict(schema='task-pair-dev-job-acceptance-v1', game=game,
            passed=True, retry_allowed=False, supervision_sha256=sha(supervision),
            job_input_sha256=sha(input_raw), receipt_sha256={k: sha(v) for k, v in bundle.items()}))
        files[stage / 'job-results' / (game + '.supervision.local.json')] = supervision
        files[stage / 'job-results' / (game + '.acceptance.local.json')] = accepted
        files[stage / 'job-inputs' / (game + '.json')] = input_raw
        campaign['receipts'].append(dict(game=game, supervision_sha256=sha(supervision),
                                         acceptance_sha256=sha(accepted)))
    campaign['comparison_receipts'] = []
    for (reference, candidate, alias, opponent, side), inner in cmp_index.items():
        left = f'dev-{alias}-{opponent}-{side}-{reference}'
        right = f'dev-{alias}-{opponent}-{side}-{candidate}'
        lrow, rrow = row_by_game[left], row_by_game[right]
        document = dict(schema='task-pair-dev-comparison-v1', reference_game=left,
            candidate_game=right, reference_replay_sha256=lrow['replay_sha256'],
            candidate_replay_sha256=rrow['replay_sha256'], reference_trace_sha256=lrow['trace_sha256'],
            candidate_trace_sha256=rrow['trace_sha256'], comparison=deepcopy(inner))
        raw = encoded(document)
        files[driver._comparison_file(stage, left, candidate)] = raw
        campaign['comparison_receipts'].append(dict(reference_game=left, candidate_game=right, sha256=sha(raw)))
    files[stage / 'campaign-result.local.json'] = encoded(campaign)
    local = SimpleNamespace(manifest=manifest, stage=stage,
                            read_bundle=lambda _plan: bundle, verify_files=lambda *_args: True)
    return manifest, local, files, row_by_game


class EvidenceTests(unittest.TestCase):
    def execute_chain(self, *, swap=False, swap_coordinates=False):
        manifest, local, files, rows = synthetic_chain(
            swap_comparisons=swap, swap_coordinates=swap_coordinates)
        with patch.object(evidence, '_read', side_effect=lambda _stage, path, _limit: files[path]), \
             patch.object(acceptance, 'accept', side_effect=lambda _panel, plan, *_args: rows[plan['game']]):
            return evidence.verify_and_project(manifest=manifest, local_driver=local, check=lambda: True)

    def test_linked_comparisons_project(self):
        public = self.execute_chain(swap=False)
        pair = next(p for p in public['paired'] if p['comparison'] == 'baseline_to_control'
                    and p['seed_alias'] == 'S02' and p['opponent'] == 'market' and p['side'] == 0)
        self.assertEqual(pair['common_history']['stop_reason'], 'input_difference')

    def test_swapped_inner_comparisons_are_refused(self):
        # Outer replay/trace links and their receipt hashes remain internally
        # valid. Swapping two inner objects within one seed keeps all 16 inner
        # coordinates and aggregated opportunity support unchanged.
        with self.assertRaises(ValueError):
            self.execute_chain(swap=True)

    def test_swapped_inner_coordinates_are_refused(self):
        # Inner payloads stay attached to their true outer replay/trace source,
        # but the two opponent labels are swapped; this would mislabel detail.
        with self.assertRaises(ValueError):
            self.execute_chain(swap_coordinates=True)


if __name__ == '__main__':
    unittest.main()
