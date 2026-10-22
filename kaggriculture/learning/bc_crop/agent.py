"""One frozen controller, with exactly one crop-choice expression intercepted.

No imports from privileged replay/audit modules. No teacher fallback. The AST
patch is source-hash pinned, count checked, and subjected to complete-game parity.
"""
import ast
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .features import encode, schema_hash, CROPS, SCHEMA_VERSION

ROOT = Path(__file__).resolve().parents[2]
TEACHER_SHA = '0844f0152e742aba684108b5914964fb85082b3c0881d0da4d41649907a9b6ff'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Network:
    """Float64 CPU evaluation of exported float32 weights; first argmax tie."""
    def __init__(self, path, expected_sha):
        if not expected_sha or sha(path) != expected_sha:
            raise ValueError('Missing or mismatched checkpoint identity')
        self.sha = expected_sha
        with np.load(path, allow_pickle=False) as z:
            self.meta = json.loads(str(z['metadata']))
            self.mean, self.scale = z['mean'].astype('float64'), z['scale'].astype('float64')
            self.weights = [(z['w'+str(i)].astype('float64'), z['b'+str(i)].astype('float64')) for i in range(3)]
        m = self.meta
        if (m['schema_version'] != SCHEMA_VERSION or m['crops'] != list(CROPS)
                or m['teacher_sha256'] != TEACHER_SHA or m['schema_sha256'] != schema_hash(m['features'])):
            raise ValueError('Incompatible feature schema or teacher')
        dimensions = [len(m['features']), 128, 64, 5]
        if self.mean.shape != (dimensions[0],) or self.scale.shape != self.mean.shape:
            raise ValueError('Invalid normalization shape')
        if not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all() or not (self.scale > 0).all():
            raise ValueError('Invalid normalization')
        for i, (w, b) in enumerate(self.weights):
            if w.shape != (dimensions[i], dimensions[i+1]) or b.shape != (dimensions[i+1],):
                raise ValueError('Invalid layer shape')
            if not np.isfinite(w).all() or not np.isfinite(b).all():
                raise ValueError('Non-finite weights')

    def logits(self, x, names):
        if list(names) != self.meta['features']:
            raise ValueError('Runtime encoder schema mismatch')
        z = (np.asarray(x, dtype='float64') - self.mean) / self.scale
        for i, (w, b) in enumerate(self.weights):
            z = z @ w + b
            if i < 2:
                z = np.maximum(z, 0)
        if not np.isfinite(z).all():
            raise ValueError('Non-finite model output')
        return z


class CropAgent:
    def __init__(self, mode='teacher', checkpoint=None, expected_sha=None, trace=None):
        if mode not in ('teacher', 'model'):
            raise ValueError('Unknown selector')
        self.mode = mode
        self.network = Network(checkpoint, expected_sha) if mode == 'model' else None
        self.trace = Path(trace) if trace else None
        self.records, self.last = [], None
        raw = (ROOT / 'bots/teacher_fertilizer_terminal_v2.py').read_bytes()
        if hashlib.sha256(raw).hexdigest() != TEACHER_SHA:
            raise ValueError('Frozen teacher changed')
        tree = ast.parse(raw)
        class Inject(ast.NodeTransformer):
            choices = returns = 0
            def visit_Assign(patch, node):
                # The outer opportunity gate assigns best_crop, not c.
                if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                        and node.targets[0].id == 'c' and isinstance(node.value, ast.Call)
                        and isinstance(node.value.func, ast.Name) and node.value.func.id == 'max'):
                    patch.choices += 1
                    node.value = ast.parse('_bc_select(locals())', mode='eval').body
                return patch.generic_visit(node)
            def visit_Return(patch, node):
                if isinstance(node.value, ast.Dict) and [k.value for k in node.value.keys] == ['farmer', 'hands', 'market']:
                    patch.returns += 1
                    node.value = ast.Call(ast.Name('_bc_finish', ast.Load()),
                                          [ast.Call(ast.Name('locals', ast.Load()), [], []), node.value], [])
                return patch.generic_visit(node)
        patch = Inject()
        tree = patch.visit(tree)
        if (patch.choices, patch.returns) != (1, 1):
            raise ValueError('Unexpected teacher integration points')
        ns = {'__name__': 'bc_frozen_controller', '_bc_select': self.select, '_bc_finish': self.finish}
        exec(compile(ast.fix_missing_locations(tree), '<hash-pinned-controller>', 'exec'), ns)
        self.controller = ns['agent']

    def select(self, ctx):
        # Encode BEFORE computing the supervision label. Scores never enter x.
        x, names = encode(ctx)
        teacher = max(CROPS, key=lambda c: (ctx['crop_score'](c), c))
        tick = time.perf_counter()
        logits = self.network.logits(x, names) if self.network else None
        chosen = CROPS[int(np.argmax(logits))] if logits is not None else teacher
        seconds = time.perf_counter() - tick
        accepted = ctx['crop_score'](chosen) > 0
        has_seed = ctx['seeds'].get(chosen, 0) > 0
        at_target = tuple(ctx['units'][ctx['i']]) == ctx['pos']
        outcome = ('economic_reject' if not accepted else 'request_seed' if not has_seed
                   else 'plant_request' if at_target else 'move_with_seed')
        self.records.append(dict(step=ctx['observation']['step'], decision=len(self.records), unit=ctx['i'],
            target=list(ctx['pos']), x=x.tolist(), schema_sha256=schema_hash(names), teacher=teacher, chosen=chosen,
            outcome=outcome, gate_reason='teacher_crop_score_nonpositive' if not accepted else None,
            inference_seconds=seconds, logits=logits.tolist() if logits is not None else None,
            prior=dict(seeds=deepcopy(ctx['seeds']), needed_seeds=deepcopy(ctx['needed_seeds']),
                       supply=deepcopy(ctx['supply']), projected_shed=deepcopy(ctx['projected_shed']),
                       unit_actions=deepcopy(ctx['actions'])),
            selected_seed_before=ctx['seeds'].get(chosen, 0),
            source='model' if self.network else 'teacher'))
        return chosen

    def finish(self, ctx, result):
        for r in self.records:
            r['unit_request'] = deepcopy(ctx['actions'][r['unit']])
            r['buy_seed_orders'] = [deepcopy(a) for a in result['market'] if a[:2] == ['BUY_SEED', r['chosen']]]
        self.last = dict(step=ctx['observation']['step'], records=self.records,
                         final_reservations=dict(seeds=ctx['seeds'], supply=ctx['supply'], needed_seeds=ctx['needed_seeds']),
                         checkpoint_sha256=self.network.sha if self.network else None,
                         load_errors=0, inference_errors=0, fallbacks=0)
        if self.trace:
            self.trace.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(self.trace, 'at', encoding='utf-8') as f:
                f.write(json.dumps(self.last, separators=(',', ':')) + '\n')
        return result

    def __call__(self, observation, configuration):
        self.records = []  # no cross-turn, cross-seat or cross-game hidden state
        return self.controller(observation, configuration)
