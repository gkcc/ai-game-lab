"""Independent, single-use replay reviewer. Run only after explicit authorization.

Uses official reexecute and shared compact_replay/validate_trace/Meter solely for
engine execution and input validation; episode and effect logic is independent.
"""
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

from .experiment import OUT, ready, verify_commit, deadline_check
from .campaign import verify_model_seal

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def emit(path, obj):
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=True, indent=2, allow_nan=False)
        f.write('\n')

class FailureReceiptError(RuntimeError):
    def __init__(self, original, receipt):
        super().__init__('Independent failure receipt could not be written')
        self.original_error_kind = type(original).__name__
        self.receipt_error_kind = type(receipt).__name__


def create_marker(path, state):
    with path.open('x', encoding='utf-8') as stream:
        state['created'] = True
        json.dump(dict(attempt=1, retry_allowed=False), stream)
        stream.write('\n')
    state['complete'] = True


def effect(op, e, day):
    try:
        a, b = e['tile_before'], e['tile_after']
        before, after, u = e['before'], e['after'], e['unit']
        def inv(resources, item):
            return dict(resources['inventories'][u]).get(item, 0)
        def plant(x):
            return isinstance(x, dict) and x['kind'] == 'PLANT'
        if op == 'DIG':
            return a is not None and a != 'LOCKED' and b is None
        if op == 'WATER':
            return plant(a) and plant(b) and a['crop'] == b['crop'] and a['watered_today'] is False and b['watered_today'] is True
        if op == 'PLANT':
            crop = e['effective'][1]
            return a is None and plant(b) and b['crop'] == crop and before['seeds'].get(crop, 0) - after['seeds'].get(crop, 0) == 1
        if op == 'HARVEST':
            if not plant(a): return False
            crop, quantity = a['crop'], a['yield_units']
            return (isinstance(quantity, (int, float)) and quantity > 0 and
                    (b is None or (plant(b) and b['crop'] == crop and b['yield_units'] == 0)) and
                    inv(after, crop) - inv(before, crop) == quantity)
        if op == 'FERTILIZE':
            return (type(day) is int and plant(a) and plant(b) and a['crop'] == b['crop'] and
                    inv(before, 'FERTILIZER') - inv(after, 'FERTILIZER') == 1 and
                    b['fertilized_until_day'] == max(a.get('fertilized_until_day', -1), day + 2))
    except (KeyError, IndexError, TypeError, ValueError):
        return False
    return False

class IndependentEpisodes:
    def __init__(self, game, side):
        self.game, self.side, self.active, self.finished = game, side, {}, []
    def close(self, unit, status='unknown'):
        row = self.active.pop(unit); row['status'] = status; self.finished.append(row)
    def step(self, n, selections, events, day):
        seen = set()
        for s in selections:
            u = s['unit']
            if type(u) is not int or u < 0 or u in seen or s['step'] != n or type(s['ordinal']) is not int or len(s['target']) != 2 or any(type(x) is not int for x in s['target']) or not isinstance(s['op'], str) or not isinstance(s['chosen'], str) or not isinstance(s['unit_request'], list):
                raise ValueError('Invalid selection batch')
            seen.add(u)
        for u in list(self.active):
            if u not in seen: self.close(u)
        for s in selections:
            u, target, op = s['unit'], list(s['target']), s['op']
            old = self.active.get(u)
            if old and (old['target'] != target or old['op'] != op):
                self.close(u); old = None
            if old is None:
                self.active[u] = dict(game=self.game, side=self.side, first_step=n, first_ordinal=s['ordinal'], unit=u, target=target, op=op, chosen=s['chosen'], last_step=n)
            row = self.active[u]; row['last_step'] = n
            matches = [e for e in events if e.get('step') == n and e.get('player') == self.side and e.get('unit') == u and e.get('phase') == 'unit']
            req = s['unit_request']
            if len(matches) == 1:
                e = matches[0]
                if (op in {'DIG','WATER','PLANT','HARVEST','FERTILIZE'} and e.get('status') == 'executed' and e.get('position') == target and req and req[0] == op and e.get('requested') == req and e.get('effective') == req and effect(op, e, day)):
                    self.close(u, 'completed_observed')
    def finish(self):
        for u in list(self.active): self.close(u)
        return self.finished

def choose_sample(ids):
    """Select by identity only; statuses never affect sample membership."""
    if len(ids) < 20 or len(ids) != len(set(ids)):
        raise ValueError('Invalid sample identities')
    def identity_hash(x):
        return hashlib.sha256(json.dumps(list(x), ensure_ascii=True, separators=(',', ':')).encode('ascii')).digest()
    return sorted(ids, key=identity_hash)[:20]


def main():
    if not __debug__:
        raise RuntimeError("Optimized execution is not permitted")
    from learning.task_action_impact_diagnostic_v1.analysis import Meter, compact_replay, descriptor, read_json, trace_lines, validate_trace
    from learning.task_action_observation_v1.analysis import raw_path
    from learning.task_completion_bc_v1.collect import current_game, labels_file
    from decision.full_audit import reexecute
    cfg = ready(OUT)
    verify_commit(OUT)
    verify_model_seal(OUT)
    marker = OUT/'independent-attempt.local.json'
    marker_state = dict(created=False, complete=False)
    meter = None
    transitions = 0
    last_event_step = None
    replay_started = False
    try:
        create_marker(marker, marker_state)
        meter = Meter(cfg['independent_limits']['compressed_bytes'], cfg['independent_limits']['decoded_bytes'])
        seal, fit = read(OUT/'model-seal.json'), read(OUT/'fit-result.json')
        assert seal['passed'] and seal['freeze_sha256'] == digest(OUT/'freeze.json') and seal['fit_result_sha256'] == digest(OUT/'fit-result.json') and seal['fit_transport_sha256'] == digest(OUT/'fit-transport-result.json') and seal['checkpoint_sha256'] == fit['checkpoint_sha256'] == digest(OUT/'private/model-final.npz')
        games, result = read(OUT/'games-result.json'), read(OUT/'test-data-result.json')
        assert games['passed'] and games['all_owned_processes_reaped'] and games['game_ledger_sha256'] == digest(OUT/'game-ledger.json') and games['audit_ledger_sha256'] == digest(OUT/'game-audit-ledger.json')
        assert result['passed'] and result['games'] == 8 and result['official_transitions'] == 5752 and result['model_seal_sha256'] == digest(OUT/'model-seal.json') and result['games_result_sha256'] == digest(OUT/'games-result.json') and result['label_ledger_sha256'] == digest(OUT/'test-label-ledger.json')
        ledger, label_ledger = read(OUT/'game-ledger.json'), read(OUT/'test-label-ledger.json')
        expected = {(s, o, p) for s in cfg['test_seeds'] for o in cfg['opponents'] for p in cfg['seats']}
        assert len(ledger) == len(label_ledger) == 8 and {(x['seed'], x['opponent'], x['side']) for x in ledger} == expected and len({x['game'] for x in ledger}) == 8 and all(x['policy'] == 'teacher_task' and x['split'] == 'holdout' for x in ledger)
        key = f"holdout-{cfg['test_seeds'][0]}-round2-0-teacher_task"
        row, = [x for x in ledger if x['game'] == key]
        label, = [x for x in label_ledger if x['game'] == key]
        game, cert = current_game(OUT, row)
        audit_entry, = [x for x in read(OUT/'game-audit-ledger.json') if x['game'] == key]
        assert audit_entry['certificate_sha256'] == digest(OUT/'job-results'/(key+'.audit.json')) and audit_entry['replay_sha256'] == game['replay']['sha256'] and audit_entry['trace_sha256'] == game['trace']['sha256'] and audit_entry['official_transitions'] == 719
        assert label['seed'] == row['seed'] and label['opponent'] == row['opponent'] and label['side'] == row['side'] and label['replay_sha256'] == game['replay']['sha256'] and label['trace_sha256'] == game['trace']['sha256'] and label['semantic_sha256_v2'] == row['semantic_replay_sha256_v2'] and label['official_transitions'] == 719
        labels = labels_file(OUT/'private/labels'/(key+'.jsonl.gz'), label['labels_sha256'], cfg['data_limits']['episode_file_bytes'])
        assert len(labels) == label['groups']
        ids = [(r['game'], r['first_step'], r['first_ordinal']) for r in labels]
        assert len(ids) == len(set(ids)) and all(x[0] == key and type(x[1]) is int and type(x[2]) is int for x in ids) and len(ids) >= 20
        selected = choose_sample(ids)
        selected_set = set(selected)
        chosen = {i:r for i,r in zip(ids, labels) if i in selected_set}
        replay = read_json(raw_path(OUT, game['replay'], 'replay'), game['replay'], meter)
        compact = compact_replay(replay, game)
        stream = trace_lines(raw_path(OUT, game['trace'], 'trace'), game['trace'], meter)
        selections = []
        try:
            for step in range(719):
                deadline_check(); records = validate_trace(next(stream), compact, step, descriptor())
                selections.append([dict(step=step, ordinal=i, unit=r['unit'], target=r['target'], op=r['op'], chosen=r['chosen'], unit_request=r['unit_request']) for i,r in enumerate(records)])
            assert next(stream, None) is None
        finally: stream.close()
        tracker = IndependentEpisodes(key, 0)
        pos, events = 0, []
        def flush(end):
            nonlocal pos, events
            while pos < end:
                deadline_check(); tracker.step(pos, selections[pos], events, compact['days'][pos]); events = []; pos += 1
        def sink(event):
            nonlocal last_event_step
            deadline_check(); step = event['step']
            last_event_step = step
            if type(step) is not int or not pos <= step < 719: raise ValueError('Invalid event sequence')
            flush(step)
            if event.get('phase') == 'unit' and event.get('player') == 0: events.append(event)
        replay_started = True
        audit = reexecute(replay, trace=True, label_sink=sink)
        transitions = audit['official_transitions_reexecuted']
        assert audit['passed'] and transitions == 719 and audit['engine_sha256'] == cfg['engine_sha256'] and audit['semantic_sha256'] == row['semantic_replay_sha256'] and audit['semantic_sha256_v2'] == row['semantic_replay_sha256_v2'] == cert['official']['semantic_sha256_v2']
        flush(719); rebuilt = tracker.finish()
        rebuilt_map = {(r['game'],r['first_step'],r['first_ordinal']):r for r in rebuilt}
        assert len(rebuilt_map) == len(rebuilt)
        fields = ('game','side','first_step','first_ordinal','unit','target','op','chosen','last_step','status')
        matched = sum(i in rebuilt_map and all(chosen[i][k] == rebuilt_map[i][k] for k in fields) for i in selected)
        assert matched == 20 and meter.snapshot()['files_opened'] == 2 and meter.snapshot()['files_completed'] == 2
        counts = Counter((chosen[i]['op'], chosen[i]['status']) for i in selected)
        transparency = dict(counts=[dict(op=o,status=s,count=n) for (o,s),n in sorted(counts.items())], identity_list_sha256=hashlib.sha256(json.dumps(selected, ensure_ascii=True, separators=(',', ':')).encode('ascii')).hexdigest())
        verification = dict(passed=True, game=key, official_transitions=transitions, sample_count=20, matched=matched, selection_rule=cfg['independent_sample'], raw_reads=meter.snapshot(), sample_transparency=transparency, shared_methods=['reexecute','compact_replay','validate_trace','Meter'])
        emit(OUT/'independent-replay-verification.json', verification)
        names = {'freeze_sha256':'freeze.json','model_seal_sha256':'model-seal.json','games_result_sha256':'games-result.json','game_ledger_sha256':'game-ledger.json','game_audit_ledger_sha256':'game-audit-ledger.json','test_data_result_sha256':'test-data-result.json','test_label_ledger_sha256':'test-label-ledger.json','verification_sha256':'independent-replay-verification.json'}
        review = {k:digest(OUT/v) for k,v in names.items()}
        review.update(passed=True, method_sha256=digest(Path(__file__)), official_transitions=719, sample_count=20, sample_matched=20)
        emit(OUT/'independent-result-review.json', review)
    except BaseException as exc:
        if marker_state['created']:
            failure = dict(error_kind=type(exc).__name__, marker_created=True, marker_complete=marker_state['complete'], official_transitions_completed=transitions, last_event_step=last_event_step, replay_state=('completed' if transitions else 'unknown_after_start' if replay_started else 'not_started'), partial_transition_count_is_unknown=transitions == 0 and replay_started, raw_reads=meter.snapshot() if meter is not None else None, retry_allowed=False)
            try:
                emit(OUT/'independent-failure.local.json', failure)
            except BaseException as receipt_exc:
                raise FailureReceiptError(exc, receipt_exc) from exc
        raise

if __name__ == '__main__': main()
