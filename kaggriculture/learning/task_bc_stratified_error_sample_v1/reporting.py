"""Conditional statistics from committed compact records, without model calls.

Only the extracted manifest, completion receipts, aggregate summaries and new
compact diagnostics are consumed. Original streams, selected input arrays,
checkpoints and support pickles are never opened by this reporting layer.
"""
from collections import Counter
from pathlib import Path
import time
from types import SimpleNamespace

from learning.task_bc_error_diagnostics_v1.numeric import SummaryAccumulator
from . import pipeline as p

SCHEMA = 'task-stratified-conditional-report-v1'
PHASES = ((0, 239), (240, 479), (480, 718))
MARGINALS = ('split', 'seed', 'phase', 'teacher_class', 'opponent', 'side')
CLASSES = p.TEACHER_CLASSES


def _checked_json(path, expected, deadline, label):
    p.require(p.sha(path, deadline) == p._hash(expected, label), label + ' changed')
    p.check_deadline(deadline)
    return p._read(path)


def _values(group, phase, teacher_class):
    return dict(split=group['split'], seed=group['seed'], phase=phase,
                teacher_class=teacher_class, opponent=group['opponent'], side=group['side'])


def _statistics(accumulator, deadline):
    """Keep original weighting/conflicts; export aggregate fields only."""
    p.check_deadline(deadline)
    full = accumulator.result()
    p.check_deadline(deadline)
    return {key: full[key] for key in (
        'raw_weighted', 'unique_weighted32', 'unique_weighted64',
        'candidate_count_calls')}


def _entry(value, source_rows, selected_rows, accumulator, deadline):
    stats = _statistics(accumulator, deadline)
    p.require(stats['raw_weighted']['denominator'] == selected_rows,
              'Conditional report denominator does not match selected membership')
    return dict(value=value, original_source_rows=source_rows,
        selected_rows=selected_rows,
        selection_fraction=selected_rows / source_rows if source_rows else None,
        statistics=stats)


def _coverage(manifest, cfg, deadline):
    """Validate all 300 game/phase/class cells, including empty cells."""
    registered = {g['game']: g for g in cfg['selected_groups']}
    p.require(len(registered) == len(manifest['groups']) == 20,
              'Coverage requires the twenty exact source populations')
    source_marginal = {field: Counter() for field in MARGINALS}
    selected_marginal = {field: Counter() for field in MARGINALS}
    source_cells, selected_cells = Counter(), Counter()
    per_game = []
    for group in manifest['groups']:
        p.check_deadline(deadline)
        game = group['game']; original = registered[game]
        p.require(all(group[field] == original[field] for field in p.IDENTITY_FIELDS),
                  'Coverage original game binding mismatch')
        expected = [(phase, op) for phase in range(3) for op in CLASSES]
        cells = group['strata']
        p.require(isinstance(cells, list) and
                  [(cell['phase_index'], cell['teacher_class']) for cell in cells] == expected,
                  'Coverage requires every ordered original stratum exactly once')
        metadata = {(phase, op): [] for phase, op in expected}
        for selected in group['selections']:
            p.check_deadline(deadline)
            metadata[(selected['phase_index'], selected['teacher_class'])].append(selected['source_row_index'])
        source_total = selected_total = 0
        game_cells = []
        for cell in cells:
            p.check_deadline(deadline)
            phase, op = cell['phase_index'], cell['teacher_class']
            population, count = cell['population'], cell['selected_count']
            p.require(type(population) is int and population >= 0 and type(count) is int and
                      count == min(2, population) and cell['phase_step_range'] == list(PHASES[phase]),
                      'Coverage population, fixed quota or phase mismatch')
            p.require(cell['source_row_indices'] == metadata[(phase, op)] and
                      count == len(metadata[(phase, op)]), 'Coverage selected identities mismatch')
            source_total += population; selected_total += count
            values = _values(group, phase, op)
            for field in MARGINALS:
                source_marginal[field][values[field]] += population
                selected_marginal[field][values[field]] += count
            key = (group['split'], phase, op)
            source_cells[key] += population; selected_cells[key] += count
            # Counts, without sparse row identities or original observations.
            game_cells.append(dict(phase_index=phase, teacher_class=op,
                                   original_source_rows=population, selected_rows=count))
        p.require(source_total == group['source_population_rows'] == original['source_rows'] and
                  selected_total == group['rows'], 'Coverage source/sample population mismatch')
        per_game.append(dict(split=group['split'], seed=group['seed'], opponent=group['opponent'],
            side=group['side'], original_source_rows=source_total,
            selected_rows=selected_total, strata=game_cells))
    p.require(sum(source_marginal['split'].values()) == cfg['source_scan_limits']['index_metadata_rows'],
              'Coverage source denominator does not match the fixed twenty games')
    return source_marginal, selected_marginal, source_cells, selected_cells, per_game


def report(out, cfg, deadline):
    """Write aggregate public statistics and at most five local witnesses.

    ``passed`` means that this report consumed exactly the committed outputs.
    It does not mean model admission, a population error estimate, or a causal
    explanation of the historical match reward gap.
    """
    started = time.monotonic(); out = Path(out).resolve()
    p.check_deadline(deadline)
    if 'sample_root' in cfg:
        p.require(Path(cfg['sample_root']).resolve() == out, 'Report sample root mismatch')
    current = dict(cfg, sample_root=str(out))
    folder = out / 'report'
    p.write_new(folder / 'started.json', dict(schema=SCHEMA,
        sample_manifest_sha256=p._hash(cfg['sample_manifest_sha256'], 'sample_manifest'),
        sample_freeze_sha256=p._hash(cfg['sample_freeze_sha256'], 'sample_freeze'),
        freeze_sha256=p._hash(cfg['freeze_sha256'], 'freeze'),
        comparison_sha256=p._hash(cfg['comparison_sha256'], 'comparison'),
        model_calls=0, derived_input_decodes=0, retries_allowed=False))
    try:
        read_context = SimpleNamespace(deadline=deadline, wall={'sample_hash': 0.},
            snapshot=lambda *args, **kwargs: p.check_deadline(deadline))
        _, manifest, binding = p._manifest(current, read_context)
        sample_freeze = _checked_json(out / 'sample-freeze.json', cfg['sample_freeze_sha256'],
                                      deadline, 'sample_freeze')
        p.require(sample_freeze['sample_manifest_sha256'] == cfg['sample_manifest_sha256'] and
                  sample_freeze['experiment_freeze_sha256'] == cfg['freeze_sha256'] and
                  sample_freeze['collection_receipts_sha256'] == manifest['collection_receipts_sha256'] and
                  sample_freeze['source_membership_fixed_before_first_model_score'] is True,
                  'Report sample membership/source freeze mismatch')
        candidate, reference = out / 'run/candidate', out / 'run/reference'
        completed = _checked_json(candidate / 'aggregate-receipt.json',
            cfg['aggregate_receipt_sha256'], deadline, 'aggregate_receipt')
        ref = _checked_json(reference / 'reference-receipt.json',
            cfg['reference_receipt_sha256'], deadline, 'reference_receipt')
        comparison = _checked_json(candidate / 'comparison.json',
            cfg['comparison_sha256'], deadline, 'comparison')
        reference_freeze = cfg.get('reference_freeze_sha256', cfg['freeze_sha256'])
        p.require(completed.get('passed') is True and completed.get('stage') == 'aggregate' and
                  ref.get('passed') is True and ref.get('stage') == 'reference' and
                  comparison.get('passed') is True and comparison.get('byte_identical_summary') is True,
                  'Report requires completed aggregate, reference and exact comparison')
        for key in ('sample_manifest_sha256', 'model_sha256'):
            p.require(completed[key] == ref[key] == comparison[key] == cfg[key],
                      'Report provenance mismatch: ' + key)
        p.require(completed['source_manifest_sha256'] == ref['source_manifest_sha256'] ==
                  manifest['source_manifest_sha256'], 'Report original source manifest mismatch')
        p.require(completed['freeze_sha256'] == comparison['freeze_sha256'] == cfg['freeze_sha256'] and
                  ref['freeze_sha256'] == comparison['reference_freeze_sha256'] == reference_freeze,
                  'Report source freeze mismatch')
        p.require(comparison['aggregate_receipt_sha256'] == cfg['aggregate_receipt_sha256'] and
                  comparison['reference_receipt_sha256'] == cfg['reference_receipt_sha256'],
                  'Report comparison does not bind the supplied completion receipts')
        for key in ('support_index_sha256', 'support_index_receipt_sha256',
                    'shard_receipt_sha256', 'shard_order'):
            p.require(completed[key] == current[key], 'Report index/shard binding mismatch: ' + key)
        summaries = _checked_json(candidate / 'summary.json', completed['summary_sha256'],
                                  deadline, 'aggregate_summary')
        _checked_json(reference / 'summary.json', ref['summary_sha256'], deadline, 'reference_summary')
        p.require((candidate / 'summary.json').read_bytes() == (reference / 'summary.json').read_bytes(),
                  'Previously compared summaries changed')

        source_marginal, selected_marginal, source_cells, selected_cells, coverage = _coverage(manifest, cfg, deadline)
        domains = dict(split=list(p.SPLITS), seed=sorted({g['seed'] for g in manifest['groups']}),
            phase=list(range(3)), teacher_class=list(CLASSES), opponent=['v1', 'market', 'round2'], side=[0, 1])
        marginal = {field: {value: SummaryAccumulator() for value in domains[field]} for field in MARGINALS}
        interaction = {(split, phase, op): SummaryAccumulator() for split in p.SPLITS
                       for phase in range(3) for op in CLASSES}
        groups = {group['game']: group for group in manifest['groups']}
        selected_lookup = {(group['game'], selected['source_row_index']): selected
                           for group in manifest['groups'] for selected in group['selections']}
        witnesses = {}; rows = candidates = 0
        for record in p._shard_records(candidate, current, manifest, binding, read_context):
            p.check_deadline(deadline)
            group = groups[record['game']]
            selected = selected_lookup[(record['game'], record['source_row_index'])]
            diagnostic = record['diagnostic']
            p.require(diagnostic['teacher_op'] == selected['teacher_class'] and
                      diagnostic['candidate_count'] == selected['candidate_count'],
                      'Conditional compact/stratum join mismatch')
            values = _values(group, selected['phase_index'], selected['teacher_class'])
            for field in MARGINALS:
                marginal[field][values[field]].add(diagnostic)
            interaction[(group['split'], selected['phase_index'], selected['teacher_class'])].add(diagnostic)
            rows += 1; candidates += diagnostic['candidate_count']
            op = selected['teacher_class']
            if diagnostic['candidate_mismatch'] and op not in witnesses:
                witnesses[op] = dict(game=group['game'], split=group['split'], seed=group['seed'],
                    opponent=group['opponent'], side=group['side'], selection=dict(selected),
                    sample_group_sha256=group['sha256'], source_sha256=group['source_sha256'],
                    model_sha256=cfg['model_sha256'], diagnostic=dict(diagnostic),
                    evidence='Saved conditional imitation disagreement; no input re-decode, rescore or counterfactual game.',
                    limitation='Teacher preference disagreement is not economic regret; support flags refer only to the selected training subset.')
        p.require(rows == comparison['exact_compact_rows'] == sum(g['rows'] for g in manifest['groups']) and
                  candidates == comparison['exact_candidates'] == sum(g['candidates'] for g in manifest['groups']),
                  'Report population differs from the committed exact comparison')
        for split in p.SPLITS:
            p.require(p.canonical(marginal['split'][split].result()) == p.canonical(summaries[split]),
                      'Report split accumulation differs from committed complete summary')

        tables = {}
        for field in MARGINALS:
            tables[field] = []
            for value in domains[field]:
                p.check_deadline(deadline)
                tables[field].append(_entry(value, source_marginal[field][value],
                    selected_marginal[field][value], marginal[field][value], deadline))
        cells = []
        for split in p.SPLITS:
            for phase in range(3):
                for op in CLASSES:
                    p.check_deadline(deadline)
                    key = (split, phase, op)
                    cell = _entry(dict(split=split, phase_index=phase, teacher_class=op),
                                  source_cells[key], selected_cells[key], interaction[key], deadline)
                    cell['phase_step_range'] = list(PHASES[phase]); cells.append(cell)

        missing = [dict(split=split, teacher_class=op)
                   for split in p.SPLITS for op in CLASSES
                   if sum(selected_cells[(split, phase, op)] for phase in range(3)) == 0]
        repetition = {split: dict(selected_rows=summaries[split]['raw_weighted']['denominator'],
            unique32=summaries[split]['unique_weighted32']['denominator'],
            unique64=summaries[split]['unique_weighted64']['denominator'],
            calls_in_duplicate_groups32=summaries[split]['unique_weighted32']['calls_in_duplicate_groups'],
            calls_in_duplicate_groups64=summaries[split]['unique_weighted64']['calls_in_duplicate_groups'])
            for split in p.SPLITS}
        scoring = {'reference': ref['counts'], 'shards': {}}
        for group in manifest['groups']:
            p.check_deadline(deadline)
            receipt = _checked_json(candidate / 'shards' / (group['game'] + '.receipt.json'),
                cfg['shard_receipt_sha256'][group['game']], deadline, 'score_receipt')
            for key in ('evaluated_groups', 'evaluated_candidates', 'actual_score_calls',
                        'actual_score_candidates', 'cache_hits', 'cache_misses'):
                scoring['shards'][key] = scoring['shards'].get(key, 0) + receipt['counts'][key]
        p.require(scoring['reference']['evaluated_groups'] == scoring['shards']['evaluated_groups'] == rows and
                  scoring['reference']['evaluated_candidates'] == scoring['shards']['evaluated_candidates'] == candidates,
                  'Scoring counters differ from the two complete sample passes')
        for count in scoring.values():
            p.require(0 <= count['actual_score_calls'] <= count['evaluated_groups'] and
                      0 <= count['actual_score_candidates'] <= count['evaluated_candidates'],
                      'Actual model calls exceed counted evaluations')
        local_witnesses = dict(schema=SCHEMA, public_safe=False,
            selection_rule='First candidate mismatch per teacher class in original manifest then physical source-row order.',
            witness_limit=5, model_sha256=cfg['model_sha256'], freeze_sha256=cfg['freeze_sha256'],
            sample_manifest_sha256=cfg['sample_manifest_sha256'],
            classes={op: witnesses.get(op) for op in CLASSES},
            input_arrays_read=0, model_calls=0)
        p.require(len(witnesses) <= 5, 'Witness limit exceeded')
        p.write_new(folder / 'witnesses.json', local_witnesses)
        limitations = [
            'Historical teacher demonstrations are already known; this is neither a new holdout nor student-visited deployment data.',
            'Class/phase stratification changes frequencies. Selected-row and numeric-unique rates describe only this fixed conditional sample, not population error or match reward.',
            'train_seen, range_exceeded, discrete_unseen and related support fields refer only to the selected train subset. Absence does not establish absence from full training; training self-membership is a construction fact.',
            'Unique32/64 weighting averages every call within each numeric group and preserves conflicting labels. Numeric uniqueness and conditional errors do not establish harmful distribution shift.',
            'Local witnesses locate imitation disagreements from saved outputs. No alternative action was executed, and logit gaps or teacher preferences are not cash regret.',
            'No repetition-rate decision threshold was preregistered. Actual unique and duplicate counts are reported without inventing a sufficient-coverage cutoff.',
            'Missing strata remain empty; no refill, replacement sample, new scoring pass, training or game is justified by this report.',
        ]
        result = dict(schema=SCHEMA, passed=True, report_integrity_passed=True, sample_only=True,
            historical_reuse_not_new_holdout=True, source_manifest_sha256=manifest['source_manifest_sha256'],
            sample_manifest_sha256=cfg['sample_manifest_sha256'], model_sha256=cfg['model_sha256'],
            sample_freeze_sha256=cfg['sample_freeze_sha256'],
            freeze_sha256=cfg['freeze_sha256'], aggregate_receipt_sha256=cfg['aggregate_receipt_sha256'],
            reference_receipt_sha256=cfg['reference_receipt_sha256'], comparison_sha256=cfg['comparison_sha256'],
            original_source_rows=sum(source_marginal['split'].values()), selected_rows=rows,
            selected_candidates=candidates, coverage=coverage, marginals=tables,
            phase_teacher_class_by_split=cells, repetition=repetition,
            diagnostic_status='inconclusive' if missing or not witnesses else 'local_conditional_counterexamples_only',
            missing_teacher_classes_by_split=missing,
            economic_cause_conclusion='inconclusive; no counterfactual games or new policy outcomes',
            scoring_counters=scoring,
            cache_scope='One cache across reference scoring; one new cache per candidate game shard. Counts are not a matched speed benchmark.',
            witness_count=len(witnesses), witness_present_by_class={op: op in witnesses for op in CLASSES},
            local_witnesses_sha256=p.sha(folder / 'witnesses.json', deadline),
            limitations=limitations, compact_records_read=rows, derived_input_decodes=0,
            original_streams_read=0, model_calls=0, policy_calls=0, training_runs=0,
            new_games=0, official_transitions=0,
            original_diagnostic_stage_acceptance=False, original_policy_admission=False)
        p.write_new(folder / 'diagnostics.json', result)
        p.check_deadline(deadline)
        receipt = dict(schema=SCHEMA, passed=True, sample_only=True, rows=rows, candidates=candidates,
            diagnostics_sha256=p.sha(folder / 'diagnostics.json', deadline),
            local_witnesses_sha256=result['local_witnesses_sha256'], witness_count=len(witnesses),
            diagnostic_status=result['diagnostic_status'], sample_manifest_sha256=cfg['sample_manifest_sha256'],
            sample_freeze_sha256=cfg['sample_freeze_sha256'],
            model_sha256=cfg['model_sha256'], freeze_sha256=cfg['freeze_sha256'],
            comparison_sha256=cfg['comparison_sha256'], elapsed_seconds=time.monotonic() - started,
            model_calls=0, derived_input_decodes=0, original_streams_read=0, compact_records_read=rows,
            training_runs=0, new_games=0, official_transitions=0, policy_calls=0)
        p.write_new(folder / 'receipt.json', receipt)
        return receipt
    except BaseException as error:
        p.write_new(folder / 'failure.json', dict(schema=SCHEMA, passed=False,
            error_type=type(error).__name__, retry_allowed=False,
            model_calls=0, derived_input_decodes=0, original_streams_read=0))
        raise
