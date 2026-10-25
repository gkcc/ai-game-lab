"""Eight registered in-memory JSON regressions; no dataset or model I/O.

Call run_regressions only after the new stage is frozen and locally committed.
These cases construct scalar/container values only; no inputs or fixture files
are created, and no compact records are read or evaluated.
"""
import json
import time

from .reporting import canonical_json_value, p

SCHEMA = 'task-stratified-report-repair-regressions-v1'
CASE_NAMES = (
    'integer_histogram_json_roundtrip',
    'existing_string_keys_stable',
    'changed_number_and_numeric_type_rejected',
    'changed_list_order_rejected',
    'missing_field_rejected',
    'normalized_key_collision_rejected',
    'boolean_integer_and_invalid_key_types_rejected',
    'nonfinite_and_non_json_values_rejected',
)


def _reject(value):
    try:
        canonical_json_value(value)
    except ValueError:
        return
    raise AssertionError('Strict JSON normalization accepted an unsupported value')


def run_regressions(deadline=None):
    """Return only aggregate safe evidence; the runner owns output and limits."""
    started = time.monotonic()
    cases = []

    def passed(name):
        p.check_deadline(deadline)
        p.require(name == CASE_NAMES[len(cases)], 'Regression case order changed')
        cases.append(dict(case=name, passed=True))

    p.check_deadline(deadline)
    # Real failure shape: SummaryAccumulator emits integer histogram keys;
    # frozen JSON reads them back as strings, changing sort_keys order.
    histogram = dict(candidate_count_calls={2: 4, 10: 3, 100: 1},
                     range_exceedance_calls_by_feature={1: 2, 12: 1})
    roundtrip = json.loads(p.canonical(histogram))
    p.require(p.canonical(histogram) != p.canonical(roundtrip),
              'Legacy integer/string sort-order counterexample was not reproduced')
    p.require(canonical_json_value(histogram) == canonical_json_value(roundtrip),
              'Allowed integer-key JSON roundtrip still differs')
    passed(CASE_NAMES[0])

    existing = {'10': [None, False, True, 0, 1.0, 'value'], '2': {'a': -0.0}}
    p.require(canonical_json_value(existing) == p.canonical(existing),
              'Already-string-key JSON representation changed')
    passed(CASE_NAMES[1])

    p.require(canonical_json_value({'n': 1}) != canonical_json_value({'n': 2}) and
              canonical_json_value({'n': 1}) != canonical_json_value({'n': 1.0}),
              'Changed number or numeric scalar type was hidden')
    passed(CASE_NAMES[2])

    p.require(canonical_json_value({'a': [1, 2]}) != canonical_json_value({'a': [2, 1]}),
              'JSON list order was discarded')
    passed(CASE_NAMES[3])

    p.require(canonical_json_value({'a': 1, 'b': None}) != canonical_json_value({'a': 1}),
              'A missing field was discarded')
    passed(CASE_NAMES[4])

    _reject({1: 'integer', '1': 'string'})
    passed(CASE_NAMES[5])

    p.require(canonical_json_value({'n': True}) != canonical_json_value({'n': 1}) and
              canonical_json_value({'n': False}) != canonical_json_value({'n': 0}),
              'Boolean and integer scalar types were merged')
    for key in (True, False, 1.0, None):
        p.check_deadline(deadline)
        _reject({key: 0})
    passed(CASE_NAMES[6])

    for invalid in (float('nan'), float('inf'), float('-inf'), (1, 2), {1, 2}):
        p.check_deadline(deadline)
        _reject({'nested': [invalid]})
    passed(CASE_NAMES[7])

    p.require(len(cases) == len(CASE_NAMES) == 8, 'Regression matrix incomplete')
    return dict(schema=SCHEMA, passed=True, cases=cases, case_count=len(cases),
        in_memory_only=True, public_safe=True, source_stage_acceptance=False,
        model_calls=0, evaluated_rows=0, evaluated_candidates=0,
        derived_input_decodes=0, original_streams_read=0, compact_records_read=0,
        fixture_files_created=0, training_runs=0, new_games=0, official_transitions=0,
        elapsed_seconds=time.monotonic() - started)
