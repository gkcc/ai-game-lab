# Receipt contract: freeze before execution

This stage uses only the fixed original-training case in the adopted root plan.
No model, official transition, replacement case or training is allowed. Old
source files remain byte-identical. Freeze this document with the implementation.

`source.read_case(root, cfg, deadline, ledger)` receives `fixed_case` and `budget`
from the adopted plan. Return `available`, `reason` when absent, `replay`,
`training_input`, `label`, `index`, `target_row_index`, and `certificate`.
Only private `case.json` receives the certificate and action-level evidence.
No raw observations or feature arrays are copied into that receipt.

`probe.run_primary(root, out, cfg, deadline)` writes `case.json` and safe
`primary-receipt.json`. `check.check(out, cfg, deadline)` independently reads
both once (at most 2 MiB total), then writes `check/complete.json` or
`check/failure.json` and a safe summary. Parent supplies their exact hashes.

Private schema `teacher-label-equivalence-case-v1`:

- `fixed`: `{game, side, step, callback_index_zero_based: 0}`.
- `available`, `reason`, `source_certificate`, `training_binding`.
- Available `training_binding`: `index_input_sha256`, actual saved `label`
  (`teacher_label`, `index`), `saved_input_sha256`,
  `saved_neural_projection_sha256`, `baseline_input_sha256` and
  `baseline_neural_projection_sha256`. Canonical SHA is UTF-8 sorted-map compact
  JSON without NaN or trailing newline. Neural projection includes schema
  version/hash, feature names, shared feature names/shared numeric vector and
  complete ordered candidate descriptors/x. Replay observation ordering remains
  untouched for all controllers; no cache observation drives the teacher.
- `first_candidates`: complete ordered `{id, unit, target, op}` descriptor list
  from baseline callback zero, without scores, priorities or numeric x.
- `stored_action`, `original_teacher`, `baseline` only once complete.
  Original teacher stores `action`, `call_index`, `seconds`, `completed`.
- `trials`: complete calls in baseline candidate order excluding original label.
  Each `forced_id` is unique. Stop at first full-action match or 64 trials.
- `outcome`: `equivalent_alternative`, `local_no_match`,
  `inconclusive_trial_limit`, `inconclusive_no_fixed_callback`,
  `inconclusive_source_prefix_limit`,
  `inconclusive_single_candidate`, or `failed`.
- `comparison`: `attempted`, `available_alternatives`, `exhaustive`,
  `first_full_match_trial_index` or null, `unit_only_matches_market_different`.

Each adapter call stores `call_index`, `forced_id` or null, full `action`,
`unit_vector` (farmer then hands), `events`, `first_input_sha256`,
`first_neural_projection_sha256`, `input_snapshot_sha256_before` and `_after`,
`seconds`, and `completed: true`. Snapshot SHA serializes ordered mapping pairs,
so input mutation or bag-order changes cannot hide behind sorted-key hashing.
Each call has fresh deep copies and a fresh controller instance.

Each event stores `callback_index`, global `encoding_sequence`, `input_sha256`,
`candidate_sha256`, `candidate_count`, `teacher_label`, `chosen`, `forced`,
and chosen `unit`, `target`, `op`. Only callback zero of a trial may be forced;
all other choices equal that callback's recomputed teacher label. Encodings
are fresh, contiguous and counted. Target input/candidate sequence matches the
baseline. No cached suffix or model prediction is permitted.

Full-action equality uses recursively type-sensitive JSON values. Arrays retain
order, including units and market orders; map field order alone is not an order
of economic actions. Boolean/integer and integer/float are distinct. Unit-only
equality is not a full-action positive. A positive is scoped to one state/prefix;
alternatives cannot be merged into a context-independent positive label set.

Safe schema `teacher-label-equivalence-primary-v1`: `status` complete/failed,
`case` path/SHA/bytes, fixed public identity, outcome, comparison counts without
IDs/actions, `protocol_checks`, `ledger`, timing/phase/failure type. Never include
raw traceback. Ledger separates source reads/decodes/unknowns, original teacher,
baseline and trial calls, encodings/candidates, starts/completions, and zero
model/training/game/official counts. Missing failed work is partial/unknown, not
zero. Parent retains failed logs locally and measures OUT bytes even on failure.

Unavailable fixed callback is accepted as inconclusive only if the bound index
prefix establishes the reason without substitution. Reaching the 256-row index
admission bound without overrunning it gives `inconclusive_source_prefix_limit`;
it does not prove that the fixed callback is absent. Exceeded budgets, identity
or behavior errors are failed execution, not inconclusive success.

Pure teacher crop selections still encode features. Record separate crop events
and started/completed crop callback, encoding and feature-value counters. Each
crop event has callback index, global crop encoding sequence, teacher/chosen,
feature-vector SHA/count, unit and `model_present: false`. Additional bounds are
16 crop callbacks per adapter call, 1040 crop encodings, 1024 values per vector
and 1064960 values in aggregate. Crop and task encodings are not network forwards.
Original call index is zero, baseline one, then trials two onward. All original
and adapter snapshots must match; task/crop encoding sequences separately start
at one. The source certificate and safe ledger share their completed timestamp;
the safe source ledger excludes only the private index-prefix list.

Six tiny memory checks run only within the frozen run: two source checks and
four probe checks for ordered actions/types, ordered input snapshot/mutation,
isolated first-choice override/reset, and early stopping. At most 32 synthetic
candidates; no real controllers/models or temporary files in these checks.
