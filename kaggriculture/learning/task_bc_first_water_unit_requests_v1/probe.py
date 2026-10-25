"""One bounded historical first-WATER intervention, never an evaluator.

Only run_primary performs work.  The caller owns the original primary deadline,
process watchdog, frozen-source check, and independent receipt checker.  No
source, model, synthetic check, or controller is executed at import time.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
import traceback

from learning.task_bc_error_diagnostics_v1.shadow import ordered_hash, first_difference
from learning.task_bc_v1.deployment import JointAgent
from learning.task_choice_v1.features import candidate_id

from . import model_io, source


OPS = ("DIG", "FERTILIZE", "HARVEST", "PLANT", "WATER")
MODEL_COUNTS = (
    "task_forward_calls", "task_forward_calls_completed",
    "task_scored_candidates", "task_scored_candidates_completed",
    "crop_forward_calls", "crop_forward_calls_completed",
    "crop_logits_values", "crop_logits_values_completed",
)
CONTROLLER_COUNTS = (
    "real_calls", "real_calls_completed", "baseline_calls", "baseline_calls_completed",
    "intervention_calls", "intervention_calls_completed", "forced_choices",
    "callbacks", "callbacks_completed", "task_candidate_encodings",
    "task_candidate_encodings_completed", "generated_candidate_objects",
)
PRIMARY_BYTES_MAX = 64 * 1024
PAIR_BYTES_MAX = 256 * 1024
PAIR_TOTAL_BYTES_MAX = 1024 * 1024 - PRIMARY_BYTES_MAX


class ProbeStopped(ValueError):
    """A safe, value-free failure code; never an instruction to retry."""


def _require(condition, code):
    if not condition:
        raise ProbeStopped(code)


def _exact(left, right, code):
    _require(first_difference(left, right) is None, code)


def _semantic_crop(record):
    return {k: v for k, v in record.items() if k != "inference_seconds"}


def _semantic_trace(trace):
    # Retain every other key, its insertion order, array order, and numeric type.
    return {k: [_semantic_crop(r) for r in v] if k == "crop_records" else v
            for k, v in trace.items()}


def _context_snapshot(context):
    # Never copy locals(): it also contains a function closing over live state.
    keys = ("tasks", "choices", "units", "actions", "seeds", "needed_seeds",
            "supply", "projected_shed", "stored_fertilizer", "carried_total",
            "day", "hour", "tpd", "remaining", "today", "final_day", "last_plant_age")
    result = {key: deepcopy(context[key]) for key in keys}
    result["available"] = {"representation": "sorted_set", "units": sorted(context["available"])}
    ordered_hash(result)  # Reject any undeclared/nonfinite/non-JSON local value.
    return result


def _descriptors(inputs):
    return [[c["unit"], c["target"][0], c["target"][1], OPS.index(c["op"])]
            for c in inputs["candidates"]]


def _ids(descriptors):
    return [candidate_id(unit, [x, y], OPS[op]) for unit, x, y, op in descriptors]


def _student_choice(inputs, student):
    """Always consult the original student on this newly encoded input."""
    chosen = student(inputs)
    _require(isinstance(chosen, str) and chosen in [c["id"] for c in inputs["candidates"]],
             "student_returned_illegal_id")
    return chosen


def _forced_choice(inputs, expected_inputs, teacher_id):
    _exact(inputs, expected_inputs, "override_input_mismatch")
    _require(teacher_id in [c["id"] for c in inputs["candidates"]], "override_illegal_id")
    return teacher_id


def _vector_comparison(historical, baseline, intervention):
    changed = [i for i in range(max(len(baseline), len(intervention)))
               if i >= len(baseline) or i >= len(intervention)
               or first_difference(baseline[i], intervention[i]) is not None]
    return dict(historical_baseline_equal=first_difference(historical, baseline) is None,
                baseline_intervention_equal=first_difference(baseline, intervention) is None,
                changed_unit_indices=changed, changed_unit_count=len(changed),
                vector_lengths=dict(historical=len(historical), baseline=len(baseline),
                                    intervention=len(intervention)),
                first_difference=first_difference(baseline, intervention))


class _Meter:
    def __init__(self, budget, ledger, deadline):
        self.budget, self.ledger, self.deadline = budget, ledger, deadline
        self.call_deadline = deadline
        ledger["models"] = dict.fromkeys(MODEL_COUNTS, 0)
        ledger["models"].update(training_calls=0, numeric_cache_reuse=False)
        ledger["controller"] = dict.fromkeys(CONTROLLER_COUNTS, 0)
        ledger["controller"].update(full_teacher_policy_calls=0, official_transitions=0, new_games=0)

    def check_time(self):
        _require(time.perf_counter() < min(self.deadline, self.call_deadline), "primary_or_call_deadline")

    def reserve(self, group, **amounts):
        self.check_time()
        counts = self.ledger[group]
        limits = self.budget[group]
        special = {"callbacks": limits.get("real_calls_max", 0) * limits.get("callbacks_per_controller_call_max", 0),
                   "forced_choices": limits.get("forced_choices_exact_on_completed_stage", 0)}
        for key, value in amounts.items():
            limit = special[key] if key in special else limits[key + "_max"]
            _require(counts[key] + value <= limit, "budget_" + key)
        for key, value in amounts.items():
            counts[key] += value

    def complete(self, group, **amounts):
        for key, value in amounts.items():
            self.ledger[group][key + "_completed"] += value

    def observed_choices(self, count):
        # The controller enumerated choices before calling us.  Charge what
        # already exists, even if the subsequent encoding is refused.
        self.ledger["controller"]["generated_candidate_objects"] += count
        _require(count <= self.budget["controller"]["candidates_per_callback_max"],
                 "budget_candidates_per_callback")
        _require(self.ledger["controller"]["generated_candidate_objects"] <=
                 self.budget["controller"]["generated_candidate_objects_max"],
                 "budget_generated_candidate_objects")
        self.check_time()

    def unknown(self, reason, **details):
        self.ledger["unknown_accounting"]["failures"].append(dict(reason=reason, **details))

    def snapshot(self):
        return {"models": {k: self.ledger["models"][k] for k in MODEL_COUNTS},
                "controller": {k: self.ledger["controller"][k] for k in CONTROLLER_COUNTS}}

    def delta(self, before):
        after = self.snapshot()
        return {group: {k: v - before[group][k] for k, v in values.items()}
                for group, values in after.items()}


def _protocol_checks(budget, deadline, result):
    """Four tiny in-memory checks; no files, model, controller, or transitions."""
    limits = budget["one_off_protocol_checks"]

    def charge(events, candidates):
        _require(time.perf_counter() < deadline, "protocol_deadline")
        for key, amount, maximum in (("cases", 1, "in_memory_cases_max"),
                                    ("fake_callback_events", events, "fake_callback_events_max"),
                                    ("fake_candidates", candidates, "fake_candidates_max")):
            _require(result[key] + amount <= limits[maximum], "protocol_budget_" + key)
        result["cases"] += 1
        result["fake_callback_events"] += events
        result["fake_candidates"] += candidates

    def rejects(callback, code):
        try:
            callback()
        except ProbeStopped as error:
            _require(str(error) == code, "protocol_wrong_rejection")
        else:
            raise ProbeStopped("protocol_missing_rejection")

    charge(0, 0)
    a, b = [["WATER"], ["PASS"]], [["PASS"], ["WATER"]]
    compared = _vector_comparison(a, a, b)
    _require(not compared["baseline_intervention_equal"] and compared["changed_unit_indices"] == [0, 1],
             "protocol_unit_order")

    charge(2, 4)
    first = {"candidates": [{"id": "a"}, {"id": "b"}], "planned": [0]}
    changed = {"candidates": [{"id": "a"}, {"id": "b"}], "planned": [1]}
    rejects(lambda: _forced_choice(changed, first, "b"), "override_input_mismatch")
    rejects(lambda: _forced_choice(first, first, "old"), "override_illegal_id")

    charge(3, 2)
    seen = []
    def student(current):
        seen.append(current["candidates"][0]["id"])
        return seen[-1]
    suffix_a, suffix_b = {"candidates": [{"id": "new-a"}]}, {"candidates": [{"id": "new-b"}]}
    _require(_student_choice(suffix_a, student) == "new-a", "protocol_first_suffix")
    _require(_student_choice(suffix_b, student) == "new-b" and seen == ["new-a", "new-b"],
             "protocol_fresh_suffix")
    rejects(lambda: _student_choice(suffix_b, lambda _: "new-a"), "student_returned_illegal_id")

    charge(1, 1)
    fake_budget = deepcopy(budget)
    fake_budget["models"]["task_forward_calls_max"] = 0
    fake_ledger = {"unknown_accounting": {"planned_prefix": [], "failures": []}}
    meter = _Meter(fake_budget, fake_ledger, deadline)
    invoked = []
    def guarded_fake_score():
        meter.reserve("models", task_forward_calls=1, task_scored_candidates=1)
        invoked.append(True)
    rejects(guarded_fake_score, "budget_task_forward_calls")
    _require(not invoked and fake_ledger["models"]["task_forward_calls"] == 0,
             "protocol_scoring_after_limit")
    result["passed"] = True


class _ProbeAgent(JointAgent):
    def __init__(self, root, cfg, meter):
        self.meter, self.current = meter, None
        self.current_event = None
        models = cfg["models"]
        super().__init__(root / models["crop"]["path"], models["crop"]["sha256"],
                         root / models["task"]["path"], models["task"]["sha256"])
        self.student = self.task_selector

    def install_forward_meters(self):
        """Install only after the constructor-only model-read context closes."""
        self.original_score = self.student.score
        self.original_logits = self.crop.network.logits
        self.student.score = self._task_score
        self.crop.network.logits = self._crop_logits

    def _task_score(self, shared, candidates):
        event = self.current_event
        _require(event is not None and event["selector"] == "student"
                 and event["task_forward_calls"] == 0, "unexpected_task_forward")
        count = len(candidates)
        _require(count == event["candidate_count"], "task_forward_candidate_count")
        self.meter.reserve("models", task_forward_calls=1, task_scored_candidates=count)
        event.update(task_forward_calls=1, task_scored_candidates=count,
                     score_sequence=self.meter.ledger["models"]["task_forward_calls"])
        try:
            values = self.original_score(shared, candidates)
        except BaseException:
            self.meter.unknown("task_forward_partial_work", sequence=event["score_sequence"], candidates=count)
            raise
        self.meter.complete("models", task_forward_calls=1, task_scored_candidates=count)
        event.update(task_forward_calls_completed=1, task_scored_candidates_completed=count)
        _require(values.shape == (count,) and str(values.dtype) == "float64", "task_forward_output_shape_or_dtype")
        self.meter.check_time()
        return values

    def _crop_logits(self, x, names):
        _require(self.current is not None, "crop_forward_outside_call")
        self.meter.reserve("models", crop_forward_calls=1, crop_logits_values=5)
        event = dict(index=len(self.current["result"]["crop_events"]),
                     sequence=self.meter.ledger["models"]["crop_forward_calls"],
                     forward_calls=1, forward_calls_completed=0, logits_values=5,
                     logits_values_completed=0, record_sha256=None, status="started")
        self.current["result"]["crop_events"].append(event)
        try:
            values = self.original_logits(x, names)
        except BaseException:
            self.meter.unknown("crop_forward_partial_work", sequence=event["sequence"], reserved_values=5)
            raise
        actual_values = int(values.size)
        self.meter.complete("models", crop_forward_calls=1, crop_logits_values=actual_values)
        event.update(forward_calls_completed=1, logits_values_completed=actual_values, status="complete")
        _require(values.shape == (5,) and str(values.dtype) == "float64", "crop_forward_output_shape_or_dtype")
        self.meter.check_time()
        return values

    def select(self, context):
        _require(self.current is not None and self.task_selector is self.student, "callback_state_not_reset")
        state, meter = self.current, self.meter
        index = len(state["result"]["events"])
        # This callback already occurred; record it even if a guard below stops
        # before any feature encoding or numerical forward may start.
        meter.ledger["controller"]["callbacks"] += 1
        event = dict(index=index, status="started", inputs_sha256=None, context_sha256=None,
                     candidate_ids_sha256=None, legal_candidates=[], candidate_count=len(context["choices"]),
                     chosen_id=None, teacher_id=None, selector="student", record_sha256=None,
                     callback_sequence=meter.ledger["controller"]["callbacks"], encoding_sequence=None,
                     score_sequence=None, encoding_calls_reserved=0, encoding_calls_completed=0,
                     task_forward_calls=0, task_forward_calls_completed=0,
                     task_scored_candidates=0, task_scored_candidates_completed=0)
        state["result"]["events"].append(event)
        meter.observed_choices(len(context["choices"]))
        _require(meter.ledger["controller"]["callbacks"] <=
                 meter.budget["controller"]["real_calls_max"] *
                 meter.budget["controller"]["callbacks_per_controller_call_max"], "budget_callbacks")
        _require(index < meter.budget["controller"]["callbacks_per_controller_call_max"],
                 "budget_callbacks_per_controller_call")
        snapshot = _context_snapshot(context)
        event["context_sha256"] = ordered_hash(snapshot)
        target = state["sample"]["first_callback_index_0_based"]
        is_override = state["mode"] == "intervention" and index == target
        if state["mode"] == "baseline" and index <= target:
            state["contexts"].append(snapshot)
        elif state["mode"] == "intervention" and index <= target:
            _exact(snapshot, state["contexts"][index], "intervention_context_mismatch")
        teacher_choice = max(context["choices"])
        unit, task_index = teacher_choice[-2:]
        target_xy, action, _priority = context["tasks"][task_index]
        live_teacher_id = candidate_id(unit, target_xy, action[0])
        event["teacher_id"] = live_teacher_id
        event["selector"] = "override" if is_override else "student"
        meter.reserve("controller", task_candidate_encodings=1)
        event.update(encoding_calls_reserved=1,
                     encoding_sequence=meter.ledger["controller"]["task_candidate_encodings"])
        original_student = self.task_selector

        def select_current(inputs):
            meter.complete("controller", task_candidate_encodings=1)
            event["encoding_calls_completed"] = 1
            event["inputs_sha256"] = ordered_hash(inputs)
            descriptors = _descriptors(inputs)
            ids = _ids(descriptors)
            _exact(ids, [c["id"] for c in inputs["candidates"]], "encoded_candidate_identity")
            _require(len(ids) == event["candidate_count"] and len(ids) == len(set(ids)), "encoded_candidate_count")
            _require(live_teacher_id in ids, "current_teacher_illegal")
            event.update(legal_candidates=descriptors, candidate_ids_sha256=ordered_hash(ids))
            aligned = state["mode"] == "baseline" or index <= target
            old = None
            if aligned:
                _require(index < len(state["document"]["records"]), "unexpected_baseline_callback")
                old = state["document"]["records"][index]
                _exact(inputs, old["inputs"], "recorded_input_mismatch")
                _require(live_teacher_id == old["teacher_label"], "recorded_teacher_mismatch")
                if state["mode"] == "intervention":
                    baseline_event = state["pair"]["baseline"]["events"][index]
                    _require(event["inputs_sha256"] == baseline_event["inputs_sha256"]
                             and event["context_sha256"] == baseline_event["context_sha256"],
                             "baseline_prefix_hash_mismatch")
            if is_override:
                _require(state["result"]["override_count"] == 0, "duplicate_override")
                chosen_before = next(c for c in inputs["candidates"] if c["id"] == old["chosen"])
                teacher_now = next(c for c in inputs["candidates"] if c["id"] == live_teacher_id)
                _require(chosen_before["op"] == teacher_now["op"] == "WATER"
                         and chosen_before["unit"] != teacher_now["unit"]
                         and first_difference(chosen_before["target"], teacher_now["target"]) is not None,
                         "bound_disagreement_not_water_unit_and_target")
                _require(teacher_now["unit"] in context["available"], "override_unit_unavailable")
                chosen = _forced_choice(inputs, old["inputs"], live_teacher_id)
                meter.reserve("controller", forced_choices=1)
                state["result"]["override_count"] += 1
            else:
                chosen = _student_choice(inputs, original_student)
                if aligned:
                    _require(chosen == old["chosen"], "recorded_student_choice_mismatch")
                    if index < target:
                        _require(chosen == live_teacher_id, "earlier_disagreement")
                    elif index == target:
                        _require(chosen != live_teacher_id, "missing_bound_disagreement")
            event["chosen_id"] = chosen
            meter.check_time()
            return chosen

        self.current_event = event
        self.task_selector = select_current
        try:
            result = super().select(context)  # Exactly one original encoding.
        except BaseException:
            if not event["encoding_calls_completed"]:
                meter.unknown("candidate_encoding_partial_work", callback_sequence=event["callback_sequence"],
                              observed_choices=event["candidate_count"])
            raise
        finally:
            self.task_selector = original_student
            self.current_event = None
        _require(self.records[-1]["chosen"] == event["chosen_id"]
                 and self.records[-1]["teacher_label"] == live_teacher_id, "callback_record_mismatch")
        event["status"] = "complete"
        meter.complete("controller", callbacks=1)
        meter.check_time()
        return result


def _empty_call():
    return dict(status="pending", call_index=None, unit_actions=None, returned_unit_actions=None,
                trace_sha256=None, events=[], crop_events=[], counts={}, active_seconds=0.0, override_count=0)


def _make_pair(sample, document):
    semantic = _semantic_trace(document)
    historical = dict(unit_actions=deepcopy(document["final_reservations"]["unit_actions"]),
                      trace_sha256=ordered_hash(semantic),
                      events=[dict(index=i, inputs_sha256=ordered_hash(r["inputs"]),
                                   chosen_id=r["chosen"], teacher_id=r["teacher_label"],
                                   record_sha256=ordered_hash(r)) for i, r in enumerate(document["records"])],
                      crop_events=[dict(index=i, record_sha256=ordered_hash(_semantic_crop(r)))
                                   for i, r in enumerate(document["crop_records"])])
    return dict(schema="first-water-unit-request-pair-v1", status="pending",
                sample=dict(game=sample["game"], side=sample["side"], step=sample["step"],
                            target_index=sample["first_callback_index_0_based"],
                            source_line_sha256=sample["source_line_sha256"]),
                historical=historical, baseline=_empty_call(), intervention=_empty_call(), comparison=None)


def _run_call(agent, item, mode, call_index):
    meter, pair = agent.meter, item["pair"]
    result = pair[mode]
    start, before = time.perf_counter(), meter.snapshot()
    meter.call_deadline = min(meter.deadline, start + 10.0)
    result.update(status="started", call_index=call_index)
    agent.current = dict(item, mode=mode, result=result)
    agent.current_event = None
    agent.records, agent.last = [], None
    agent.crop.records, agent.crop.last = [], None
    agent.student.last = None
    try:
        inputs = item["document"]["records"][0]["inputs"]
        observation, configuration = deepcopy(inputs["observation"]), deepcopy(inputs["configuration"])
        own = observation["farms"][observation["player"]]
        unit_count = 1 + len(own["hands"])
        _require(unit_count <= meter.budget["controller"]["units_per_observation_max"], "budget_units")
        meter.reserve("controller", **{"real_calls": 1, mode + "_calls": 1})
        returned = agent(observation, configuration)
        meter.complete("controller", **{"real_calls": 1, mode + "_calls": 1})
        meter.check_time()
        _require(agent.task_selector is agent.student, "student_not_restored")
        trace = agent.last
        _require(all(trace[k] == 0 for k in ("load_errors", "inference_errors", "fallbacks")), "runtime_error_or_fallback")
        _require(len(result["events"]) == len(trace["records"]), "callback_record_count")
        _require(len(result["crop_events"]) == len(trace["crop_records"]), "crop_record_count")
        for event, record in zip(result["events"], trace["records"]):
            event["record_sha256"] = ordered_hash(record)
        for event, record in zip(result["crop_events"], trace["crop_records"]):
            event["record_sha256"] = ordered_hash(_semantic_crop(record))
        result["trace_sha256"] = ordered_hash(_semantic_trace(trace))
        result["unit_actions"] = deepcopy(trace["final_reservations"]["unit_actions"])
        result["returned_unit_actions"] = deepcopy([returned["farmer"], *returned["hands"]])
        _exact(result["unit_actions"], result["returned_unit_actions"], "trace_returned_vector_mismatch")
        _require(len(result["unit_actions"]) == unit_count, "unit_vector_length")
        if mode == "baseline":
            _exact(_semantic_trace(item["document"]), _semantic_trace(trace), "baseline_full_trace_mismatch")
            _exact(pair["historical"]["unit_actions"], result["unit_actions"], "historical_baseline_vector_mismatch")
            _require(result["override_count"] == 0, "baseline_override")
        else:
            _require(result["override_count"] == 1, "intervention_override_count")
        meter.check_time()
        result["status"] = "complete"
    except BaseException:
        if meter.ledger["controller"]["real_calls"] - before["controller"]["real_calls"] > \
                meter.ledger["controller"]["real_calls_completed"] - before["controller"]["real_calls_completed"]:
            meter.unknown("controller_partial_work", call_index=call_index,
                          observed_callbacks=len(result["events"]))
        raise
    finally:
        result["active_seconds"] = time.perf_counter() - start
        result["counts"] = meter.delta(before)
        if result["status"] != "complete":
            result["status"] = "failed"
        agent.current, agent.current_event = None, None
        agent.task_selector = agent.student
        meter.call_deadline = meter.deadline
    _require(result["active_seconds"] <= 10.0, "call_time_limit")
    if mode == "intervention":
        _require(pair["baseline"]["active_seconds"] + result["active_seconds"] <=
                 meter.budget["per_sample_pair_seconds_max"], "pair_time_limit")


def _encoded(value):
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _write_new(path, encoded, accounting=None):
    # One unbuffered physical write, with a measured return value; never replace.
    with path.open("xb", buffering=0) as stream:
        if accounting is not None:
            accounting["opened"] = True
        try:
            written = stream.write(encoded)
        except BaseException:
            if accounting is not None:
                accounting["unreturned_partial_write_unknown"] = True
            raise
        if accounting is not None:
            accounting["bytes_returned_by_write"] = written
    _require(written == len(encoded), "receipt_short_write")
    return written


def run_primary(root: Path, out: Path, cfg: dict, deadline: float) -> dict:
    """Run once under the caller's original deadline, always attempt a receipt."""
    started = time.perf_counter()
    # The read guards install their own groups once, and reject a preexisting
    # group as a forbidden retry.  Absent groups are filled only in finalization.
    ledger = {"unknown_accounting": {"planned_prefix": [], "failures": []}}
    receipt = dict(schema="first-water-unit-request-primary-v1", status="failed", baseline_gate_passed=False,
                   completed_baselines=0, completed_interventions=0, pairs=[],
                   protocol_checks=dict(cases=0, fake_callback_events=0, fake_candidates=0,
                                        real_model_calls=0, real_controller_calls=0, official_transitions=0, passed=False),
                   ledger=ledger,
                   timing=dict(primary_started_perf=started, deadline_perf=deadline, elapsed_seconds=0.0,
                               model_load_seconds=0.0),
                   phase="initialization", failure_code=None, safe_sample_results=[],
                   output=dict(pair_files_written=0, pair_bytes_written=0, primary_receipt_bytes=0))
    items = []
    meter = _Meter({}, ledger, deadline)
    try:
        out.mkdir(parents=True, exist_ok=True)
        (out / "pairs").mkdir(exist_ok=True)
        meter.budget = cfg["budget"]
        _require(len(cfg["samples"]) == 6, "fixed_sample_count")
        _require(len({s["game"] for s in cfg["samples"]}) == 6, "duplicate_sample")
        meter.check_time()
        receipt["phase"] = "protocol_checks"
        _protocol_checks(cfg["budget"], deadline, receipt["protocol_checks"])
        receipt["phase"] = "source_prefixes"
        targets = source.read_targets(root, cfg, deadline, ledger)
        _require(len(targets) == 6, "returned_target_count")
        for index, target in enumerate(targets):
            _require(target["sample"] == cfg["samples"][index], "target_sample_order")
            items.append(dict(target, pair=_make_pair(target["sample"], target["document"]), contexts=[]))
        receipt["phase"] = "model_load"
        load_start = time.perf_counter()
        try:
            with model_io.model_reads(root, cfg, ledger,
                                      min(deadline, load_start + cfg["budget"]["model_load_phase_seconds_max"])):
                agent = _ProbeAgent(root, cfg, meter)
        finally:
            receipt["timing"]["model_load_seconds"] = time.perf_counter() - load_start
        meter.check_time()
        _require(receipt["timing"]["model_load_seconds"] <= cfg["budget"]["model_load_phase_seconds_max"],
                 "model_load_time_limit")
        agent.install_forward_meters()
        receipt["phase"] = "baselines"
        for index, item in enumerate(items):
            _run_call(agent, item, "baseline", index)
            receipt["completed_baselines"] += 1
        receipt["baseline_gate_passed"] = True
        receipt["phase"] = "interventions"
        for index, item in enumerate(items):
            _run_call(agent, item, "intervention", 6 + index)
            pair = item["pair"]
            comparison = _vector_comparison(pair["historical"]["unit_actions"],
                                            pair["baseline"]["unit_actions"], pair["intervention"]["unit_actions"])
            comparison["baseline_returned_equal"] = first_difference(
                pair["baseline"]["unit_actions"], pair["baseline"]["returned_unit_actions"]) is None
            comparison["intervention_returned_equal"] = first_difference(
                pair["intervention"]["unit_actions"], pair["intervention"]["returned_unit_actions"]) is None
            pair.update(comparison=comparison, status="complete")
            receipt["completed_interventions"] += 1
            receipt["safe_sample_results"].append(dict(game=item["sample"]["game"],
                unit_vectors_equal=comparison["baseline_intervention_equal"],
                changed_unit_count=comparison["changed_unit_count"]))
        _require(ledger["controller"]["forced_choices"] == 6, "stage_override_count")
        _require(not ledger["unknown_accounting"]["failures"], "unknown_partial_work")
        meter.check_time()
        receipt["status"], receipt["phase"] = "complete", "completed"
    except BaseException as error:
        traceback.print_exc()  # Existing worker-private stderr, never the safe receipt.
        receipt["failure_code"] = str(error) if isinstance(error, ProbeStopped) else "unexpected_" + type(error).__name__
    finally:
        ledger.setdefault("source", {})
        ledger.setdefault("model_io", {})
        # Do not reread any source/model/output.  These receipts are the checker's
        # single bounded input, including known work retained on failure.
        for item in items:
            pair = item["pair"]
            if pair["status"] != "complete":
                pair["status"] = "failed"
            write_accounting = dict(opened=False, bytes_returned_by_write=0,
                                    unreturned_partial_write_unknown=False)
            try:
                encoded = _encoded(pair)
                _require(len(encoded) <= PAIR_BYTES_MAX, "pair_receipt_size_limit")
                _require(receipt["output"]["pair_bytes_written"] + len(encoded) <= PAIR_TOTAL_BYTES_MAX,
                         "pair_receipts_total_size_limit")
                relative = "pairs/" + item["sample"]["game"] + ".json"
                written = _write_new(out / relative, encoded, write_accounting)
                receipt["output"]["pair_files_written"] += 1
                receipt["pairs"].append(dict(game=item["sample"]["game"], path=relative,
                                            sha256=hashlib.sha256(encoded).hexdigest(), bytes=written,
                                            status=pair["status"]))
            except BaseException as error:
                traceback.print_exc()  # Preserve the private cause without publishing its text.
                receipt["status"] = "failed"
                receipt["failure_code"] = str(error) if isinstance(error, ProbeStopped) else "pair_write_" + type(error).__name__
                ledger["unknown_accounting"]["failures"].append(dict(reason="pair_receipt_write_incomplete",
                    game=item["sample"]["game"], **write_accounting))
                break
            finally:
                receipt["output"]["pair_bytes_written"] += write_accounting["bytes_returned_by_write"]
        receipt["timing"]["elapsed_seconds"] = time.perf_counter() - started
        if time.perf_counter() >= deadline:
            receipt["status"], receipt["failure_code"] = "failed", "primary_deadline_at_finalization"
        encoded = _encoded(receipt)
        while receipt["output"]["primary_receipt_bytes"] != len(encoded):
            receipt["output"]["primary_receipt_bytes"] = len(encoded)
            encoded = _encoded(receipt)
        _require(len(encoded) <= PRIMARY_BYTES_MAX, "primary_receipt_size_limit")
        _write_new(out / "primary-receipt.json", encoded)
    return receipt
