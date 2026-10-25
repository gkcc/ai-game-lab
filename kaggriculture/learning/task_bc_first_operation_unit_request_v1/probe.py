"""Adapt the accepted WATER probe at its operation-specific choice boundary.

The original model I/O, numerical meters, trace comparison and call lifecycle
remain imported unchanged. Only the target predicate and prefix rule differ.
"""
from copy import deepcopy
import hashlib
import time
import traceback

from learning.task_bc_first_water_unit_requests_v1 import probe as old
from learning.task_bc_first_water_unit_requests_v1 import model_io
from . import source
from .select import validate_prefix


def factual_choice(chosen, teacher, saved, index, target):
    old._require(chosen == saved["chosen"], "recorded_student_choice_mismatch")
    if index < target:
        old._require(chosen.rsplit(":", 1)[-1] == teacher.rsplit(":", 1)[-1],
                     "earlier_operation_disagreement")
    elif index == target:
        old._require(chosen.rsplit(":", 1)[-1] != teacher.rsplit(":", 1)[-1],
                     "missing_selected_operation_disagreement")


class OperationAgent(old._ProbeAgent):
    def select(self, context):
        old._require(self.current is not None and self.task_selector is self.student,
                     "callback_state_not_reset")
        state, meter = self.current, self.meter
        index = len(state["result"]["events"])
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
        limits = meter.budget["controller"]
        old._require(meter.ledger["controller"]["callbacks"] <=
                     limits["real_calls_max"] * limits["callbacks_per_controller_call_max"], "budget_callbacks")
        old._require(index < limits["callbacks_per_controller_call_max"], "budget_callbacks_per_call")
        snapshot = old._context_snapshot(context)
        event["context_sha256"] = old.ordered_hash(snapshot)
        target = state["sample"]["first_callback_index_0_based"]
        forced = state["mode"] == "intervention" and index == target
        if state["mode"] == "baseline" and index <= target:
            state["contexts"].append(snapshot)
        elif state["mode"] == "intervention" and index <= target:
            old._exact(snapshot, state["contexts"][index], "intervention_context_mismatch")
        unit, task_index = max(context["choices"])[-2:]
        target_xy, action, _ = context["tasks"][task_index]
        teacher = old.candidate_id(unit, target_xy, action[0])
        event.update(teacher_id=teacher, selector="override" if forced else "student")
        meter.reserve("controller", task_candidate_encodings=1)
        event.update(encoding_calls_reserved=1,
                     encoding_sequence=meter.ledger["controller"]["task_candidate_encodings"])
        original_student = self.task_selector

        def choose(inputs):
            meter.complete("controller", task_candidate_encodings=1)
            event["encoding_calls_completed"] = 1
            event["inputs_sha256"] = old.ordered_hash(inputs)
            descriptors = old._descriptors(inputs)
            ids = old._ids(descriptors)
            old._exact(ids, [c["id"] for c in inputs["candidates"]], "encoded_candidate_identity")
            old._require(len(ids) == event["candidate_count"] == len(set(ids)), "encoded_candidate_count")
            old._require(teacher in ids, "current_teacher_illegal")
            event.update(legal_candidates=descriptors, candidate_ids_sha256=old.ordered_hash(ids))
            aligned = state["mode"] == "baseline" or index <= target
            saved = None
            if aligned:
                old._require(index < len(state["document"]["records"]), "unexpected_baseline_callback")
                saved = state["document"]["records"][index]
                old._exact(inputs, saved["inputs"], "recorded_input_mismatch")
                old._require(teacher == saved["teacher_label"], "recorded_teacher_mismatch")
                if state["mode"] == "intervention":
                    baseline = state["pair"]["baseline"]["events"][index]
                    old._require(event["inputs_sha256"] == baseline["inputs_sha256"] and
                                 event["context_sha256"] == baseline["context_sha256"], "baseline_prefix_hash_mismatch")
            if forced:
                old._require(state["result"]["override_count"] == 0, "duplicate_override")
                before = next(c for c in inputs["candidates"] if c["id"] == saved["chosen"])
                preferred = next(c for c in inputs["candidates"] if c["id"] == teacher)
                old._require(before["op"] != preferred["op"], "selected_operations_not_different")
                old._require(preferred["unit"] in context["available"], "override_unit_unavailable")
                chosen = old._forced_choice(inputs, saved["inputs"], teacher)
                meter.reserve("controller", forced_choices=1)
                state["result"]["override_count"] += 1
            else:
                chosen = old._student_choice(inputs, original_student)
                if aligned:
                    factual_choice(chosen, teacher, saved, index, target)
            event["chosen_id"] = chosen
            meter.check_time()
            return chosen

        self.current_event, self.task_selector = event, choose
        try:
            # Skip the old WATER-specific override, retain original TaskAgent
            # feature encoding, legal-ID handling and records exactly once.
            result = old.JointAgent.select(self, context)
        except BaseException:
            if not event["encoding_calls_completed"]:
                meter.unknown("candidate_encoding_partial_work", callback_sequence=event["callback_sequence"],
                              observed_choices=event["candidate_count"])
            raise
        finally:
            self.task_selector, self.current_event = original_student, None
        old._require(self.records[-1]["chosen"] == event["chosen_id"] and
                     self.records[-1]["teacher_label"] == teacher, "callback_record_mismatch")
        event["status"] = "complete"
        meter.complete("controller", callbacks=1)
        meter.check_time()
        return result


def protocol_checks(budget, deadline, result):
    old._protocol_checks(budget, deadline, result)
    limits = budget["one_off_protocol_checks"]
    old._require(result["cases"] + 1 <= limits["in_memory_cases_max"] and
                 result["fake_callback_events"] + 2 <= limits["fake_callback_events_max"] and
                 result["fake_candidates"] + 4 <= limits["fake_candidates_max"], "extra_protocol_budget")
    result["cases"] += 1
    result["fake_callback_events"] += 2
    result["fake_candidates"] += 4
    result["passed"] = False
    h = "a" * 64
    turns = [[0, 0, h], [1, 0, h], [2, 2, h]]
    prefix = [[2, 0, h, "WATER", "WATER"], [2, 1, h, "WATER", "PLANT"]]
    target = dict(step=2, target_index=1, source_line_sha256=h,
                  teacher={"op": "WATER"}, chosen={"op": "PLANT"})
    validate_prefix(prefix, turns, target)
    factual_choice("u0:x0:y0:WATER", "u1:x1:y1:WATER", {"chosen": "u0:x0:y0:WATER"}, 0, 1)
    factual_choice("u0:x0:y0:PLANT", "u1:x1:y1:WATER", {"chosen": "u0:x0:y0:PLANT"}, 1, 1)
    try:
        factual_choice("u0:x0:y0:PLANT", "u1:x1:y1:WATER", {"chosen": "u0:x0:y0:PLANT"}, 0, 1)
    except old.ProbeStopped as error:
        old._require(str(error) == "earlier_operation_disagreement", "wrong_prefix_rejection")
    else:
        raise old.ProbeStopped("missing_prefix_rejection")
    old._require(time.perf_counter() < deadline, "protocol_deadline")
    result["passed"] = True


def run_primary(root, out, cfg, selection, deadline):
    started = time.perf_counter()
    ledger = {"unknown_accounting": {"planned_prefix": [], "failures": []}}
    meter = old._Meter(cfg["budget"], ledger, deadline)
    result = dict(schema="first-operation-unit-request-primary-v1", status="failed",
                  baseline_gate_passed=False, completed_baselines=0, completed_interventions=0, pair=None,
                  protocol_checks=dict(cases=0, fake_callback_events=0, fake_candidates=0,
                                       real_model_calls=0, real_controller_calls=0, official_transitions=0, passed=False),
                  ledger=ledger, timing=dict(primary_started_perf=started, deadline_perf=deadline,
                                            elapsed_seconds=0.0, model_load_seconds=0.0),
                  phase="initialization", failure_code=None, safe_result=None,
                  output=dict(pair_files_written=0, pair_bytes_written=0, primary_receipt_bytes=0))
    item = None
    try:
        result["phase"] = "protocol_checks"
        protocol_checks(cfg["budget"], deadline, result["protocol_checks"])
        result["phase"] = "source_prefix"
        item = source.read_target(root, cfg, selection, deadline, ledger)
        item["pair"] = old._make_pair(item["sample"], item["document"])
        item["pair"]["schema"] = "first-operation-unit-request-pair-v1"
        item["contexts"] = []
        result["phase"] = "model_load"
        tick = time.perf_counter()
        try:
            with model_io.model_reads(root, cfg, ledger,
                                      min(deadline, tick + cfg["budget"]["model_load_phase_seconds_max"])):
                agent = OperationAgent(root, cfg, meter)
        finally:
            result["timing"]["model_load_seconds"] = time.perf_counter() - tick
        old._require(result["timing"]["model_load_seconds"] <= cfg["budget"]["model_load_phase_seconds_max"],
                     "model_load_time_limit")
        agent.install_forward_meters()
        result["phase"] = "baseline"
        old._run_call(agent, item, "baseline", 0)
        result.update(baseline_gate_passed=True, completed_baselines=1, phase="intervention")
        old._run_call(agent, item, "intervention", 1)
        result["completed_interventions"] = 1
        pair = item["pair"]
        comparison = old._vector_comparison(pair["historical"]["unit_actions"],
                                            pair["baseline"]["unit_actions"], pair["intervention"]["unit_actions"])
        comparison.update(baseline_returned_equal=True, intervention_returned_equal=True)
        pair.update(comparison=comparison, status="complete")
        result["safe_result"] = dict(game=item["sample"]["game"],
                                   unit_vectors_equal=comparison["baseline_intervention_equal"],
                                   changed_unit_count=comparison["changed_unit_count"])
        old._require(ledger["controller"]["forced_choices"] == 1 and
                     not ledger["unknown_accounting"]["failures"], "incomplete_or_unknown_work")
        meter.check_time()
        result.update(status="complete", phase="completed")
    except BaseException as error:
        traceback.print_exc()  # Existing private worker log; never publish it.
        result["failure_code"] = str(error) if isinstance(error, old.ProbeStopped) else type(error).__name__
    finally:
        ledger.setdefault("source", {})
        ledger.setdefault("model_io", {})
        if item is not None and "pair" in item:
            accounting = dict(opened=False, bytes_returned_by_write=0, unreturned_partial_write_unknown=False)
            try:
                if item["pair"]["status"] != "complete":
                    item["pair"]["status"] = "failed"
                encoded = old._encoded(item["pair"])
                old._require(len(encoded) <= cfg["scope_budget"]["outputs"]["pair_receipt_bytes_max"], "pair_size_limit")
                written = old._write_new(out / "pair.json", encoded, accounting)
                result["pair"] = dict(path="pair.json", sha256=hashlib.sha256(encoded).hexdigest(), bytes=written)
                result["output"]["pair_files_written"] = 1
            except BaseException as error:
                traceback.print_exc()
                result.update(status="failed", failure_code="pair_write_" + type(error).__name__)
                ledger["unknown_accounting"]["failures"].append(dict(reason="pair_write_incomplete", **accounting))
            finally:
                result["output"]["pair_bytes_written"] = accounting["bytes_returned_by_write"]
        result["timing"]["elapsed_seconds"] = time.perf_counter() - started
        if time.perf_counter() >= deadline:
            result.update(status="failed", failure_code="primary_final_deadline")
        encoded = old._encoded(result)
        while result["output"]["primary_receipt_bytes"] != len(encoded):
            result["output"]["primary_receipt_bytes"] = len(encoded)
            encoded = old._encoded(result)
        old._require(len(encoded) <= cfg["scope_budget"]["outputs"]["primary_receipt_bytes_max"], "primary_size_limit")
        old._write_new(out / "primary-receipt.json", encoded)
    return result
