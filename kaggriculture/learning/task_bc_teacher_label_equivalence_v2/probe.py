"""V2 integration of the repaired prerequisite with one frozen teacher context.

Pure helpers and the original four probe fixtures remain the exact V1 functions.
Only protocol wiring/partial bookkeeping and the public primary schema change.
"""
from copy import deepcopy
import hashlib
import time

from learning.task_bc_teacher_label_equivalence_v1.probe import (
    require, alive, raw_json, digest, same, snapshot, projection, descriptors,
    unit_vector, alternatives, FirstOverride, new_controller_counts, safe_ledger,
    protocol_checks as probe_protocol_checks,
)


def run_primary(root, out, cfg, deadline):
    from learning.task_bc_teacher_label_equivalence_v1 import source
    from learning.task_bc_teacher_label_protocol_repair_v1.protocol import protocol_checks as source_protocol_checks
    from learning.task_bc_teacher_label_protocol_repair_v1.experiment import check_primary as source_protocol_gate
    from learning.task_choice_v1 import agent as task_module
    from learning.bc_crop import agent as crop_module
    from decision.audit_market_teacher import observation as actual_observation
    from learning.task_bc_first_operation_unit_request_v1.experiment import write
    import numpy as np

    start = time.monotonic()
    limits, fixed = cfg["budget"], cfg["fixed_case"]
    ledger = dict(controller=new_controller_counts(),
        prohibited=dict(model_read_attempts=0, np_load_attempts=0, model_forwards=0,
                        training_runs=0, new_games=0, official_transitions=0),
        unknown_accounting=[])
    counts = ledger["controller"]
    case = dict(schema="teacher-label-equivalence-case-v1",
        fixed={k: fixed[k] for k in ("game", "side", "step", "callback_index_zero_based")},
        available=False, reason=None, source_certificate=None, training_binding=None,
        first_candidates=[], trials=[], outcome="failed", comparison=None)
    status, phase, failure = "failed", "protocol", None
    tests = dict(schema="teacher-label-equivalence-protocol-v2", passed=False,
        cases=None, synthetic_candidates=None, cases_known_complete=0,
        source=dict(started=False, completed=False, result=None, ledger={}, gate_receipt=None),
        probe=dict(started=False, completed=False, result=None, partial_consumption_unknown=False))
    original_encode, original_crop_class = task_module.encode_candidates, task_module.CropAgent
    original_crop_encode, original_network, original_load = crop_module.encode, crop_module.Network, np.load
    active = {}

    def task_encode(context):
        alive(deadline)
        number = len(context["choices"])
        require(0 < number <= limits["candidates_per_callback_max"], "Task candidate cap")
        require(counts["encodings_started"] < limits["candidate_encodings_max"], "Encoding cap")
        require(counts["encoded_candidates_started"] + number <= limits["encoded_candidate_objects_max"],
                "Encoded-candidate reservation cap")
        counts["encodings_started"] += 1
        counts["encoded_candidates_started"] += number
        result = original_encode(context)
        require(len(result[0]["candidates"]) == number, "Candidate generation count changed")
        counts["encodings_completed"] += 1
        counts["encoded_candidates_completed"] += number
        alive(deadline)
        return result

    def crop_encode(context):
        alive(deadline)
        require(counts["crop_encodings_started"] < limits["crop_encodings_max"], "Crop encoding cap")
        counts["crop_encodings_started"] += 1
        result = original_crop_encode(context)
        values = int(result[0].size)
        counts["crop_feature_values_started"] += values
        require(values <= limits["per_crop_vector_values_max"] and
                counts["crop_feature_values_started"] <= limits["crop_feature_values_max"], "Crop feature value cap")
        counts["crop_feature_values_completed"] += values
        counts["crop_encodings_completed"] += 1
        alive(deadline)
        return result

    def blocked_network(*args, **kwargs):
        ledger["prohibited"]["model_read_attempts"] += 1
        raise ValueError("Models are prohibited in the pure-teacher diagnostic")

    def blocked_load(*args, **kwargs):
        ledger["prohibited"]["np_load_attempts"] += 1
        raise ValueError("np.load is prohibited in the pure-teacher diagnostic")

    class MeteredCrop(original_crop_class):
        def select(self, context):
            alive(deadline)
            index = len(self.records)
            require(index < limits["crop_callbacks_per_adapter_call_max"], "Crop callback cap")
            require(self.mode == "teacher" and self.network is None and self.trace is None,
                    "Only untraced pure teacher crop mode is allowed")
            counts["crop_callbacks_started"] += 1
            before = counts["crop_encodings_completed"]
            result = super().select(context)
            record = self.records[-1]
            require(counts["crop_encodings_completed"] == before + 1 and record["chosen"] == record["teacher"]
                    and record["logits"] is None and record["source"] == "teacher", "Pure crop completion")
            active["crop_events"].append(dict(callback_index=index,
                encoding_sequence=counts["crop_encodings_completed"], teacher=record["teacher"],
                chosen=record["chosen"], x_sha256=digest(record["x"]), x_values=len(record["x"]),
                unit=record["unit"], model_present=False))
            counts["crop_callbacks_completed"] += 1
            alive(deadline)
            return result

    class TeacherProbe(task_module.TaskAgent):
        def __init__(self, forced):
            self.override = FirstOverride(forced)
            self.events, self.first_input = [], None
            super().__init__(task_selector=None, crop_mode="teacher")

        def select(self, context):
            alive(deadline)
            index = len(self.records)
            require(index < limits["callbacks_per_adapter_call_max"], "Task callback cap")
            counts["callbacks_started"] += 1
            before = counts["encodings_completed"]
            forced = self.override.at(index)
            self.task_selector = (lambda inputs: forced) if forced is not None else None
            try:
                result = super().select(context)
            finally:
                self.task_selector = None
            record = self.records[-1]
            inputs = record["inputs"]
            require(counts["encodings_completed"] == before + 1, "Exactly one fresh task encoding")
            if index == 0:
                self.first_input = deepcopy(inputs)
            if forced is not None:
                require(record["chosen"] == forced, "Wrong forced choice")
                counts["forced_choices"] += 1
            else:
                require(record["chosen"] == record["teacher_label"], "Suffix must be current pure teacher")
            self.events.append(dict(callback_index=index, encoding_sequence=counts["encodings_completed"],
                input_sha256=digest(inputs), candidate_sha256=digest(descriptors(inputs["candidates"])),
                candidate_count=len(inputs["candidates"]), teacher_label=record["teacher_label"],
                chosen=record["chosen"], forced=forced is not None,
                **{k: record[k] for k in ("unit", "target", "op")}))
            counts["callbacks_completed"] += 1
            alive(deadline)
            return result

    def call(kind, observation, configuration, forced=None):
        alive(deadline)
        require(counts["calls_started"] < limits["teacher_controller_calls_max"], "Teacher call cap")
        caps = dict(original=1, baseline=1, trials=limits["alternative_calls_max"])
        require(counts[kind + "_started"] < caps[kind], "Teacher call kind cap")
        index = counts["calls_started"]
        counts["calls_started"] += 1
        counts[kind + "_started"] += 1
        tick = time.monotonic()
        obs, configuration = deepcopy(observation), deepcopy(configuration)
        before = snapshot(obs, configuration)
        active["crop_events"] = []
        actor = None
        if kind == "original":
            code = (root / "bots/teacher_fertilizer_terminal_v2.py").read_bytes()
            require(hashlib.sha256(code).hexdigest() == fixed["original_teacher_sha256"], "Teacher identity")
            namespace = {"__name__": "label_equivalence_original_teacher"}
            exec(compile(code, "<frozen-original-teacher>", "exec"), namespace)
            action = namespace["agent"](obs, configuration)
        else:
            actor = TeacherProbe(forced)
            require(not actor.records and actor.last is None and not actor.crop.records
                    and actor.crop.last is None and actor.crop.network is None, "Fresh isolated controller")
            action = actor(obs, configuration)
            require(actor.task_selector is None, "Override did not restore pure teacher")
        seconds = time.monotonic() - tick
        after = snapshot(obs, configuration)
        require(before == after, "Input snapshot or inventory ordering mutated")
        require(seconds <= limits["per_controller_call_seconds_max"], "Individual controller time cap")
        vector = unit_vector(action)
        require(len(vector) <= limits["callbacks_per_adapter_call_max"], "Unit vector cap")
        alive(deadline)
        result = dict(call_index=index, forced_id=forced, action=action, unit_vector=vector,
            input_snapshot_sha256_before=before, input_snapshot_sha256_after=after,
            seconds=seconds, completed=True)
        if actor is not None:
            require(actor.last is not None and same(actor.last["final_reservations"]["unit_actions"], vector),
                    "Returned complete unit vector differs from controller")
            require(same(actor.first_input is None, not bool(actor.events)), "First callback accounting")
            result.update(events=actor.events, crop_events=deepcopy(active["crop_events"]),
                first_input_sha256=digest(actor.first_input) if actor.first_input is not None else None,
                first_neural_projection_sha256=digest(projection(actor.first_input)) if actor.first_input is not None else None)
        counts["calls_completed"] += 1
        counts[kind + "_completed"] += 1
        return result, actor

    try:
        synthetic = cfg["new_synthetic_limits"]
        first = tests["source"]
        first["started"] = True
        try:
            first["result"] = source_protocol_checks(deadline, first["ledger"])
            first["completed"] = True
        finally:
            # This count is only completed cases; bytes and failures remain in
            # the live ledger even if the function never returns a result.
            tests["cases_known_complete"] = sum(
                row.get("status") == "complete" and row.get("passed") is True
                for row in first["ledger"].get("cases", []))
        require(first["result"] == dict(schema="teacher-label-protocol-repair-v1", passed=True, scenario_count=3)
                and first["ledger"].get("passed") is True, "Repaired source protocol result")
        first["gate_receipt"] = source_protocol_gate(dict(
            passed=first["result"]["passed"], result=first["result"], ledger=first["ledger"]))
        measured = first["gate_receipt"]["observed_totals"]
        require(first["gate_receipt"]["passed"] is True
                and first["gate_receipt"]["cases"] == synthetic["source_protocol_cases"] == 3
                and tests["cases_known_complete"] == 3
                and measured["raw_bytes_returned"] <= synthetic["synthetic_raw_bytes_max"]
                and measured["returned_payload_bytes"] <= synthetic["synthetic_returned_payload_bytes_max"]
                and measured["json_decodes_completed"] == synthetic["synthetic_json_decodes"] == 2,
                "Repaired source protocol admission")
        second = tests["probe"]
        second["started"] = True
        try:
            second["result"] = probe_protocol_checks(deadline)
            second["completed"] = True
        except BaseException:
            # The unchanged four-case helper has no partial callback. Preserve
            # that limitation instead of reporting four or zero completed work.
            second["partial_consumption_unknown"] = True
            raise
        result = second["result"]
        require(result["passed"] is True and result["cases"] == synthetic["probe_protocol_cases"] == 4
                and result["synthetic_candidates"] == synthetic["synthetic_candidates_on_success"] == 6
                and all(type(result[key]) is int and result[key] == 0 for key in
                    ("real_controller_calls", "model_calls", "official_transitions", "temporary_files")),
                "Original probe protocol result")
        tests["cases_known_complete"] += result["cases"]
        tests.update(cases=first["result"]["scenario_count"] + result["cases"],
                     synthetic_candidates=result["synthetic_candidates"])
        require(tests["cases"] == tests["cases_known_complete"] == 7
                and tests["cases"] <= limits["tiny_in_memory_cases_max"]
                and tests["synthetic_candidates"] <= limits["synthetic_candidates_max"], "Protocol scope")
        tests["passed"] = True
        phase = "source"
        data = source.read_case(root, cfg, deadline, ledger)
        case.update(available=data["available"], reason=data.get("reason"), source_certificate=data["certificate"])
        if not data["available"]:
            case.update(outcome="inconclusive_source_prefix_limit" if data.get("reason") == "index_prefix_limit"
                        else "inconclusive_no_fixed_callback", comparison=dict(attempted=0,
                available_alternatives=0, exhaustive=False, first_full_match_trial_index=None,
                unit_only_matches_market_different=0))
        else:
            phase = "baseline"
            replay, saved_input, label = data["replay"], data["training_input"], data["label"]
            obs = actual_observation(replay["steps"][fixed["step"]], fixed["side"])
            configuration = replay["configuration"]
            require(obs["player"] == fixed["side"] and obs["step"] == fixed["step"], "Factual observation identity")
            case["stored_action"] = deepcopy(replay["steps"][fixed["step"] + 1][fixed["side"]]["action"])
            case["original_teacher"], _ = call("original", obs, configuration)
            require(same(case["original_teacher"]["action"], case["stored_action"]), "Original teacher must reproduce stored action")
            task_module.encode_candidates, task_module.CropAgent = task_encode, MeteredCrop
            crop_module.encode, crop_module.Network, np.load = crop_encode, blocked_network, blocked_load
            case["baseline"], baseline_actor = call("baseline", obs, configuration)
            base = case["baseline"]
            require(same(base["action"], case["stored_action"]), "Adapter must reproduce complete original action")
            require(base["events"] and baseline_actor.first_input is not None, "Saved group must have first callback")
            saved_sha = digest(saved_input)
            require(saved_sha == data["index"]["input_sha256"] == base["first_input_sha256"], "Actual saved training input binding")
            require(same(projection(saved_input), projection(baseline_actor.first_input)), "Actual neural input/order equality")
            choices = descriptors(baseline_actor.first_input["candidates"])
            require(type(label["index"]) is int and 0 <= label["index"] < len(choices)
                    and label["teacher_label"] == choices[label["index"]]["id"]
                    == base["events"][0]["teacher_label"] == base["events"][0]["chosen"], "Actual serialized single label")
            case["first_candidates"] = choices
            case["training_binding"] = dict(index_input_sha256=data["index"]["input_sha256"], label=label,
                saved_input_sha256=saved_sha, saved_neural_projection_sha256=digest(projection(saved_input)),
                baseline_input_sha256=base["first_input_sha256"],
                baseline_neural_projection_sha256=base["first_neural_projection_sha256"])
            choices_to_try = alternatives(choices, label["teacher_label"])
            match = None
            unit_only = 0
            phase = "alternatives"
            for forced in choices_to_try[:limits["alternative_calls_max"]]:
                trial, actor = call("trials", obs, configuration, forced)
                case["trials"].append(trial)
                require(trial["first_input_sha256"] == base["first_input_sha256"]
                        and trial["first_neural_projection_sha256"] == base["first_neural_projection_sha256"],
                        "Trial first context must remain factual teacher context")
                require(actor.override.used and trial["events"][0]["forced"], "Exactly one initial substitution")
                if same(trial["action"], base["action"]):
                    match = len(case["trials"]) - 1
                    break
                if same(trial["unit_vector"], base["unit_vector"]):
                    unit_only += 1
            exhaustive = len(case["trials"]) == len(choices_to_try)
            case["comparison"] = dict(attempted=len(case["trials"]), available_alternatives=len(choices_to_try),
                exhaustive=exhaustive, first_full_match_trial_index=match,
                unit_only_matches_market_different=unit_only)
            case["outcome"] = ("inconclusive_single_candidate" if not choices_to_try else
                "equivalent_alternative" if match is not None else "local_no_match" if exhaustive
                else "inconclusive_trial_limit")
        alive(deadline)
        status, phase = "complete", "complete"
    except BaseException as error:
        failure = type(error).__name__
        ledger["unknown_accounting"].append(dict(phase=phase, kind=failure,
            meaning="Failed or partial work is preserved; unfinished encoding/output work is not assumed zero."))
        raise
    finally:
        task_module.encode_candidates, task_module.CropAgent = original_encode, original_crop_class
        crop_module.encode, crop_module.Network, np.load = original_crop_encode, original_network, original_load
        ledger["patches_restored"] = True
        case_raw = raw_json(case) + b"\n"
        if len(case_raw) > limits["new_private_case_receipt_bytes_max"]:
            status = "failed"
            failure = failure or "PrivateCaseSizeLimit"
            ledger["unknown_accounting"].append(dict(phase="case_write", kind="oversize_private_case",
                                                    bytes=len(case_raw)))
            # Retain actual failure metadata; do not publish or truncate a false complete case.
            case = dict(schema=case["schema"], fixed=case["fixed"], available=False,
                        outcome="failed", original_case_bytes=len(case_raw),
                        original_case_sha256=hashlib.sha256(case_raw).hexdigest(), private_case_not_written=True)
            case_raw = raw_json(case) + b"\n"
        with (out / "case.json").open("xb", buffering=0) as stream:
            require(stream.write(case_raw) == len(case_raw), "Incomplete private case write")
        receipt = dict(schema="teacher-label-equivalence-primary-v2", status=status,
            case=dict(path="case.json", sha256=hashlib.sha256(case_raw).hexdigest(), bytes=len(case_raw)),
            fixed=case["fixed"], outcome=case["outcome"], comparison=case.get("comparison"),
            protocol_checks=tests, ledger=safe_ledger(ledger), phase=phase, failure_type=failure,
            elapsed_seconds=time.monotonic() - start)
        write(out / "primary-receipt.json", receipt)
        alive(deadline)
    require(status == "complete", "Primary stopped with preserved failure")
    return receipt
