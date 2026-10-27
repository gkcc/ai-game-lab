"""Conservative local attribution of observed unit effects to selected tasks."""

from copy import deepcopy

OPS = {"DIG", "WATER", "PLANT", "HARVEST", "FERTILIZE"}


def _inventory(resources, unit, item):
    inventories = resources["inventories"]
    inventory = inventories[unit]
    return dict(inventory).get(item, 0)


def observed_effect(op, event, day):
    """Return True only for an unambiguous, operation-specific atomic effect."""
    try:
        before, after = event["tile_before"], event["tile_after"]
        rb, ra, unit = event["before"], event["after"], event["unit"]
        if op == "DIG":
            return before is not None and before != "LOCKED" and after is None
        if op == "WATER":
            return (isinstance(before, dict) and before["kind"] == "PLANT"
                    and before["watered_today"] is False and isinstance(after, dict)
                    and after["kind"] == "PLANT" and after["crop"] == before["crop"]
                    and after["watered_today"] is True)
        if op == "PLANT":
            request = event["effective"]
            crop = request[1]
            return (before is None and isinstance(after, dict) and after["kind"] == "PLANT"
                    and after["crop"] == crop
                    and rb["seeds"].get(crop, 0) - ra["seeds"].get(crop, 0) == 1)
        if op == "HARVEST":
            if not isinstance(before, dict) or before["kind"] != "PLANT":
                return False
            crop, amount = before["crop"], before["yield_units"]
            if not isinstance(amount, (int, float)) or amount <= 0:
                return False
            tile_ok = (after is None or (isinstance(after, dict) and after["kind"] == "PLANT"
                       and after["crop"] == crop and after["yield_units"] == 0))
            return tile_ok and _inventory(ra, unit, crop) - _inventory(rb, unit, crop) == amount
        if op == "FERTILIZE":
            return (isinstance(day, int) and not isinstance(day, bool)
                    and isinstance(before, dict) and before["kind"] == "PLANT"
                    and isinstance(after, dict) and after["kind"] == "PLANT"
                    and after["crop"] == before["crop"]
                    and _inventory(rb, unit, "FERTILIZER") - _inventory(ra, unit, "FERTILIZER") == 1
                    and after["fertilized_until_day"] == max(before.get("fertilized_until_day", -1), day + 2))
    except (KeyError, IndexError, TypeError, ValueError):
        return False
    return False


class EpisodeTracker:
    """Feed complete per-step selection/event batches in strictly increasing step order.

    Each batch covers one player and game. Missing selections interrupt pending
    episodes. Duplicate units or malformed selections invalidate the batch and
    interrupt pending episodes; duplicate matching events cannot prove effects.
    Returned rows contain identity only, never snapshots or teacher data.
    """
    def __init__(self, game_key, side):
        self.game, self.player = game_key, side
        self.pending = {}
        self.last_step = None
        self.rows = []

    def _close(self, unit, status="unknown", reason=None):
        row = self.pending.pop(unit)
        row["status"] = status
        row["reason"] = "atomic_effect" if status == "completed_observed" else (reason or row.pop("_last_reason", "episode_ended"))
        row.pop("_last_reason", None)
        self.rows.append(row)

    def step(self, step, selections, events, day=None):
        if not isinstance(step, int) or isinstance(step, bool) or (self.last_step is not None and step <= self.last_step):
            raise ValueError("steps must be strictly increasing integers")
        if self.last_step is not None and step != self.last_step + 1:
            for unit in list(self.pending):
                self._close(unit, reason="task_interrupted_or_changed")
        self.last_step = step
        valid = isinstance(selections, list) and isinstance(events, list)
        seen = set()
        if valid:
            for s in selections:
                if (not isinstance(s, dict) or not isinstance(s.get("unit"), int) or isinstance(s.get("unit"), bool)
                    or s["unit"] < 0 or s["unit"] in seen or s.get("step") != step
                    or not isinstance(s.get("ordinal"), int) or isinstance(s.get("ordinal"), bool)
                    or not isinstance(s.get("target"), (list, tuple)) or len(s["target"]) != 2
                    or any(not isinstance(x, int) or isinstance(x, bool) for x in s["target"])
                    or not isinstance(s.get("op"), str) or not isinstance(s.get("chosen"), str)
                    or not isinstance(s.get("unit_request"), list)):
                    valid = False
                    break
                seen.add(s["unit"])
        if not valid:
            for unit in list(self.pending):
                self._close(unit, reason="task_interrupted_or_changed")
            return
        for unit in list(self.pending):
            if unit not in seen:
                self._close(unit, reason="task_interrupted_or_changed")
        for s in selections:
            unit, target, op = s["unit"], list(s["target"]), s["op"]
            old = self.pending.get(unit)
            if old and (old["target"] != target or old["op"] != op):
                self._close(unit, reason="task_interrupted_or_changed")
                old = None
            if old is None:
                self.pending[unit] = {"game": self.game, "side": self.player, "first_step": step,
                                      "first_ordinal": s["ordinal"], "unit": unit, "target": target,
                                      "op": op, "chosen": s["chosen"], "last_step": step,
                                      "had_blocked": False, "observed_steps": 0, "reason_counts": {}, "_last_reason": "episode_ended"}
            row = self.pending[unit]
            row["last_step"] = step
            row["observed_steps"] += 1
            matches = [e for e in events if isinstance(e, dict) and e.get("step") == step
                       and e.get("player") == self.player and e.get("unit") == unit
                       and e.get("phase") == "unit"]
            request = s["unit_request"]
            reason = None
            if len(matches) != 1:
                reason = "missing_or_ambiguous_event"
                e = None
            else:
                e = matches[0]
                if e.get("requested") != request:
                    reason = "request_mismatch"
                elif e.get("status") == "blocked" or ("effective" in e and e["requested"] != e["effective"]):
                    row["had_blocked"] = True
                    reason = "official_rewrite_blocked"
                elif e.get("position") != target or not request or request[0] != op:
                    reason = "move_or_not_at_target"
                elif e.get("status") == "no_effect":
                    reason = "official_no_effect"
                else:
                    reason = "missing_or_ambiguous_event"
            if e is not None and (op in OPS and e.get("status") == "executed" and e.get("position") == target
                    and request and request[0] == op and e.get("requested") == request
                    and e.get("effective") == request and observed_effect(op, e, day)):
                self._close(unit, "completed_observed")
            else:
                row["_last_reason"] = reason
                row["reason_counts"][reason] = row["reason_counts"].get(reason, 0) + 1

    def finish(self):
        for unit in list(self.pending):
            self._close(unit, reason=(self.pending[unit].get("_last_reason") or "episode_ended"))
        return deepcopy(sorted(self.rows, key=lambda r: (r["first_step"], r["first_ordinal"], r["unit"])))
