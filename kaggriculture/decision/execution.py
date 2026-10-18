"""Privileged offline execution tracing of pinned official functions.

Calls original functions exactly once, forwards arguments/returns unchanged and
restores hooks even on failure. No alternate transition rules or deployment
features. Not thread safe: overlapping trace scopes are rejected. Market labels
are order aggregates; fills retain an ordered digest, not a joint legality mask.
"""
from collections import Counter, deque
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from threading import Lock

_LOCK = Lock()


def resources(farm, private):
    return {"money": farm["money"], "seeds": dict(private["seeds"]), "shed": dict(private["shed"]),
            "inventories": [list(i.items()) for i in private["inventories"]],
            "positions": [list(farm["farmer"]), *[list(p) for p in farm["hands"]]],
            "quadrants": list(farm["unlocked_quadrants"]), "hires_today": farm["hires_today"]}


class ExecutionTrace:
    def __init__(self, rules, sink=None):
        self.rules, self.sink = rules, sink
        self.records = []
        self.counts = Counter()
        self.plants = Counter()
        self.effect_hash = hashlib.sha256()
        self.state = None
        self.sequence = 0

    def emit(self, record):
        self.records.append(record)
        self.counts[record["phase"] + ":" + record["status"]] += 1
        if self.sink is not None:
            self.sink(record)

    def event(self, value):
        self.sequence += 1
        self.effect_hash.update(json.dumps([self.sequence, value], sort_keys=True, separators=(",", ":")).encode() + b"\n")

    def player(self, farm):
        return next(p for p, f in enumerate(self.state[0].observation.farms) if f is farm)

    def unit(self, original, farm, private, idx, effective, *args, **kwargs):
        p = self.player(farm)
        a = self.state[p].action
        actions = [a.get("farmer", ["PASS"]), *a.get("hands", [])]
        requested = deepcopy(actions[idx])
        before = resources(farm, private)
        pos = self.rules._farmer_position(farm, idx)
        pos = list(pos) if pos is not None else None
        tile = deepcopy(farm["tiles"][pos[1]][pos[0]]) if pos is not None else None
        record = {"step": self.step, "player": p, "phase": "unit", "unit": idx,
                  "requested": requested, "effective": deepcopy(effective), "position": pos,
                  "before": before, "tile_before": tile}
        try:
            result = original(farm, private, idx, effective, *args, **kwargs)
        except Exception:
            record["status"] = "error"
            self.emit(record)
            raise
        after = resources(farm, private)
        tile_after = deepcopy(farm["tiles"][pos[1]][pos[0]]) if pos is not None else None
        changed = before != after or tile != tile_after
        status = "blocked" if requested != effective else "executed" if changed else "no_effect"
        if requested == effective == ["PASS"]:
            status = "pass"
        record.update(after=after, tile_after=tile_after, status=status,
                      reason="official_atomic_rewrite" if requested != effective else "official_effect" if changed else "official_no_effect")
        # These are observed effects, not an assumption that every request wins.
        if (isinstance(effective, list) and len(effective) >= 2 and effective[0] == "PLANT"
                and status == "executed" and isinstance(tile_after, dict)
                and tile_after.get("crop") == effective[1]
                and before["seeds"].get(effective[1], 0) - after["seeds"].get(effective[1], 0) == 1):
            record["successful_plant"] = True
            self.plants[(p, *pos)] += 1
        # Explicit quantities on transfer operations can execute partially.
        if changed and effective[0] in ("PICKUP", "PLACE") and len(effective) >= 2:
            item = effective[1]
            inv_before = dict(before["inventories"][idx]).get(item, 0)
            inv_after = dict(after["inventories"][idx]).get(item, 0)
            moved = abs(inv_after - inv_before)
            record["transferred_units"] = moved
            if effective[0] == "PLACE" and item in self.rules.ANIMALS and isinstance(tile_after, dict) and tile_after.get("animal") == item:
                wanted = 1  # Official animal placement ignores quantity.
            else:
                wanted = int(effective[2]) if len(effective) >= 3 else 1
            if 0 < moved < wanted:
                record["status"] = "partial"
        if changed and effective[0] == "DROP":
            carried = sum(dict(before["inventories"][idx]).values())
            stored = sum(after["shed"].values()) - sum(before["shed"].values())
            record.update(transferred_units=stored, discarded_units=carried-stored)
            if stored < carried:
                record["status"] = "partial"
        self.event([p, "unit", idx, requested, effective, record["status"]])
        self.emit(record)
        return result

    def market(self, original, state, env):
        queues = []
        self.current = {}
        self.orders = []
        limit = max(1, int(env.configuration.maxMarketOrdersPerTurn))
        for p, s in enumerate(state):
            orders = s.action.get("market", [])
            if not isinstance(orders, list):
                queues.append([])
                continue
            queues.append(orders[:limit])
            for i, raw in enumerate(orders):
                if i >= limit:
                    self.emit({"step": self.step, "player": p, "phase": "market", "order": i,
                               "requested": deepcopy(raw), "status": "ignored", "reason": "official_order_limit"})
        self.parse_sequence = deque((p, i) for i in range(max(map(len, queues), default=0))
                                    for p, q in enumerate(queues) if i < len(q))
        result = original(state, env)
        assert not self.parse_sequence
        for p in self.current:
            self.finish_order(p)
        for rec in self.orders:
            n, wanted = rec["executed_units"], rec["requested_units"]
            rec["status"] = "executed" if n == wanted and n > 0 else "partial" if n > 0 else "no_effect"
            rec["reason"] = "official_commit_outcome" if not rec.get("invalid") else "official_parser_rejected"
            self.emit(rec)
        return result

    def parse(self, original, raw):
        parsed = original(raw)
        p, i = self.parse_sequence.popleft()
        self.finish_order(p)
        rec = {"step": self.step, "player": p, "phase": "market", "order": i,
               "requested": deepcopy(raw), "requested_units": parsed.get("remaining", 1) if parsed else 0,
               "executed_units": 0, "cash_delta": 0, "failed_commits": 0,
               "invalid": parsed is None,
               "parse_context": resources(self.state[0].observation.farms[p], self.state[p].observation.private)}
        self.current[p] = rec
        self.orders.append(rec)
        return parsed

    def finish_order(self, p):
        if p in self.current:
            self.current[p]["after"] = resources(self.state[0].observation.farms[p], self.state[p].observation.private)

    def commit(self, original, op, item, price, farm, private, market, *args, **kwargs):
        p = self.player(farm)
        rec = self.current[p]
        if "before" not in rec:
            rec["before"] = resources(farm, private)
            rec["first_market_inventory"] = dict(market["inventory"])
        bank = farm["money"]
        result = original(op, item, price, farm, private, market, *args, **kwargs)
        rec["executed_units"] += bool(result)
        rec["failed_commits"] += not result
        rec["cash_delta"] += farm["money"] - bank
        rec["first_quote"] = rec.get("first_quote", price)
        rec["last_quote"] = price
        self.event([p, "market_fill", rec["order"], op, item, price, bool(result)])
        return result

    def atomic(self, original, farm, *args, **kwargs):
        p = self.player(farm)
        rec = self.current[p]
        private = self.state[p].observation.private
        rec["before"] = resources(farm, private)
        result = original(farm, *args, **kwargs)
        rec["after"] = resources(farm, private)
        rec["executed_units"] = int(rec["before"] != rec["after"])
        rec["cash_delta"] = rec["after"]["money"] - rec["before"]["money"]
        self.event([p, "market_atomic", rec["order"], rec["requested"], rec["executed_units"]])
        return result

    @contextmanager
    def installed(self, env):
        if not _LOCK.acquire(blocking=False):
            raise RuntimeError("Execution tracing requires exclusive single-threaded official-engine use")
        names = ("_apply_unit_action", "_process_market", "_parse_order", "_commit_unit", "_do_hire", "_do_buy_land")
        old = {n: getattr(self.rules, n) for n in names}
        interpreter = env.interpreter
        def interpret(state, engine):
            self.state = state
            self.step = state[0].observation.step
            self.records = []
            return interpreter(state, engine)
        wrappers = {"_apply_unit_action": self.unit, "_process_market": self.market, "_parse_order": self.parse,
                    "_commit_unit": self.commit, "_do_hire": self.atomic, "_do_buy_land": self.atomic}
        try:
            for name in names:
                def wrapped(*args, _fn=old[name], _hook=wrappers[name], **kwargs):
                    return _hook(_fn, *args, **kwargs)
                setattr(self.rules, name, wrapped)
            env.interpreter = interpret
            yield self
        finally:
            env.interpreter = interpreter
            for name, fn in old.items():
                setattr(self.rules, name, fn)
            _LOCK.release()
