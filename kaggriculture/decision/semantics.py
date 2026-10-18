"""Versioned replay semantics. Only unit inventory mapping order is material.

Ordinary JSON object field order is immaterial. Timing is excluded in both
versions. Version 1 is retained solely to verify historical evidence.
"""
import hashlib
import json

AUDIT_VERSION = 2


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items() if k != "remainingOverageTime"}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def canonical(value, path=()):
    if isinstance(value, dict):
        # Frame -> player state -> observation -> private -> inventories -> unit.
        if path[-4:-1] == ("observation", "private", "inventories") and isinstance(path[-1], int):
            return {"__inventory_items_in_order_v2__": [[k, canonical(v, (*path, k))] for k, v in value.items()]}
        return {k: canonical(v, (*path, k)) for k, v in value.items() if k != "remainingOverageTime"}
    if isinstance(value, list):
        return [canonical(v, (*path, i)) for i, v in enumerate(value)]
    return value


def digest(value, version=AUDIT_VERSION):
    if version not in (1, 2):
        raise ValueError("Unsupported audit version")
    data = clean(value) if version == 1 else canonical(value)
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
