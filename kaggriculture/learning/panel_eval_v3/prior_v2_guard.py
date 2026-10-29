"""Bind the aborted v2 panel's reserved roles to its public failure evidence.

Callers must also confirm the returned roles are present in the preselection
snapshot and each dynamic rescan before using new roles.  Values stay private.
"""
from . import failed_history
from learning.panel_safety_v1.seed_inventory import SeedInventory

V2_PREREGISTRATION_COMMIT = 'e2d96abe357dd2a59f9cc23a23953bca8e967e7c'
V2_FAILURE_CERTIFICATE_COMMIT = 'd66411872751ffc06adca8c671501645b2190101'


def require_v2_consumed_chain(descendant_commit):
    """Return reserved roles only after committed and local evidence match."""
    return failed_history.load_failed_v2(V2_FAILURE_CERTIFICATE_COMMIT,descendant_commit)


def require_excluded(prior, inventory):
    """Fail if a historical snapshot omitted any reserved v2 role."""
    if (not isinstance(prior,SeedInventory) or prior.count != 4
        or not isinstance(inventory,SeedInventory)
        or any(not inventory.contains(value) for value in prior.private_values())):
        raise ValueError('Prior v2 reserved roles missing from exclusion inventory')
