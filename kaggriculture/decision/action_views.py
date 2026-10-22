"""Read-only action views; these do not normalize or validate official actions."""


def opcode_view(action):
    """Keep string opcodes; label missing/non-string ones without changing payloads."""
    if isinstance(action, list) and action and isinstance(action[0], str) and action[0]:
        return action[0]
    return 'invalid'


def non_sell_slots(market):
    """Retain raw non-SELL orders and their indices, including malformed orders.

    The comparison contract expects a market list, not arbitrary action objects.
    Filtering is by opcode only: this is not an official legality mask.
    """
    assert isinstance(market, list), 'comparison requires a market order list'
    return [(i, order) for i, order in enumerate(market) if opcode_view(order) != 'SELL']
