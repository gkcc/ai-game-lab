from pathlib import Path
from decision.api import FrozenAdapter, act
_scope = {"__name__": "frozen_baseline", "__file__": str(Path(__file__).with_name("baseline.py"))}
exec(compile(Path(_scope["__file__"]).read_bytes(), _scope["__file__"], "exec"), _scope)
_policy = FrozenAdapter(_scope["agent"], buy_first_land=True)
def agent(observation, configuration):
    return act(_policy, observation, configuration)
