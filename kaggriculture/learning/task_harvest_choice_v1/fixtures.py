"""Public constructed observations for a small action-expression check.

These are input fixtures, not an alternative server or a natural game history.
Only the separately invoked, fixed official engine produces the next state.
This module imports no engine, model, runtime bootstrap, or private evidence.
"""
from copy import deepcopy


NAMES = ('mature5dry', 'mature5dry_full_low', 'immature', 'already_watered')
OPS = ('WATER', 'HARVEST')
START_STEP, DAY, HOUR = 241, 10, 1
TARGET = (4, 4)
CROPS = ('WHEAT', 'CARROT', 'TOMATO', 'STRAWBERRY', 'MELON')
PRODUCTS = (*CROPS, 'EGG', 'MILK', 'WOOL', 'FERTILIZER')
ANIMALS = ('GOOSE', 'COW', 'SHEEP')
PRICES = dict(zip(PRODUCTS, (25, 35, 60, 120, 250, 50, 160, 200, 100)))
CONFIGURATION = dict(episodeSteps=720, boardSize=10, startingMoney=3000,
    maxMarketOrdersPerTurn=10, turnsPerDay=24, shedCapacity=100,
    weedSpawnChance=0.005, townShopUnlockInterval=3, townShopSellInterval=4,
    townCenterSellInterval=24, farmHandCostMult=1, marketParams={})


def _check_name(name):
    if type(name) is not str or name not in NAMES:
        raise ValueError('Unknown public harvest fixture')


def _plant(crop, planted_day, *, watered, units):
    # Literal fixture fields from the pinned official _new_plant schema. These
    # values construct inputs only; they never implement an update or reward.
    max_day = {'WHEAT': 4, 'MELON': 12}[crop]
    return dict(kind='PLANT', crop=crop, planted_day=planted_day,
        watered_today=watered, consecutive_unwatered=0 if watered else 1,
        yield_units=units, max_lifespan_step=(planted_day+max_day+1)*24,
        fertilized_until_day=-1)


def empty_private():
    """Fresh synthetic zero stock; no opponent or historical data is read."""
    return dict(shed={item:0 for item in (*PRODUCTS, *ANIMALS)},
        seeds={crop:0 for crop in CROPS}, inventories=[{}])


def _farm():
    return dict(money=0, farmer=list(TARGET), hands=[],
        unlocked_quadrants=['NW'], hires_today=0,
        tiles=[[_plant('WHEAT', DAY, watered=True, units=1)
                if x < 5 and y < 5 else 'LOCKED' for x in range(10)] for y in range(10)])


def observation(name, *, side=0):
    """Return independent legal-view inputs (observation, configuration).

    ``side`` places the target on that player's public farm and supplies only
    that player's synthetic private inventory. Other private stock is absent.
    """
    _check_name(name)
    if type(side) is not int or side not in (0, 1):
        raise ValueError('Expected one player side')
    farms = [_farm(), _farm()]
    x, y = TARGET
    farms[side]['tiles'][y][x] = _plant('MELON', DAY if name == 'immature' else 0,
        watered=name == 'already_watered', units=1 if name == 'immature' else 5)
    private = empty_private()
    market = dict(inventory={item:10000 for item in PRODUCTS}, prices=deepcopy(PRICES), params={})
    if name == 'mature5dry_full_low':
        private['shed']['FERTILIZER'] = CONFIGURATION['shedCapacity']
        market['inventory']['MELON'], market['prices']['MELON'] = 10200, 1
    obs = dict(player=side, step=START_STEP, day=DAY, hour=HOUR, farms=farms,
        private=private, market=market, town={'unlocked_shops':[]})
    return obs, deepcopy(CONFIGURATION)


def scenes():
    return [dict(name=name, observation=observation(name)[0],
                 configuration=deepcopy(CONFIGURATION)) for name in NAMES]


def direct_boundary(name, op):
    _check_name(name)
    if op not in OPS:
        raise ValueError('Unknown probe atom')
    return (name, op) in (('immature', 'HARVEST'), ('already_watered', 'WATER'))


def expectation(name, op):
    """Predeclared one-step assertions, not a simulator or policy preference."""
    _check_name(name)
    if op not in OPS:
        raise ValueError('Unknown probe atom')
    if direct_boundary(name, op):
        return dict(status='no_effect', target_removed=False,
            yield_after=1 if name == 'immature' else 5,
            watered_after=name == 'already_watered', harvested_units=0)
    if op == 'HARVEST':
        return dict(status='executed', target_removed=True, yield_after=None,
                    watered_after=None, harvested_units=5)
    return dict(status='executed', target_removed=False,
                yield_after=1 if name == 'immature' else 6,
                watered_after=True, harvested_units=0)


def specification():
    return dict(names=list(NAMES), actions=list(OPS), step=START_STEP, day=DAY, hour=HOUR,
        own_units=1, own_money=0, target_crop='MELON',
        mature_planted_day=0, mature_yield_units=5,
        immature_planted_day=DAY, immature_yield_units=1,
        already_watered_yield_units=5,
        background='other unlocked tiles contain young watered WHEAT',
        full_shed_item='FERTILIZER', full_shed_units=100,
        low_market_melon_inventory=10200, low_market_melon_price=1,
        direct_nonagent_boundaries=[dict(scene='immature', op='HARVEST'),
                                   dict(scene='already_watered', op='WATER')],
        agent_selected_requests=6, direct_boundary_requests=2,
        max_initializations=8, max_action_transitions=8,
        fixed_fixture_seed_source='existing_public_verify_mechanisms_env',
        counter_placeholders_not_replay=True, natural_trajectory=False,
        future_return_evaluated=False, model_forwards=0, training=False)
