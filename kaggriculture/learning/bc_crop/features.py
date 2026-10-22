"""Decision-time, deployment-visible features. No teacher scores or offline IDs."""
import hashlib
import json
import numpy as np
from decision.api import observe, CROPS, PRODUCTS

SHOPS = ('BAKERY', 'PIZZA_SHOP', 'BRUNCH_SPOT', 'YARN_STORE', 'ICE_CREAM_SHOP',
         'PET_CAFE', 'SMOOTHIE_SHOP', 'FARMERS_MARKET')
SHAPES = ('linear', 'sqrt', 'log', 'log10', 'sq', 'hinge')
CONFIG = ('episodeSteps', 'boardSize', 'turnsPerDay', 'shedCapacity',
          'maxMarketOrdersPerTurn', 'townShopSellInterval', 'farmHandCostMult')
SCHEMA_VERSION = 'crop-choice-v1'


def encode(context):
    # This is the same privacy boundary used by deployment. Offline metadata and
    # all unknown nested farm fields are discarded before numerical encoding.
    view = observe(context['observation'], context['configuration'])
    o, cfg = view.observation, view.configuration
    p, own, private = o['player'], view.own, view.private
    values, names = [], []
    def add(name, value):
        names.append(name)
        values.append(float(value))
    for k in CONFIG:
        add('config.' + k, cfg[k])
    for k in ('step', 'day', 'hour'):
        add(k, o[k])
    for k in ('remaining', 'today', 'last_plant_age'):
        add('planning.' + k, context[k])
    add('planning.final_day', context['final_day'])
    add('unit.index', context['i'])
    for k, pos in [('unit', context['units'][context['i']]), ('target', context['pos'])]:
        add(k + '.x', pos[0]); add(k + '.y', pos[1])
    add('planning.available_units', len(context['available']))
    add('planning.tasks_remaining', len(context['tasks']))
    for k in ('PLANT', 'WATER', 'HARVEST', 'FERTILIZE', 'DIG', 'DROP', 'PICKUP'):
        add('planning.prior_actions.' + k, sum(a[0] == k for a in context['actions']))
    for shop in SHOPS:
        add('town.' + shop, o['town']['unlocked_shops'].count(shop))
    for role, owner in [('own', p), ('rival_public', 1-p)]:
        f = o['farms'][owner]
        tiles = [t for row in f['tiles'] for t in row]
        add(role + '.money', f['money'])
        add(role + '.hands', len(f['hands']))
        add(role + '.land', len(f['unlocked_quadrants']))
        add(role + '.hires_today', f['hires_today'])
        add(role + '.empty', sum(t is None for t in tiles))
        add(role + '.weeds', sum(isinstance(t, dict) and t.get('kind') == 'WEED' for t in tiles))
        for crop in CROPS:
            plants = [t for t in tiles if isinstance(t, dict) and t.get('crop') == crop]
            prefix = role + '.' + crop + '.'
            add(prefix + 'plants', len(plants))
            add(prefix + 'yield', sum(t['yield_units'] for t in plants))
            add(prefix + 'age_sum', sum(o['day']-t['planted_day'] for t in plants))
            add(prefix + 'dry', sum(not t['watered_today'] for t in plants))
            add(prefix + 'fertilized', sum(t['fertilized_until_day'] >= o['day'] for t in plants))
            for lo, hi in ((0, 2), (2, 8), (8, 10), (10, 100)):
                add(prefix + 'ages_' + str(lo), sum(lo <= o['day']-t['planted_day'] < hi for t in plants))
    for item in PRODUCTS:
        add('shed.' + item, private['shed'].get(item, 0))
        add('projected_shed.' + item, context['projected_shed'].get(item, 0))
        add('carried_all.' + item, sum(inv.get(item, 0) for inv in private['inventories']))
        add('carried_unit.' + item, private['inventories'][context['i']].get(item, 0))
    params = o['market'].get('params', cfg.get('marketParams', {}))
    for crop in CROPS:
        add('seed.initial.' + crop, private['seeds'].get(crop, 0))
        add('seed.remaining.' + crop, context['seeds'].get(crop, 0))
        add('seed.requested.' + crop, context['needed_seeds'].get(crop, 0))
        # Existing controller forecast/reservations, not its crop score or rank.
        add('planned_supply.' + crop, context['supply'][crop])
        add('market.stock.' + crop, o['market']['inventory'][crop])
        add('market.price.' + crop, o['market']['prices'][crop])
        for k in ('base', 'I0', 'T', 'below_target', 'above_target'):
            add('market.param.' + crop + '.' + k, params.get(crop, {}).get(k, 0))
        for k in ('below_func', 'above_func'):
            for shape in SHAPES:
                add('market.param.' + crop + '.' + k + '.' + shape, params.get(crop, {}).get(k) == shape)
    x = np.asarray(values, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError('Non-finite visible feature')
    return x, names


def schema_hash(names):
    return hashlib.sha256(json.dumps([SCHEMA_VERSION, list(CROPS), names], separators=(',', ':')).encode()).hexdigest()
