# Appended verbatim to the frozen terminal-v2 teacher to build the standalone
# candidate. The prefix is byte-preserved; no farm controller changes.
_frozen_teacher = agent


def shed_after_transfers(observation, configuration, action):
    """Own DROP/PICKUP bookkeeping only; official engine still executes actions.

    DROP discards overflow before market execution, with inventory insertion
    order preserved. This projection never invents space by counting later SELL.
    The frozen teacher uses only DROP and PICKUP for shed transfers.
    """
    private = observation['private']
    farm = observation['farms'][observation['player']]
    shed = dict(private['shed'])
    cap = configuration['shedCapacity']
    half = configuration['boardSize']//2
    accesses = {(half-1,half-1),(half,half-1),(half-1,half),(half,half)}
    carry = 0
    for pos, inv, a in zip([farm['farmer'],*farm['hands']], private['inventories'],
                           [action['farmer'],*action['hands']]):
        carry += sum(inv.values())
        if tuple(pos) not in accesses:
            continue
        if a[0] == 'DROP':
            for c,n in inv.items():
                shed[c] = shed.get(c,0) + min(n,max(0,cap-sum(shed.values())))
            carry -= sum(inv.values())
        elif a[0] == 'PICKUP':
            c,n = a[1:3]
            take = min(n,shed.get(c,0))
            shed[c] = shed.get(c,0)-take
            carry += take
    return shed,carry


def sale_plan(observation, configuration, action):
    """Bounded intraday forecast, NOT official future price or opponent stock.

    Existing SELL slots retain at least one unit, preserving every non-SELL
    slot, including its simultaneous-opponent alignment. No order is displaced.
    Always clear crops at hour tpd-2, after it, and throughout the final day.
    Thus holding cannot roll forward forever; no global or cross-game history.
    """
    shed,carry = shed_after_transfers(observation,configuration,action)
    step,hour,day = observation['step'],observation['hour'],observation['day']
    tpd = configuration['turnsPerDay']
    last = configuration['episodeSteps']-2
    farm = observation['farms'][observation['player']]
    orders = [list(o) for o in action['market']]
    until = max(0,tpd-2-hour)
    # Cash guard uses already banked money, never predicted sale proceeds.
    clear = until == 0 or step//tpd == last//tpd or farm['money'] < 1000
    incoming = sum(o[2] for o in orders if o[0] in ('BUY_PRODUCT','BUY_ANIMAL'))
    # Retain room for carried goods, baseline purchases and another harvest wave.
    capacity = max(0,configuration['shedCapacity']-carry-incoming-16)
    fixed = sum(n for c,n in shed.items() if c not in CROPS)
    keep_budget = max(0,min(capacity-fixed,configuration['shedCapacity']//2))
    market = observation['market']
    params = market.get('params',configuration.get('marketParams',{}))
    si,ci = max(1,configuration['townShopSellInterval']),max(1,configuration['townCenterSellInterval'])
    shop_ticks = sum(t%si == 0 for t in range(step,step+until))
    center_ticks = sum(t%ci == 0 for t in range(step,step+until))
    plans = {}
    for c in sorted(CROPS):
        n = shed.get(c,0)
        if n <= 0:
            continue
        draw = center_ticks
        for shop in observation['town'].get('unlocked_shops',[]):
            products = SHOPS.get(shop,())
            if c in products:
                draw += shop_ticks*(2 if len(products)==1 else 1)
        rival_supply = 0
        for i,other in enumerate(observation['farms']):
            if i == observation['player']:
                continue
            for row in other['tiles']:
                for tile in row:
                    if (isinstance(tile,dict) and tile.get('kind')=='PLANT'
                            and tile.get('crop')==c and day-tile['planted_day']>=CROPS[c][1]):
                        rival_supply += tile['yield_units']
        # Visible ripe yield is a conservative supply scenario, not a private
        # inventory estimate. Unknown carried goods and future actions remain risk.
        net_draw = max(0,draw-rival_supply)
        keep = 0 if clear else min(max(0,n-1),net_draw,keep_budget)
        if keep:
            inventory = market['inventory'][c]
            now = sum(quote(c,inventory+n-keep+i,params) for i in range(keep))
            later = sum(quote(c,inventory+n-keep-net_draw+i,params) for i in range(keep))
            if later-now < max(keep,now*.01):
                keep = 0
        plans[c] = n-keep
        keep_budget -= keep
    seen = set()
    for order in orders:
        if order[0]=='SELL' and order[1] in plans:
            order[2] = plans[order[1]]
            seen.add(order[1])
    # Baseline may have held every unit at a very low price. New sales only use
    # trailing free slots; purchases/hires keep exact indices and quantities.
    for c,n in sorted(plans.items()):
        if c not in seen and n and len(orders)<configuration['maxMarketOrdersPerTurn']:
            orders.append(['SELL',c,n])
    return orders


def agent(observation, configuration):
    action = _frozen_teacher(observation,configuration)
    return {**action,'market':sale_plan(observation,configuration,action)}
