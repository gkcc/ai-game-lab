"""Maximum-cardinality, maximum-logit matching over current opportunities.

One unit and one target cell may appear once in a provisional matching. This
inherits the frozen controller's reservation policy, not official legality.
Only one chosen ID is returned for execution; the next callback must rebuild
its graph. No seed, teacher label, observation or model is accessed here.
"""
from dataclasses import dataclass
import math
from numbers import Real


class DecoderError(ValueError):
    """Malformed identities, nonfinite scores or inconsistent matching input."""


def _require(value):
    if not value: raise DecoderError('Invalid current candidate graph or scores')


@dataclass(frozen=True, slots=True)
class Selection:
    choice_id: str
    matching_ids: tuple[str, ...]
    cardinality: int
    # None means the exact sum lies outside finite binary64. The matching is
    # still decided using exact integer-scaled binary64 scores, not infinity.
    total_logit: float | None


def candidate_key(candidate):
    """The existing encoder's public canonical order: unit, y, x, op, ID."""
    _require(type(candidate) is dict and type(candidate.get('id')) is str and candidate['id']
        and type(candidate.get('unit')) is int and candidate['unit']>=0
        and type(candidate.get('target')) in (list,tuple) and len(candidate['target'])==2
        and all(type(v) is int for v in candidate['target'])
        and type(candidate.get('op')) is str and candidate['op'])
    x,y=candidate['target']
    return candidate['unit'],y,x,candidate['op'],candidate['id']


def _graph(candidates, logits):
    _require(type(candidates) in (list,tuple) and len(candidates)>0
        and type(logits) in (list,tuple) and len(logits)==len(candidates))
    identities=set();pairs={}
    for candidate,value in zip(candidates,logits):
        key=candidate_key(candidate)
        _require(candidate['id'] not in identities and isinstance(value,Real) and not isinstance(value,bool))
        try:score=float(value)
        except (ValueError,TypeError,OverflowError):raise DecoderError('Invalid current candidate score') from None
        _require(math.isfinite(score));identities.add(candidate['id'])
        pair=(candidate['unit'],tuple(candidate['target']))
        edge=(candidate['id'],score,key)
        current=pairs.get(pair)
        if current is None or score>current[1] or (score==current[1] and key<current[2]):pairs[pair]=edge
    return pairs


def _add(graph,left,right,cost):
    forward=[right,len(graph[right]),1,cost]
    reverse=[left,len(graph[left]),0,-cost]
    graph[left].append(forward);graph[right].append(reverse)
    return forward


def _augment(graph,start,end):
    """Longest augmenting path in an optimal residual matching, with integers.

    Successive augmentation first maximizes flow/cardinality, even when every
    available path has negative weight. Bellman-Ford handles residual reverse
    edges without an optional dependency or a fixed official worker limit.
    """
    size=len(graph);distance=[None]*size;previous=[None]*size;distance[start]=0
    for _ in range(size-1):
        changed=False
        for left,edges in enumerate(graph):
            if distance[left] is None:continue
            for index,(right,_reverse,capacity,cost) in enumerate(edges):
                if not capacity:continue
                trial=distance[left]+cost
                if distance[right] is None or trial>distance[right]:
                    distance[right]=trial;previous[right]=(left,index);changed=True
        if not changed:break
    # An improving residual cycle would violate the optimum maintained after
    # each augmentation; fail instead of returning a silently approximate plan.
    for left,edges in enumerate(graph):
        if distance[left] is None:continue
        for right,_reverse,capacity,cost in edges:
            _require(not capacity or (distance[right] is not None and distance[right]>=distance[left]+cost))
    if distance[end] is None:return False
    node=end;seen=set()
    while node!=start:
        _require(node not in seen and previous[node] is not None);seen.add(node)
        left,index=previous[node];edge=graph[left][index]
        _require(edge[2]==1)
        edge[2]=0;graph[node][edge[1]][2]=1;node=left
    return True


def choose(candidates, logits):
    """Choose one current ID using frozen scores aligned with candidate order.

    Objective: maximum cardinality, then maximum exact sum of the supplied
    binary64 logits, then lexicographically smallest sorted canonical keys.
    Multiple operations on one unit/cell retain their best score (key on ties).
    From that matching choose the largest raw logit, canonical key on ties.
    Input order cannot settle a tie. Scores are preferences, not cash values.
    """
    pairs=_graph(candidates,logits)
    units=sorted({u for u,_target in pairs});targets=sorted({t for _unit,t in pairs})
    unit_nodes={u:i+1 for i,u in enumerate(units)}
    target_nodes={t:i+1+len(units) for i,t in enumerate(targets)}
    end=1+len(units)+len(targets);graph=[[] for _ in range(end+1)]
    for unit in units:_add(graph,0,unit_nodes[unit],0)
    for target in targets:_add(graph,target_nodes[target],end,0)
    ordered=sorted(pairs.items(),key=lambda pair:pair[1][2])
    ratios=[edge[1].as_integer_ratio() for _pair,edge in ordered]
    exponent=max(denominator.bit_length()-1 for _n,denominator in ratios)
    # Binary canonical-key preference is subordinate to a one-unit difference
    # in the exact scaled sum. Cardinality is handled by augmenting to exhaustion.
    tie_bits=len(ordered);tracked=[]
    for rank,((unit,target),(identifier,score,key)) in enumerate(ordered):
        numerator,denominator=ratios[rank]
        weight=numerator << (exponent-(denominator.bit_length()-1))
        cost=(weight << tie_bits)+(1 << (tie_bits-1-rank))
        forward=_add(graph,unit_nodes[unit],target_nodes[target],cost)
        tracked.append((identifier,score,weight,key,forward))
    cardinality=0
    while _augment(graph,0,end):cardinality+=1
    matching=[edge for edge in tracked if edge[4][2]==0]
    _require(cardinality==len(matching)>0)
    chosen=min(matching,key=lambda edge:(-edge[1],edge[3]))
    try:total=sum(edge[2] for edge in matching)/(1 << exponent)
    except OverflowError:total=None
    _require(total is None or math.isfinite(total))
    return Selection(chosen[0],tuple(edge[0] for edge in matching),cardinality,total)
