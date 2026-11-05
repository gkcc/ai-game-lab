"""Explicit short official-entry regression, using only synthetic test data.

The caller supplies a fresh owned scratch directory and owns its cleanup.
This is four recorded states and three official action turns, not a 720-state
competitive match, and it never loads a learned checkpoint or a real panel.
"""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch

from .contract import hide_parsed_argv
from learning.task_pair_dev_games_v1.panel import ENGINE_COMMIT, ENGINE_SHA


REPO = Path(__file__).resolve().parents[3]
GAME = REPO.parent.parent / 'kaggriculture'
SEED = 17  # Synthetic fixture only; never a panel-seed source.


def verify(owned):
    if not __debug__:
        raise ValueError('Assertions must be enabled for this regression')
    owned = Path(owned).resolve(strict=True)
    if not (owned / '.agent-workflow-owner.json').is_file():
        raise ValueError('An owned scratch directory is required')
    stage = owned / 'official-entry'
    stage.mkdir(exist_ok=False)
    source = GAME / 'run_match.py'
    if source.read_bytes() != (REPO / 'kaggriculture/run_match.py').read_bytes():
        raise ValueError('Official CLI entry snapshot differs')
    spec = importlib.util.spec_from_file_location('teacher_panel_entry_fixture', source)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    events = stage / 'events.jsonl'
    bot = stage / 'probe.py'
    bot.write_text('''import json,sys
_events = %r
def _argv():
    assert '--seed' not in sys.argv and '17' not in sys.argv
def _record(value):
    with open(_events,'a',encoding='utf-8') as stream:
        stream.write(json.dumps(value,separators=(',',':'))+'\\n')
_argv()
_record({'kind':'loaded'})
def agent(observation,configuration):
    _argv()
    assert 'seed' not in observation and 'info' not in observation
    assert configuration.get('seed') is None
    assert all(not {'private','inventories','shed','seeds'}.intersection(farm)
               for farm in observation['farms'])
    _record({'kind':'callback','step':observation['step'],
             'player':observation['player'],'private':observation['private']})
    crop = 'WHEAT' if observation['player']==0 else 'CARROT'
    return {'farmer':['PASS'],'hands':[],
            'market':[['BUY_SEED',crop,1]] if observation['step']==0 else []}
''' % str(events), encoding='utf-8')
    before_path, before_argv = list(sys.path), sys.argv
    try:
        sys.path.insert(0, str(GAME))
        # Keep native package announcements out of the publication summary.
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            import kaggle_environments
            from decision.semantics import canonical
            from learning.task_single_intervention_v2.runner import _orders
        original_make = kaggle_environments.make
        calls = []
        def short_make(name, *, configuration, **kwargs):
            assert name == 'kaggriculture' and configuration == {'seed': SEED}
            calls.append(name)
            return original_make(name, configuration={**configuration, 'episodeSteps': 4}, **kwargs)
        sys.argv = [str(source), '--bot', str(bot), '--opponent', str(bot),
                    '--seed', str(SEED), '--name', 'fixture', '--output-root', str(stage)]
        with hide_parsed_argv(entry), patch.object(kaggle_environments, 'make', short_make), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            assert entry.main() == 0
        assert len(calls) == 1
        folder = stage / 'fixture'
        result = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
        replay = json.loads((folder / 'replay.json').read_text(encoding='utf-8'))
        recorded = [json.loads(line) for line in events.read_text(encoding='utf-8').splitlines()]
        loaded = [event for event in recorded if event['kind'] == 'loaded']
        callbacks = [event for event in recorded if event['kind'] == 'callback']
        assert result['passed'] and result['statuses'] == ['DONE', 'DONE']
        assert result['upstream_commit'] == ENGINE_COMMIT
        assert result['engine_sha256'] == ENGINE_SHA and result['engine_matches_official_commit']
        assert result['recorded_states'] == 4 and result['action_turns'] == 3
        assert not result['agent_errors'] and len(replay['steps']) == 4
        assert len(loaded) == 2 and len(callbacks) == 6
        assert {(event['player'], event['step']) for event in callbacks} == {
            (player, step) for player in (0, 1) for step in range(3)}
        distinct = False
        for event in callbacks:
            frame = replay['steps'][event['step']]
            player = event['player']
            assert event['private'] == frame[player]['observation']['private']
            if event['step'] > 0:
                distinct |= event['private'] != frame[1-player]['observation']['private']
        assert distinct, 'Fixture must exercise different player-private contents'
        env = original_make('kaggriculture',
            configuration={**replay['configuration'], 'seed': SEED}, debug=True)
        assert canonical(env.state) == canonical(replay['steps'][0])
        for expected in replay['steps'][1:]:
            env.step([state['action'] for state in expected])
            assert canonical(env.state) == canonical(expected)
            assert all(_orders(env.state, player) == _orders(expected, player) for player in (0, 1))
        assert env.done and len(env.steps) == 4
        return dict(passed=True, synthetic=True, recorded_states=4, action_turns=3,
                    agent_top_level_loads=len(loaded), checked_callbacks=len(callbacks),
                    distinct_private_contents_checked=True, official_replay_transitions=3,
                    fixed_official_engine_checked=True,
                    new_panel_games=0, model_forwards=0,
                    boundary='argv and ordinary observation path only; no adversarial process sandbox')
    finally:
        sys.path[:] = before_path
        sys.argv = before_argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--owned-scratch', required=True, type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.owned_scratch), sort_keys=True))
    except BaseException as exc:
        print(json.dumps(dict(passed=False, error_category=type(exc).__name__)))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
