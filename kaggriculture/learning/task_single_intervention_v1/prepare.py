"""Freeze one already-consumed development case; do not run a game."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback
import types

GAME_SOURCE = Path(__file__).resolve().parents[2]
REPO = GAME_SOURCE.parent

FILES = ('prepare.py', 'worker.py', 'runner.py', 'agent_adapter.py', 'intervention.py', 'trusted_imports.py')
ENGINE_COMMIT = 'f994f2050d58d84f649329c1e35ce09ada7007c5'
PANEL_RESULT_COMMIT = '3a3e3172b0403a171a38f6e78e37d07f387be2cb'
TRUSTED_IMPORTS_SHA = '329d3b6415e3fac1cabea5d27571db3510281cbabd9c5118c55cec9aa43b115a'


class QuietParser(argparse.ArgumentParser):
    def error(self, _message):
        raise ValueError('Invalid command arguments')


def dependencies():
    """Check registered public modules before importing the older helpers."""
    raw = (REPO/'review/evidence/task-teacher-merged-eval-v3-formal-20261030/freeze.json').read_bytes()
    require(sha(raw.replace(b'\r\n', b'\n')) ==
            '249d72b0770bb97cb4a1cc66fc14586dd674c778185f6aebf8c50c73fad9cb4d',
            'Registered public runtime identity differs')
    freeze = json.loads(raw.decode('utf-8-sig'))
    for relative, digest in freeze['public_runtime_sha256'].items():
        path = (GAME_SOURCE/relative).resolve(strict=True)
        require(path.is_relative_to(GAME_SOURCE.resolve()) and path.is_file()
                and sha(path.read_bytes()) == digest,
                'Registered public runtime changed')
    for relative, digest in {
        'learning/task_prefix_diagnostic_v1/runner.py': '3e3aa71181f6e176866b9d9625760c7761bede401a9f4cd9976a0746bcce9898',
        'learning/task_prefix_diagnostic_v1/pairing.py': '7c4341f78c5f2c2758d83b62b0388e3b0b98a97518a2ec3acdf9467881132b18',
        'learning/task_prefix_diagnostic_v1/__init__.py': 'b390682ccf45f9d8abe27f10827aa6a87b16b20bec06019a83142283995ff2ff',
    }.items():
        require(sha((GAME_SOURCE/relative).read_bytes().replace(b'\r\n', b'\n')) == digest,
                'Prior diagnostic source changed')
    trusted_path = Path(__file__).resolve().parent/'trusted_imports.py'
    trusted_raw = trusted_path.read_bytes()
    require(sha(trusted_raw) == TRUSTED_IMPORTS_SHA, 'Import guard source changed')
    trusted = types.ModuleType('frozen_intervention_imports')
    exec(compile(trusted_raw, str(trusted_path), 'exec', dont_inherit=True), trusted.__dict__)
    trusted.bind(GAME_SOURCE, check_public_prefix=True)
    from learning.task_action_impact_diagnostic_v1 import analysis
    from learning.task_bc_error_diagnostics_v1.shadow import ordered_hash
    from learning.task_prefix_diagnostic_v1 import runner as panel
    return analysis, ordered_hash, panel


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_new(path, data):
    with path.open('xb') as stream:
        stream.write(data)


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8')


def within(root, relative):
    p = Path(relative)
    require(not p.is_absolute() and not any(x in ('.', '..') for x in p.parts), 'Invalid frozen path')
    value = (root/p).resolve(strict=True)
    require(value.is_relative_to(root.resolve(strict=True)) and value.is_file(), 'Frozen file outside game root')
    return value


def verify_runtime(game_root, cfg, freeze):
    """Recheck the runtime recorded by the completed panel before new play."""
    require(isinstance(freeze.get('runtime_files'), dict), 'Missing frozen runtime')
    for relative, expected in freeze['runtime_files'].items():
        require(sha(within(game_root, relative).read_bytes()) == expected, 'Frozen runtime changed')
    for name in ('merged', 'crop'):
        require(sha(within(game_root, cfg[name+'_checkpoint']).read_bytes()) == cfg[name+'_checkpoint_sha256'],
                'Frozen checkpoint changed')
    require(sha(within(game_root, 'learning/task_smoke_weight_probe_v1/core.py').read_bytes()) ==
            '771eca5e1d0093fa05b9aee5c26ccb2414c87aed75903c21843bbf71e0e98553',
            'Trace validation source changed')
    require(sha(within(game_root, 'learning/panel_eval_v3/launch.py').read_bytes()) ==
            '1bb87548635ca57552608ca8c69257717e541c497ab307e8c7f9ad178a941cd5',
            'NUL process launcher source changed')
    installed = game_root/'.venv/Lib/site-packages/kaggle_environments'
    require(sha((installed/'envs/kaggriculture/kaggriculture.py').read_bytes()) == cfg['engine_sha256']
            and sha((installed/'core.py').read_bytes()) == cfg['official_core_sha256']
            and cfg['engine_commit'] == ENGINE_COMMIT, 'Official engine identity differs')


def target_from_trace(panel_root, game, analysis, ordered_hash):
    """Consume the already accepted trace once; keep target data private."""
    trace_path = panel_root/game['trace']['path']
    meter = analysis.Meter(compressed_limit=64*2**20, decoded_limit=4*2**30)
    target, length = None, 0
    for step, trace in enumerate(analysis.trace_lines(trace_path, game['trace'], meter)):
        require(step < 719 and isinstance(trace, dict) and trace.get('step') == step
                and trace.get('checkpoint_sha256') == game['row']['crop_checkpoint_sha256']
                and trace.get('task_checkpoint_sha256') == game['row']['task_checkpoint_sha256']
                and all(trace.get(k) == 0 for k in ('load_errors', 'inference_errors', 'fallbacks')),
                'Historical trace identity differs')
        if step == 2:
            records = trace['records']
            require(isinstance(records, list) and len(records) > 5, 'Target selection missing')
            record = records[5]
            choices = record['inputs']['candidates']
            require(isinstance(choices, list) and choices and record.get('source') == 'callback',
                    'Invalid target selection')
            by_id = {c['id']: c for c in choices}
            require(len(by_id) == len(choices) and record['chosen'] in by_id
                    and record['teacher_label'] in by_id and record['chosen'] != record['teacher_label'],
                    'Target candidates invalid')
            model, teacher = by_id[record['chosen']], by_id[record['teacher_label']]
            require(model['op'] == teacher['op'] == 'PLANT' and model['target'] == teacher['target']
                    and model['unit'] != teacher['unit'], 'Unexpected historical target difference')
            target = dict(input_sha256=ordered_hash(record['inputs']), original_choice=record['chosen'],
                          teacher_choice=record['teacher_label'])
        length += 1
    require(length == 719 and target is not None and len(meter.receipts) == 1, 'Incomplete historical trace')
    return target


def wrapper(mode):
    require(mode in ('control', 'intervene'), 'Invalid arm')
    return ('''from pathlib import Path\nimport hashlib,json,sys,types\n'''
        '''_source=Path(globals().get('__file__',sys._getframe().f_code.co_filename)).resolve()\n'''
        '''_stage=_source.parent.parent\n'''
        '''_root=next(p for p in _stage.parents if (p/'run_match.py').is_file())\n'''
        '''sys.path.insert(0,str(_root))\n'''
        '''_gate=json.loads((_stage/'gate.local.json').read_text(encoding='utf-8'))\n'''
        '''_raw=(_stage/'sources/agent_adapter.py').read_bytes()\n'''
        '''if hashlib.sha256(_raw).hexdigest()!=_gate['adapter_source_sha256']: raise ValueError('Adapter changed')\n'''
        '''_module=types.ModuleType('frozen_single_adapter')\n'''
        '''exec(compile(_raw,'frozen-single-adapter','exec',dont_inherit=True),_module.__dict__)\n'''
        + f'''_agent=_module.build(_stage,_root,{mode!r})\n'''
        + '''def agent(observation,configuration):\n    return _agent(observation,configuration)\nsubmission_agent=agent\n''').encode('utf-8')


def main():
    parser = QuietParser(description=__doc__)
    parser.add_argument('--panel', required=True, type=Path)
    parser.add_argument('--status', required=True, type=Path)
    parser.add_argument('--game-root', required=True, type=Path)
    parser.add_argument('--stage', required=True, type=Path)
    args = parser.parse_args()
    analysis, ordered_hash, panel = dependencies()
    root = args.game_root.resolve(strict=True)
    panel_root = args.panel.resolve(strict=True)
    status = args.status.resolve(strict=True)
    stage = args.stage.resolve()
    require(stage.parent == (root/'artifacts').resolve(strict=True)
            and not stage.exists() and not stage.is_relative_to(panel_root), 'Experiment output is not new')
    cfg = panel.load_small(panel.REGISTRATION, panel.REGISTRATION_SHA256, normalized=True)
    freeze_path = REPO/'review/evidence/task-teacher-merged-eval-v3-formal-20261030/freeze.json'
    freeze = panel.load_small(freeze_path, panel.FREEZE_SHA256, normalized=True)
    selected = panel.checked_rows(panel_root, status)
    game = selected[('market', 0, 'merged')]
    verify_runtime(root, cfg, freeze)
    source = Path(__file__).resolve().parent
    sources = {name: (source/name).read_bytes() for name in FILES}
    stage.mkdir(parents=False, exist_ok=False)
    try:
        for name in ('sources', 'agents', 'traces', 'events', 'opponents', 'games'):
            (stage/name).mkdir(exist_ok=False)
        target = target_from_trace(panel_root, game, analysis, ordered_hash)
        hashes = {name: sha(raw) for name, raw in sources.items()}
        for name, raw in sources.items():
            write_new(stage/'sources'/name, raw)
        opponent = (panel_root/'sources/market.py').read_bytes()
        require(sha(opponent) == cfg['source_sha256']['market.py'], 'Frozen opponent changed')
        write_new(stage/'opponents/market.py', opponent)
        wrappers = {mode: wrapper(mode) for mode in ('control', 'intervene')}
        for mode, raw in wrappers.items():
            write_new(stage/'agents'/(mode+'.py'), raw)
        gate = dict(schema='single-task-gate-v1', expected=target,
            crop_checkpoint=cfg['crop_checkpoint'], crop_sha256=cfg['crop_checkpoint_sha256'],
            task_checkpoint=cfg['merged_checkpoint'], task_sha256=cfg['merged_checkpoint_sha256'],
            limits=cfg['game_limits'], intervention_source_sha256=hashes['intervention.py'],
            adapter_source_sha256=hashes['agent_adapter.py'])
        private = dict(schema='single-task-input-v1', source_panel_result_commit=PANEL_RESULT_COMMIT,
            seed=game['row']['seed'], side=0, historical_game=game['row']['game'],
            historical_replay_sha256=game['replay']['sha256'], historical_trace_sha256=game['trace']['sha256'],
            engine_sha256=cfg['engine_sha256'], engine_commit=ENGINE_COMMIT,
            opponent_sha256=sha(opponent), source_sha256=hashes,
            wrapper_sha256={mode: sha(raw) for mode, raw in wrappers.items()})
        write_new(stage/'gate.local.json', encode(gate))
        write_new(stage/'input-private.json', encode(private))
        write_new(stage/'prepared.json', encode(dict(passed=True, official_transitions=0, new_games=0,
                                                      target_step=2, selection_record_index=5)))
    except BaseException as exc:
        with (stage/'prepare-failure.local.log').open('x', encoding='utf-8') as stream:
            stream.write(traceback.format_exc())
        print(json.dumps({'passed': False, 'error_type': type(exc).__name__}), flush=True)
        raise SystemExit(1) from None
    print(json.dumps({'passed': True, 'new_games': 0, 'official_transitions': 0}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        raise
    except BaseException as exc:
        print(json.dumps({'passed': False, 'error_type': type(exc).__name__}), flush=True)
        raise SystemExit(1) from None
