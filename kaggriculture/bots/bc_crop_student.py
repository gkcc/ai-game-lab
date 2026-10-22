"""Limited crop-selection BC entry. Weights are local; missing files fail closed."""
import json
import os
from pathlib import Path
import sys

# The pinned official file loader execs with empty globals (no __file__),
# but compiles with the actual filename. Worker loading supplies __file__.
_source_path = globals().get('__file__', sys._getframe().f_code.co_filename)
ROOT = Path(_source_path).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from learning.bc_crop.agent import CropAgent

_card_path = Path(os.environ.get('BC_CROP_MODEL_CARD', 'models/bc_crop_v1/checkpoint.json'))
if not _card_path.is_absolute():
    _card_path = ROOT / _card_path
_card = json.loads(_card_path.read_text(encoding='utf-8'))
_model_path = (_card_path.parent / _card['file']).resolve()
_runner = CropAgent('model', _model_path, _card['sha256'], trace=os.environ.get('BC_CROP_TRACE'))


def agent(observation, configuration):
    return _runner(observation, configuration)


# Kaggle's file loader chooses the last callable in insertion order.
submission_agent = agent
