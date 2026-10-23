"""Two frozen learning heads; local traces are never model inputs."""
import gzip
import json
from pathlib import Path

from learning.task_choice_v1.agent import TaskAgent
from .model import TaskNetwork


class JointAgent(TaskAgent):
    def __init__(self, crop_path, crop_sha, task_path=None, task_sha=None, trace=None):
        selector = TaskNetwork(task_path, task_sha) if task_path else None
        self.task_sha, self.trace = task_sha, Path(trace) if trace else None
        super().__init__(selector, 'model', crop_path, crop_sha)

    def finish(self, context, result):
        result = super().finish(context, result)
        self.last['task_checkpoint_sha256'] = self.task_sha
        if self.trace:
            self.trace.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(self.trace, 'at', encoding='utf-8') as stream:
                stream.write(json.dumps(self.last, separators=(',', ':'), allow_nan=False)+'\n')
        return result
