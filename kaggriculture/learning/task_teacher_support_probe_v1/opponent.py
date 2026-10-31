"""Record a frozen teacher's selected task intent for private prefix checks."""
from copy import deepcopy

from learning.task_multi_teacher_probe_v1.teacher import ProbeError, TeacherProbe


class SupportTeacher(TeacherProbe):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._used = False

    def __call__(self, observation, configuration):
        if self._used:
            self.clear()
            raise ProbeError('a support teacher permits only one turn')
        self._used = True
        return super().__call__(observation, configuration)

    def _select(self, context, original_max):
        before = len(self._records)
        choice = super()._select(context, original_max)
        if len(self._records) != before + 1:
            raise ProbeError('teacher task record differs')
        task_index = choice[-1]
        task = context['tasks'][task_index]
        action = task[1]
        if type(action) is not list or not action:
            raise ProbeError('teacher task intent differs')
        self._records[-1]['intended_action'] = deepcopy(action)
        return choice
