"""Development-only conditional teacher correction, NOT a deployable student.

The frozen model is always evaluated. Only when the conditional teacher prefers
STRAWBERRY and the model disagrees is its selected label overridden. There is no
training, checkpoint mutation, hidden fallback, or privileged observation input.
"""
import numpy as np
from learning.bc_crop.agent import CropAgent
from learning.bc_crop.features import CROPS


class StrawberryCorrection(CropAgent):
    def __init__(self, checkpoint, expected_sha, trace=None):
        super().__init__('model', checkpoint, expected_sha, trace)

    def select(self, ctx):
        network = self.network
        teacher = max(CROPS, key=lambda c: (ctx['crop_score'](c), c))
        captured = {}

        class DiagnosticOutput:
            sha = network.sha

            def logits(self, x, names):
                raw = network.logits(x, names)
                prediction = CROPS[int(np.argmax(raw))]
                applied = teacher == 'STRAWBERRY' and prediction != teacher
                effective = raw.copy()
                if applied:
                    effective[CROPS.index(teacher)] = float(np.max(raw)) + 1.
                    if CROPS[int(np.argmax(effective))] != teacher:
                        raise ValueError('Diagnostic override is not representable')
                captured.update(model_logits=raw.tolist(), model_prediction=prediction,
                                correction_applied=applied)
                return effective

        self.network = DiagnosticOutput()
        try:
            chosen = super().select(ctx)
        finally:
            self.network = network
        self.records[-1].update(captured)
        self.records[-1]['source'] = 'diagnostic_teacher_correction'
        self.records[-1]['logits_semantics'] = 'effective diagnostic choice logits; original model_logits retained separately'
        return chosen
