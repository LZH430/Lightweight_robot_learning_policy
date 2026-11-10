"""Dex1 unit conversion profiles; neither profile chooses when to grasp/release."""
from __future__ import annotations
import math


class GripperCalibration:
    """Keep model action units, observed state units and initial state distinct."""
    def __init__(self, profile, simulator_to_joint, simulator_to_state):
        if profile not in ('simulator_v2', 'dataset_v1'):
            raise ValueError(f'Unknown gripper_calibration: {profile}')
        self.profile = profile
        self._simulator_to_joint = simulator_to_joint
        self._simulator_to_state = simulator_to_state
        self.state_max = 5.6 if profile == 'dataset_v1' else 5.4

    @staticmethod
    def _finite(value):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError('Gripper value must be finite')
        return value

    def action_to_joint(self, value):
        value = self._finite(value)
        if self.profile == 'simulator_v2':
            return self._simulator_to_joint(value)
        # Historical DDS conversion used by the published Aug 2025 dataset.
        # The caller still clips this PD target to the physical finger limits.
        value = min(5.6, max(0.0, value))
        return 0.03 - 0.05 * value / 5.6

    def joint_to_state(self, position):
        position = self._finite(position)
        if self.profile == 'simulator_v2':
            return self._simulator_to_state(position)
        # Preserve the original unrounded observation convention. In particular,
        # physical q=.0245 reports .616, while physical q=-.02 reports 5.6.
        position = min(0.03, max(-0.02, position))
        return 5.6 * (0.03 - position) / 0.05

    def initial_joint_from_state(self, state):
        # initial_state stores measured observations, not a fresh action command.
        position = self.action_to_joint(state)
        if self.profile == 'dataset_v1':
            position = min(0.0245, max(-0.02, position))
        return position

    def validate_physical_limits(self, limits):
        if self.profile == 'dataset_v1':
            if len(limits) != 4 or any(len(pair) != 2 or
                    not all(math.isfinite(float(v)) for v in pair) or
                    abs(float(pair[0]) + .02) > 1e-6 or
                    abs(float(pair[1]) - .0245) > 1e-6 for pair in limits):
                raise ValueError('dataset_v1 requires the verified Dex1 physical limits [-.02, .0245]')

    def metadata(self):
        if self.profile == 'simulator_v2':
            return {'profile': self.profile, 'source': 'current simulator tools.data_convert',
                    'command_source': 'model'}
        return {'profile': self.profile, 'command_source': 'model',
                'action_to_joint': '0.03 - 0.05 * clip(action, 0, 5.6) / 5.6; then physical-limit clipping',
                'joint_to_state': '5.6 * (0.03 - clip(q, -0.02, 0.03)) / 0.05; no rounding',
                'physical_limits': [-.02, .0245],
                'reference_sim_commit': '9101b548',
                'mapping_change_commit': 'b075304',
                'dataset_upload_commit': '1df63c16eb0665a706dfeb60a5e7040a22f91bfc',
                'dataset_upload_date': '2025-08-28', 'sim_mapping_change_date': '2025-09-04'}
