"""Calibration regressions that run without Isaac Sim, Torch or a GPU.

Run from the student repository with:
    python -m unittest discover -s tests -p 'test_gripper_calibration.py'
SIM_REPO can override the simulator checkout used for delegate compatibility.
"""

import importlib.util
import math
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deployment_script.gripper_calibration import GripperCalibration


class DatasetCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.to_joint = Mock(side_effect=AssertionError("Dataset profile used simulator action mapping"))
        self.to_state = Mock(side_effect=AssertionError("Dataset profile used simulator state mapping"))
        self.calibration = GripperCalibration("dataset_v1", self.to_joint, self.to_state)

    def test_historical_action_endpoints_and_nominal_open_command(self):
        self.assertAlmostEqual(self.calibration.action_to_joint(0.0), 0.03, places=14)
        self.assertAlmostEqual(self.calibration.action_to_joint(5.4), -0.018214285714285714, places=14)
        self.assertAlmostEqual(self.calibration.action_to_joint(5.6), -0.02, places=14)
        self.assertEqual(self.calibration.action_to_joint(-100.0), self.calibration.action_to_joint(0.0))
        self.assertEqual(self.calibration.action_to_joint(100.0), self.calibration.action_to_joint(5.6))

    def test_historical_feedback_including_physical_closed_limit(self):
        self.assertAlmostEqual(self.calibration.joint_to_state(0.03), 0.0, places=14)
        self.assertAlmostEqual(self.calibration.joint_to_state(0.0245), 0.616, places=13)
        self.assertAlmostEqual(self.calibration.joint_to_state(-0.02), 5.6, places=13)
        self.assertEqual(self.calibration.joint_to_state(1.0), 0.0)
        self.assertAlmostEqual(self.calibration.joint_to_state(-1.0), 5.6, places=13)

    def test_reachable_physical_positions_and_states_round_trip(self):
        for position in (-0.02, -0.018214285714285714, 0.0, 0.008, 0.019, 0.0245):
            with self.subTest(position=position):
                state = self.calibration.joint_to_state(position)
                self.assertAlmostEqual(self.calibration.action_to_joint(state), position, places=14)
                self.assertAlmostEqual(self.calibration.initial_joint_from_state(state), position, places=14)
        for state in (0.616, 1.0, 2.332, 4.0, 5.4, 5.6):
            with self.subTest(state=state):
                position = self.calibration.action_to_joint(state)
                self.assertAlmostEqual(self.calibration.joint_to_state(position), state, places=13)

    def test_measured_initial_state_is_recovered_then_physically_clamped(self):
        self.assertAlmostEqual(self.calibration.initial_joint_from_state(0.616), 0.0245, places=14)
        # The actual float32 observation present in the published dataset.
        self.assertAlmostEqual(
            self.calibration.initial_joint_from_state(0.6160000562667847),
            0.024499999497617993,
            places=14,
        )
        self.assertEqual(self.calibration.initial_joint_from_state(0.0), 0.0245)
        self.assertAlmostEqual(self.calibration.initial_joint_from_state(5.6), -0.02, places=14)
        # An action remains a PD target; its physical clipping is the caller's job.
        self.assertEqual(self.calibration.action_to_joint(0.0), 0.03)

    def test_feedback_retains_submillimetre_position_information(self):
        first = self.calibration.joint_to_state(0.0080)
        second = self.calibration.joint_to_state(0.0081)
        self.assertGreater(first, second)
        self.assertAlmostEqual(first - second, 0.0112, places=13)
        self.assertAlmostEqual(first, 2.464, places=13)
        self.assertAlmostEqual(second, 2.4528, places=13)

    def test_verified_four_finger_limits_are_accepted(self):
        self.calibration.validate_physical_limits([[-0.02, 0.0245] for _ in range(4)])
        self.calibration.validate_physical_limits(
            [[-0.019999999552965164, 0.02449999935925007] for _ in range(4)]
        )

    def test_other_hardware_limits_and_shapes_are_rejected(self):
        wrong_cases = (
            [],
            [[-0.02, 0.0245] for _ in range(3)],
            [[-0.02, 0.0245] for _ in range(5)],
            [[-0.02] for _ in range(4)],
            [[-0.02, 0.0245, 0.03] for _ in range(4)],
            [[-0.019, 0.0245] for _ in range(4)],
            [[-0.02, 0.024] for _ in range(4)],
            [[-0.02, 0.03] for _ in range(4)],
            [[0.0245, -0.02] for _ in range(4)],
        )
        for limits in wrong_cases:
            with self.subTest(limits=limits):
                with self.assertRaises(ValueError):
                    self.calibration.validate_physical_limits(limits)

    def test_nonfinite_physical_limits_are_rejected(self):
        for value in (math.nan, math.inf, -math.inf):
            for endpoint in (0, 1):
                limits = [[-0.02, 0.0245] for _ in range(4)]
                limits[2][endpoint] = value
                with self.subTest(value=value, endpoint=endpoint):
                    with self.assertRaises(ValueError):
                        self.calibration.validate_physical_limits(limits)


class SimulatorCompatibilityTests(unittest.TestCase):
    def test_existing_profile_delegates_each_conversion_without_altering_result(self):
        to_joint = Mock(return_value=-17.125)
        to_state = Mock(return_value=11.75)
        calibration = GripperCalibration("simulator_v2", to_joint, to_state)
        self.assertEqual(calibration.action_to_joint(8.75), -17.125)
        to_joint.assert_called_once_with(8.75)
        self.assertEqual(calibration.joint_to_state(0.0081), 11.75)
        to_state.assert_called_once_with(0.0081)
        to_joint.reset_mock()
        self.assertEqual(calibration.initial_joint_from_state(0.616), -17.125)
        to_joint.assert_called_once_with(0.616)
        self.assertEqual(calibration.state_max, 5.4)
        # Historical hardware validation must not alter the fallback profile.
        calibration.validate_physical_limits([[-1.0, 1.0]])

    def test_existing_profile_matches_actual_simulator_functions_exactly(self):
        sim_repo = Path(os.environ.get("SIM_REPO", "/home/lzh/unitree_sim_isaaclab"))
        source = sim_repo / "tools" / "data_convert.py"
        if not source.is_file():
            self.skipTest(f"Set SIM_REPO to test the simulator conversion functions: {source}")
        spec = importlib.util.spec_from_file_location("_test_simulator_data_convert", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        calibration = GripperCalibration(
            "simulator_v2", module.convert_to_joint_range, module.convert_to_gripper_range
        )
        for value in (-10.0, 0.0, 0.616, 2.332, 5.4, 5.6, 10.0):
            with self.subTest(action=value):
                self.assertEqual(calibration.action_to_joint(value), module.convert_to_joint_range(value))
                self.assertEqual(calibration.initial_joint_from_state(value), module.convert_to_joint_range(value))
        for position in (-0.03, -0.02, 0.0, 0.0080, 0.0081, 0.024, 0.0245, 0.03):
            with self.subTest(position=position):
                self.assertEqual(calibration.joint_to_state(position), module.convert_to_gripper_range(position))


class ValidationTests(unittest.TestCase):
    def test_unknown_profiles_are_rejected(self):
        for profile in (None, "", "dataset_v2", "legacy", "DATASET_V1"):
            with self.subTest(profile=profile):
                with self.assertRaises(ValueError):
                    GripperCalibration(profile, Mock(), Mock())

    def test_nonfinite_values_fail_before_reaching_either_profile(self):
        for profile in ("dataset_v1", "simulator_v2"):
            to_joint, to_state = Mock(), Mock()
            calibration = GripperCalibration(profile, to_joint, to_state)
            for method_name in ("action_to_joint", "joint_to_state", "initial_joint_from_state"):
                for value in (math.nan, math.inf, -math.inf):
                    with self.subTest(profile=profile, method=method_name, value=value):
                        with self.assertRaises(ValueError):
                            getattr(calibration, method_name)(value)
            to_joint.assert_not_called()
            to_state.assert_not_called()


if __name__ == "__main__":
    unittest.main()
