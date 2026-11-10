"""Analytic regression cases for the standalone finite-cylinder helper."""
import math
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
import numpy as np
from deployment_script.basket_intersection import (
    axis_from_quaternion_wxyz,
    cylinder_intersects_box,
    cylinder_quaternion_intersects_box,
)


class CylinderBasketIntersectionTests(unittest.TestCase):
    lo = np.array([-0.1, -0.1, 0.0])
    hi = np.array([0.1, 0.1, 0.2])

    def hit(self, center, axis=(0, 0, 1), **kwargs):
        return cylinder_intersects_box(center, axis, self.lo, self.hi, **kwargs)

    def test_vertical_through_box(self):
        self.assertTrue(self.hit([0, 0, 0.1]))

    def test_vertical_center_outside_bottom_tip_inside(self):
        self.assertTrue(self.hit([0, 0, 0.37]))

    def test_vertical_above_opening(self):
        self.assertFalse(self.hit([0, 0, 0.3752]))

    def test_vertical_bottom_cap_touches_rim_plane(self):
        self.assertFalse(self.hit([0, 0, 0.375]))

    def test_horizontal_center_outside_tip_inside(self):
        self.assertTrue(self.hit([0.27, 0, 0.1], [1, 0, 0]))

    def test_horizontal_center_above_rim_radius_inside(self):
        self.assertTrue(self.hit([0, 0, 0.2178], [1, 0, 0]))

    def test_horizontal_radius_touches_rim_plane(self):
        self.assertFalse(self.hit([0, 0, 0.218], [1, 0, 0]))

    def test_vertical_center_outside_side_radius_inside(self):
        self.assertTrue(self.hit([0.1178, 0, 0.1]))

    def test_sidewall_tangent(self):
        self.assertFalse(self.hit([0.118, 0, 0.1]))

    def test_inset_rejects_too_shallow_penetration(self):
        self.assertFalse(self.hit([0.11795, 0, 0.1]))
        self.assertTrue(self.hit([0.11795, 0, 0.1], inward_tolerance=0))

    def test_corner_aabb_overlap_is_near_miss(self):
        # Both cylinder AABB coordinates overlap, but hypot(.0141,.0141)>r.
        self.assertFalse(self.hit([0.114, 0.114, 0.1]))

    def test_radius_enters_box_corner(self):
        self.assertTrue(self.hit([0.111, 0.111, 0.1]))

    def test_flat_cap_rejects_capsule_false_positive(self):
        # Nearest box corner is <r beyond the endpoint along the diagonal.
        # A capsule would intersect, but a finite cylinder does not.
        self.assertFalse(self.hit([0.235, 0.235, 0.1], [1, 1, 0]))

    def test_tilted_tip_enters_corner(self):
        self.assertTrue(self.hit([0.222, 0.222, 0.1], [1, 1, 0]))

    def test_oblique_endcap_corner_contact_is_false(self):
        axis = np.ones(3) / math.sqrt(3)
        center = self.hi + 0.175 * axis
        self.assertFalse(self.hit(center, axis, inward_tolerance=0))
        inset_corner = self.hi - 1e-4
        self.assertFalse(self.hit(inset_corner + 0.175 * axis, axis))

    def test_oblique_endcap_edge_contact_is_false(self):
        axis = np.array([1, 1, 0]) / math.sqrt(2)
        center = np.array([0.1, 0.1, 0.1]) + 0.175 * axis
        self.assertFalse(self.hit(center, axis, inward_tolerance=0))

    def test_tilted_above_rim_and_slight_entry(self):
        axis = np.array([1, 0, 1]) / math.sqrt(2)
        vertical_extent = (0.175 + 0.018) / math.sqrt(2)
        # Put the lower extremum over the centre of the box.
        center = np.array([0.175 / math.sqrt(2), 0, 0.2 + vertical_extent])
        self.assertFalse(self.hit(center, axis))
        center[2] -= 0.001
        self.assertTrue(self.hit(center, axis))

    def test_quaternion_wxyz_local_z_to_world_x(self):
        q = [math.sqrt(0.5), 0, math.sqrt(0.5), 0]
        np.testing.assert_allclose(axis_from_quaternion_wxyz(q), [1, 0, 0], atol=1e-14)
        self.assertTrue(cylinder_quaternion_intersects_box(
            [0.27, 0, 0.1], q, self.lo, self.hi))

    def test_scaled_quaternion_and_reversed_axis(self):
        q = [3, 0, 0, 0]
        self.assertTrue(cylinder_quaternion_intersects_box(
            [0, 0, 0.37], q, self.lo, self.hi))
        for center in ([0.114, 0.114, 0.1], [0.111, 0.111, 0.1]):
            self.assertEqual(self.hit(center, [1, 1, 2]), self.hit(center, [-1, -1, -2]))

    def test_nearly_parallel_axis(self):
        self.assertTrue(self.hit([0.1178, 0, 0.1], [1e-12, 0, 1]))
        self.assertFalse(self.hit([0.114, 0.114, 0.1], [1e-12, 0, 1]))

    def test_nearly_vertical_bbox_uses_stable_radial_extent(self):
        # Radius contributes 1.8e-10 m in Z.  sqrt(1-axis_z**2) rounds it to zero.
        self.assertTrue(self.hit(
            [0, 0, 0.375 + 5e-11], [1e-8, 0, 1], inward_tolerance=0))

    def test_bbox_overlap_but_axial_slab_empty(self):
        self.assertFalse(self.hit([0.235, 0.235, 0.1], [1, 1, 0]))

    def test_actual_measured_basket_bounds(self):
        lo = np.array([0.273623493985, 0.224628985278, 0.801065657055])
        hi = np.array([0.975623493985, 0.686628985278, 0.882565657055])
        center = np.array([0.65, 0.45, hi[2] + 0.175 + 0.0002])
        self.assertFalse(cylinder_intersects_box(center, [0, 0, 1], lo, hi))
        center[2] = hi[2] + 0.175 - 0.001
        self.assertTrue(cylinder_intersects_box(center, [0, 0, 1], lo, hi))

    def test_empty_inset_returns_false(self):
        self.assertFalse(cylinder_intersects_box(
            [0, 0, 0], [0, 0, 1], [-1e-5]*3, [1e-5]*3))

    def test_invalid_inputs_rejected(self):
        for kwargs in ({"length": 0}, {"radius": -1}, {"inward_tolerance": -1}):
            with self.assertRaises(ValueError):
                self.hit([0, 0, 0.1], **kwargs)
        with self.assertRaises(ValueError):
            self.hit([0, 0, 0.1], [0, 0, 0])
        with self.assertRaises(ValueError):
            axis_from_quaternion_wxyz([0, 0, 0, 0])


if __name__ == '__main__':
    unittest.main(verbosity=2)
