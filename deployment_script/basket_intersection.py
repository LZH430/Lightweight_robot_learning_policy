"""Finite-cylinder penetration of an axis-aligned basket interior (NumPy only).

The cylinder's circular end faces are flat, so this is a cylinder test, not a
capsule or centre/AABB proxy.  Box bounds describe the cavity, including the
bottom and opening height.  Shrinking those bounds by ``inward_tolerance``
rejects contact with the exterior wall/rim and objects above the opening.

Coordinates and dimensions use the same length unit (normally metres).  The
helper is pure geometry; it imposes no dwell, release, velocity or grasp rule.
"""

from __future__ import annotations

import numpy as np

_PAIR_I, _PAIR_J = np.triu_indices(8, 1)
_IDENTITY = np.eye(3, dtype=np.float64)
_BOX_NORMALS = np.concatenate((_IDENTITY, -_IDENTITY), axis=0)
_MACHINE_EPS = np.finfo(np.float64).eps


def _vector3(value, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain three finite coordinates")
    return result


def axis_from_quaternion_wxyz(quaternion, local_axis=(0.0, 0.0, 1.0)):
    """Rotate ``local_axis`` by a quaternion in explicitly WXYZ order.

    The quaternion and local axis are normalized.  Default local +Z matches
    Isaac Lab's CylinderCfg.  No simulator, torch or USD imports are needed.
    """
    quat = np.asarray(quaternion, dtype=np.float64)
    if quat.shape != (4,) or not np.all(np.isfinite(quat)):
        raise ValueError("quaternion must contain four finite WXYZ values")
    quat_norm = np.linalg.norm(quat)
    axis = _vector3(local_axis, "local_axis")
    axis_norm = np.linalg.norm(axis)
    if quat_norm == 0.0 or axis_norm == 0.0:
        raise ValueError("quaternion and local_axis must be nonzero")
    quat = quat / quat_norm
    axis = axis / axis_norm
    # Quaternion vector rotation: v + 2*w*(q_xyz x v)
    #                            + 2*(q_xyz x (q_xyz x v)).
    cross = np.cross(quat[1:], axis)
    result = axis + 2.0 * (quat[0] * cross + np.cross(quat[1:], cross))
    return result / np.linalg.norm(result)


def cylinder_intersects_box(
    center,
    axis,
    box_min,
    box_max,
    *,
    length=0.35,
    radius=0.018,
    inward_tolerance=1.0e-4,
    numerical_tolerance=1.0e-10,
):
    """Return whether any part of a finite cylinder enters the box interior.

    ``axis`` is a world-space direction; it is normalized here.  The box is
    inset on all six sides by ``inward_tolerance``.  This is deliberately
    conservative for grazing contact: penetration shallower than that inset
    can be rejected.  ``numerical_tolerance`` is a much smaller roundoff margin.

    First reject using the cylinder's exact enclosing AABB.  On overlap, clip
    the box by the cylinder's cap planes, then minimize squared radial distance
    to the cylinder axis over that polytope.  An intersecting axis segment has
    distance zero.  Otherwise a minimum occurs on a polytope edge: radial
    distance is constant in the axial direction, so a minimizer can be moved
    along that direction until it reaches an edge.  All edges are covered by
    the 28 pairs of the box's six planes and the cylinder's two cap planes.
    Line clipping and minimization are vectorized, with no iterative solver.

    Touching the original box wall/rim is false for positive inward tolerance.
    Numerical tangencies within ``numerical_tolerance`` may be classified as
    non-penetrating.  This function does not test the basket's solid wall mesh;
    supply the measured cavity bounds, not the basket's outer AABB.
    """
    center = _vector3(center, "center")
    axis = _vector3(axis, "axis")
    box_min = _vector3(box_min, "box_min")
    box_max = _vector3(box_max, "box_max")
    scalars = np.asarray(
        (length, radius, inward_tolerance, numerical_tolerance), dtype=np.float64
    )
    if not np.all(np.isfinite(scalars)):
        raise ValueError("dimensions and tolerances must be finite")
    if length <= 0.0 or radius <= 0.0:
        raise ValueError("length and radius must be positive")
    if inward_tolerance < 0.0 or numerical_tolerance < 0.0:
        raise ValueError("tolerances must be nonnegative")
    if np.any(box_max <= box_min):
        raise ValueError("box_max must exceed box_min on every axis")
    norm = np.linalg.norm(axis)
    if norm == 0.0:
        raise ValueError("axis must be nonzero")
    axis = axis / norm
    half_length = 0.5 * float(length)
    effective_radius = float(radius) - float(numerical_tolerance)
    if effective_radius <= 0.0:
        return False

    # Work relative to the cylinder centre, improving conditioning in world
    # coordinates.  For the usual 1e-4 m inset, float64 roundoff is negligible.
    lower = box_min + inward_tolerance - center
    upper = box_max - inward_tolerance - center
    if np.any(lower >= upper):
        return False
    # Sum the other two squared components instead of 1-axis**2: this
    # avoids cancellation for cylinder axes almost parallel to a box axis.
    perpendicular_norm = np.sqrt(
        axis[[1, 2, 0]] ** 2 + axis[[2, 0, 1]] ** 2
    )
    extent = half_length * np.abs(axis) + radius * perpendicular_norm
    if np.any(extent <= lower + numerical_tolerance) or np.any(
        -extent >= upper - numerical_tolerance
    ):
        return False

    # Reject a flat endcap merely supporting the box at a face, edge or corner.
    # AABB overlap alone does not detect an oblique cap-plane separation.
    axial_min = float(axis @ np.where(axis >= 0.0, lower, upper))
    axial_max = float(axis @ np.where(axis >= 0.0, upper, lower))
    if axial_min >= half_length - numerical_tolerance or axial_max <= -half_length + numerical_tolerance:
        return False

    # Cheap narrow-phase success: a point of the cylinder axis segment lies in
    # the inset box.  This also avoids the radial quadratic's nullspace case.
    parallel_axis = np.abs(axis) <= 32.0 * _MACHINE_EPS
    if not np.any(parallel_axis & ((lower > 0.0) | (upper < 0.0))):
        moving = ~parallel_axis
        first = lower[moving] / axis[moving]
        last = upper[moving] / axis[moving]
        entry = max(-half_length, float(np.max(np.minimum(first, last))))
        leave = min(half_length, float(np.min(np.maximum(first, last))))
        if entry <= leave:
            return True

    # N*x <= b: +/- box coordinate planes, then +/- cylinder cap planes.
    normals = np.concatenate((_BOX_NORMALS, axis[None, :], -axis[None, :]))
    offsets = np.concatenate((upper, -lower, [half_length, half_length]))
    first_n = normals[_PAIR_I]
    second_n = normals[_PAIR_J]
    directions = np.cross(first_n, second_n)
    norm_sq = np.einsum("ij,ij->i", directions, directions)
    nonparallel = norm_sq > (32.0 * _MACHINE_EPS) ** 2
    first_n = first_n[nonparallel]
    second_n = second_n[nonparallel]
    directions = directions[nonparallel]
    norm_sq = norm_sq[nonparallel]
    first_b = offsets[_PAIR_I[nonparallel]]
    second_b = offsets[_PAIR_J[nonparallel]]
    points = (
        first_b[:, None] * np.cross(second_n, directions)
        + second_b[:, None] * np.cross(directions, first_n)
    ) / norm_sq[:, None]
    directions = directions / np.sqrt(norm_sq)[:, None]

    coefficients = directions @ normals.T
    slack = offsets[None, :] - points @ normals.T
    parallel = np.abs(coefficients) <= 32.0 * _MACHINE_EPS
    scale = max(1.0, float(np.max(np.abs(offsets))))
    roundoff = 128.0 * _MACHINE_EPS * scale
    feasible = np.all(~parallel | (slack >= -roundoff), axis=1)
    ratios = np.zeros_like(slack)
    np.divide(slack, coefficients, out=ratios, where=~parallel)
    entry = np.max(np.where(coefficients < -32.0 * _MACHINE_EPS, ratios, -np.inf), axis=1)
    leave = np.min(np.where(coefficients > 32.0 * _MACHINE_EPS, ratios, np.inf), axis=1)
    feasible &= entry <= leave + roundoff
    if not np.any(feasible):
        return False
    points = points[feasible]
    directions = directions[feasible]
    entry = entry[feasible]
    leave = leave[feasible]

    radial_points = points - (points @ axis)[:, None] * axis
    radial_directions = directions - (directions @ axis)[:, None] * axis
    radial_speed_sq = np.einsum("ij,ij->i", radial_directions, radial_directions)
    numerator = -np.einsum("ij,ij->i", radial_points, radial_directions)
    optimum = np.zeros_like(numerator)
    np.divide(
        numerator,
        radial_speed_sq,
        out=optimum,
        where=radial_speed_sq > (32.0 * _MACHINE_EPS) ** 2,
    )
    optimum = np.clip(optimum, entry, leave)
    nearest = radial_points + optimum[:, None] * radial_directions
    distance_sq = np.einsum("ij,ij->i", nearest, nearest)
    return bool(np.any(distance_sq < effective_radius * effective_radius))


def cylinder_quaternion_intersects_box(
    center,
    quaternion_wxyz,
    box_min,
    box_max,
    *,
    length=0.35,
    radius=0.018,
    inward_tolerance=1.0e-4,
    numerical_tolerance=1.0e-10,
    local_axis=(0.0, 0.0, 1.0),
):
    """WXYZ-quaternion wrapper; default cylinder axis is local +Z."""
    return cylinder_intersects_box(
        center,
        axis_from_quaternion_wxyz(quaternion_wxyz, local_axis),
        box_min,
        box_max,
        length=length,
        radius=radius,
        inward_tolerance=inward_tolerance,
        numerical_tolerance=numerical_tolerance,
    )
