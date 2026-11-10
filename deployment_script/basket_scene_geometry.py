"""Read the task basket's interior from its composed USD Cube colliders.

No Isaac Sim or pxr imports occur until ``basket_interior_from_usd`` is called.
Coordinates use the stage's composed world coordinates.  In an Isaac Lab stage
these are metres.  No extra scaling based on asset-layer metersPerUnit is done:
USD references do not automatically rescale geometry from that metadata.
"""

from __future__ import annotations

import math


def basket_interior_from_usd(
    stage,
    env_root: str = "/World/envs/env_0",
    *,
    packing_table_path: str | None = None,
    placement_translation=(0.0, 0.0, 0.0),
    axis_tolerance: float = 1.0e-7,
    geometry_tolerance: float = 1.0e-5,
) -> dict:
    """Return the empty rectangular basket interior and collider provenance.

    Live stage::

        basket_interior_from_usd(stage, "/World/envs/env_0")

    Offline ``PackingTable.usd`` stage, composed as in this task::

        basket_interior_from_usd(
            stage, packing_table_path="/Root",
            placement_translation=(0.0, 0.55, -0.2),
        )

    The returned open volume is ``interior_min < point < interior_max``.
    Its lower Z is the floor collider's top; upper Z is the wall colliders'
    rim.  XY bounds are the inward faces of the opposing wall colliders.
    Pure wall/rim tangency is therefore excluded by a strict intersection
    predicate.  The visual mesh has small details that differ from the simple
    colliders; these bounds explicitly describe the physical collider interior.

    This reader intentionally accepts only the known five-Cube, horizontal,
    axis-aligned construction.  It raises ValueError on missing/ambiguous
    baskets, extra/missing collision geometry, shear, arbitrary rotations, or
    inconsistent floor/wall geometry.  There is no reward-box fallback.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    if stage is None:
        raise ValueError("A live or offline Usd.Stage is required")
    if not math.isfinite(axis_tolerance) or axis_tolerance <= 0:
        raise ValueError("axis_tolerance must be finite and positive")
    if not math.isfinite(geometry_tolerance) or geometry_tolerance <= 0:
        raise ValueError("geometry_tolerance must be finite and positive")
    try:
        translation = tuple(float(x) for x in placement_translation)
    except (TypeError, ValueError) as exc:
        raise ValueError("placement_translation must contain three finite numbers") from exc
    if len(translation) != 3 or not all(math.isfinite(x) for x in translation):
        raise ValueError("placement_translation must contain three finite numbers")

    if packing_table_path is None:
        packing_table_path = env_root.rstrip("/") + "/PackingTable"
    table = stage.GetPrimAtPath(packing_table_path)
    if not table or not table.IsActive():
        raise ValueError(f"PackingTable prim not found or inactive: {packing_table_path}")
    basket_candidates = [
        prim
        for prim in Usd.PrimRange(table, Usd.TraverseInstanceProxies())
        if prim.GetName() == "container_h20" and prim.IsActive()
    ]
    if len(basket_candidates) != 1:
        raise ValueError(
            "Expected exactly one container_h20 basket under "
            f"{packing_table_path}; found {[str(p.GetPath()) for p in basket_candidates]}"
        )
    basket = basket_candidates[0]
    collider_prims = [
        prim
        for prim in Usd.PrimRange(basket, Usd.TraverseInstanceProxies())
        if prim.HasAPI(UsdPhysics.CollisionAPI)
        and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False
    ]
    if len(collider_prims) != 5 or any(not p.IsA(UsdGeom.Cube) for p in collider_prims):
        raise ValueError(
            "Expected exactly five enabled Cube colliders in basket; found "
            + repr([(str(p.GetPath()), p.GetTypeName()) for p in collider_prims])
        )

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    colliders = []
    for prim in collider_prims:
        cube = UsdGeom.Cube(prim)
        size = float(cube.GetSizeAttr().Get())
        if not math.isfinite(size) or size <= 0:
            raise ValueError(f"Invalid Cube size on {prim.GetPath()}: {size}")
        transform = cache.GetLocalToWorldTransform(prim)
        # Each transformed basis vector must align with a distinct world axis.
        # This allows 90-degree axis permutations and reflections, but no shear.
        mapped_axes = []
        for i in range(3):
            basis = [0.0, 0.0, 0.0]
            basis[i] = 1.0
            direction = tuple(float(x) for x in transform.TransformDir(Gf.Vec3d(*basis)))
            norm = math.sqrt(sum(x * x for x in direction))
            if not math.isfinite(norm) or norm <= 0:
                raise ValueError(f"Degenerate/nonfinite collider transform: {prim.GetPath()}")
            axis = max(range(3), key=lambda j: abs(direction[j]))
            if any(abs(direction[j]) > axis_tolerance * norm for j in range(3) if j != axis):
                raise ValueError(f"Non-axis-aligned basket collider: {prim.GetPath()}")
            mapped_axes.append(axis)
        if len(set(mapped_axes)) != 3:
            raise ValueError(f"Sheared/degenerate basket collider: {prim.GetPath()}")
        points = [
            tuple(
                float(value) + translation[i]
                for i, value in enumerate(
                    transform.Transform(Gf.Vec3d(x * size / 2, y * size / 2, z * size / 2))
                )
            )
            for x in (-1, 1)
            for y in (-1, 1)
            for z in (-1, 1)
        ]
        lower = [min(p[i] for p in points) for i in range(3)]
        upper = [max(p[i] for p in points) for i in range(3)]
        lengths = [upper[i] - lower[i] for i in range(3)]
        if not all(math.isfinite(x) and x > 0 for x in lengths):
            raise ValueError(f"Invalid collider bounds: {prim.GetPath()}")
        thin_axis = min(range(3), key=lengths.__getitem__)
        if lengths[thin_axis] >= 0.25 * min(lengths[i] for i in range(3) if i != thin_axis):
            raise ValueError(f"Collider is not a thin floor/wall: {prim.GetPath()}")
        colliders.append(
            {
                "prim_path": str(prim.GetPath()),
                "usd_type": prim.GetTypeName(),
                "cube_size": size,
                "world_min": lower,
                "world_max": upper,
                "world_lengths": lengths,
                "thin_axis": thin_axis,
                "local_to_stage_world": [[float(transform[i][j]) for j in range(4)] for i in range(4)],
            }
        )

    groups = {axis: [c for c in colliders if c["thin_axis"] == axis] for axis in range(3)}
    if [len(groups[i]) for i in range(3)] != [2, 2, 1]:
        raise ValueError("Basket must have two X walls, two Y walls, and one horizontal floor")
    floor = groups[2][0]
    walls = groups[0] + groups[1]
    floor_top = floor["world_max"][2]
    wall_bottom = min(c["world_min"][2] for c in walls)
    rim_top = min(c["world_max"][2] for c in walls)
    if any(abs(c["world_min"][2] - wall_bottom) > geometry_tolerance for c in walls):
        raise ValueError("Basket walls have inconsistent bottom heights")
    if any(abs(c["world_max"][2] - rim_top) > geometry_tolerance for c in walls):
        raise ValueError("Basket walls have inconsistent rim heights")
    if not floor_top < rim_top:
        raise ValueError("Basket floor is at or above the rim")
    if wall_bottom - floor_top > floor["world_lengths"][2] + geometry_tolerance:
        raise ValueError("Basket wall-to-floor gap is too large")
    if floor_top > wall_bottom + floor["world_lengths"][2] + geometry_tolerance:
        raise ValueError("Basket floor substantially intersects the walls")

    lower = [0.0, 0.0, floor_top]
    upper = [0.0, 0.0, rim_top]
    wall_paths = {}
    for axis in (0, 1):
        low, high = sorted(
            groups[axis], key=lambda c: (c["world_min"][axis] + c["world_max"][axis]) / 2
        )
        lower[axis] = low["world_max"][axis]
        upper[axis] = high["world_min"][axis]
        if lower[axis] >= upper[axis]:
            raise ValueError(f"Opposing basket walls overlap on axis {axis}")
        wall_paths["xy"[axis] + "_min"] = low["prim_path"]
        wall_paths["xy"[axis] + "_max"] = high["prim_path"]
        if abs(low["world_lengths"][axis] - high["world_lengths"][axis]) > geometry_tolerance:
            raise ValueError(f"Opposing basket wall thicknesses differ on axis {axis}")
    for axis in (0, 1):
        if floor["world_min"][axis] > lower[axis] + geometry_tolerance:
            raise ValueError("Basket floor does not cover the interior")
        if floor["world_max"][axis] < upper[axis] - geometry_tolerance:
            raise ValueError("Basket floor does not cover the interior")
        transverse = 1 - axis
        for wall in groups[axis]:
            # The authored asset has 3 mm corner gaps between wall colliders.
            # Permit small endpoint gaps up to wall thickness, while excluding
            # a short/displaced collider that would not describe a basket.
            allowance = max(c["world_lengths"][transverse] for c in groups[transverse])
            if wall["world_min"][transverse] > lower[transverse] + allowance + geometry_tolerance:
                raise ValueError("Basket wall does not span the opening")
            if wall["world_max"][transverse] < upper[transverse] - allowance - geometry_tolerance:
                raise ValueError("Basket wall does not span the opening")

    return {
        "interior_min": lower,
        "interior_max": upper,
        "interior_center": [(a + b) / 2 for a, b in zip(lower, upper)],
        "interior_size": [b - a for a, b in zip(lower, upper)],
        "floor_top": floor_top,
        "rim_top": rim_top,
        "basket_prim_path": str(basket.GetPath()),
        "packing_table_prim_path": str(table.GetPath()),
        "floor_collider_prim_path": floor["prim_path"],
        "wall_collider_prim_paths": wall_paths,
        "collider_prim_paths": [c["prim_path"] for c in colliders],
        "provenance": {
            "method": "inward world faces of five composed USD Cube colliders",
            "region": "open rectangular physical collider interior; no visual-mesh or reward-box approximation",
            "axis_aligned": True,
            "physical_taper": False,
            "stage_root_layer": stage.GetRootLayer().identifier,
            "stage_meters_per_unit_metadata": float(UsdGeom.GetStageMetersPerUnit(stage)),
            "automatic_unit_conversion_applied": False,
            "placement_translation": list(translation),
            "wall_bottom": wall_bottom,
            "floor_to_wall_bottom_gap": wall_bottom - floor_top,
            "axis_tolerance": axis_tolerance,
            "geometry_tolerance": geometry_tolerance,
            "colliders": colliders,
        },
    }
