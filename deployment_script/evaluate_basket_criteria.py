"""Basket-entry observation and repeatable, idempotent success-rate reporting."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

RULE_VERSION = 1
RULE = "any_positive_cylinder_overlap_with_basket_interior_at_any_rollout_step"
FINAL_STATUSES = {"completed", "interrupted", "failed"}


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix="." + path.name + ".", delete=False) as file:
            name = file.name
            json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)
            file.write("\n")
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


class BasketEntryTracker:
    """Observation only: never returns a robot command or changes rollout length."""
    def __init__(self, geometry, *, length, radius, local_axis=(0., 0., 1.), tolerance=1e-4):
        from .basket_intersection import cylinder_quaternion_intersects_box
        self._intersects = cylinder_quaternion_intersects_box
        self.geometry = geometry
        self.length, self.radius = float(length), float(radius)
        self.local_axis = tuple(local_axis)
        self.tolerance = float(tolerance)
        if not all(math.isfinite(v) and v > 0 for v in
                   (self.length, self.radius, self.tolerance)):
            raise ValueError("Cylinder dimensions and entry tolerance must be finite and positive")
        if any(b - a <= 2 * self.tolerance for a, b in
               zip(geometry["interior_min"], geometry["interior_max"])):
            raise ValueError("Basket interior is too small for the entry tolerance")
        self.success = False
        self.first_entry_time = None
        self.first_entry_pose = None

    def update(self, object_pose_wxyz, simulation_time):
        if self.success:
            return True
        if not self._intersects(object_pose_wxyz[:3], object_pose_wxyz[3:7],
                                self.geometry["interior_min"], self.geometry["interior_max"],
                                length=self.length, radius=self.radius,
                                local_axis=self.local_axis, inward_tolerance=self.tolerance):
            return False
        self.success = True
        self.first_entry_time = float(simulation_time)
        self.first_entry_pose = [float(v) for v in object_pose_wxyz[:7]]
        return True

    def metadata(self):
        return {"rule_version": RULE_VERSION, "criterion": RULE,
                "geometry": self.geometry, "cylinder_length": self.length,
                "cylinder_radius": self.radius, "cylinder_local_axis": list(self.local_axis),
                "inward_tolerance_m": self.tolerance, "check_frequency_hz": 120,
                "requires_release": False, "requires_dwell": False,
                "requires_full_containment": False,
                "first_entry_time": self.first_entry_time,
                "first_entry_pose": self.first_entry_pose}


def evaluation_group(policy_config, checkpoint, export_config, arguments, tracker):
    """Keep differing weights, initialization and execution settings separate."""
    checkpoint = Path(checkpoint).resolve()
    stat = checkpoint.stat()
    cfg = dict(policy_config, checkpoint=str(checkpoint),
               export_config=str(Path(export_config).resolve()))
    condition = {"policy_config": cfg,
                 "checkpoint_identity": {"path": str(checkpoint), "size": stat.st_size,
                                         "mtime_ns": stat.st_mtime_ns},
                 "export_config_sha256": hashlib.sha256(Path(export_config).read_bytes()).hexdigest(),
                 "execution": {k: arguments[k] for k in
                               ("seconds", "observation_hz", "action_hz", "execute_steps", "camera")},
                 "rendering": {k: arguments.get(k) for k in
                               ("headless", "rendering_mode", "device", "physics_gpu", "active_gpu", "kit_args")},
                 "criterion": RULE, "rule_version": RULE_VERSION,
                 "basket_bounds": [[round(float(x), 7) for x in tracker.geometry[key]]
                                   for key in ("interior_min", "interior_max")],
                 "cylinder": [tracker.length, tracker.radius, list(tracker.local_axis)],
                 "inward_tolerance_m": tracker.tolerance}
    identity = hashlib.sha256(json.dumps(condition, sort_keys=True).encode()).hexdigest()[:16]
    return {"id": identity, "checkpoint": str(checkpoint), "conditions": condition}


def summarize_basket_results(directory):
    """Recount summaries under a lock; repeated publication never adds a trial."""
    import fcntl
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".basket_success_rate.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        groups = {}
        rows = []
        errors = []
        legacy = 0
        for path in sorted(directory.rglob("summary.json")):
            try:
                summary = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                errors.append({"summary": str(path), "error": str(exc)})
                continue
            if not isinstance(summary, dict):
                errors.append({"summary": str(path), "error": "Summary must be a JSON object"})
                continue
            evaluation = summary.get("basket_evaluation") or {}
            if not isinstance(evaluation, dict):
                errors.append({"summary": str(path), "error": "basket_evaluation must be an object"})
                continue
            if evaluation.get("rule_version") != RULE_VERSION:
                legacy += 1
                continue
            group = summary.get("evaluation_group") or {}
            if not isinstance(group, dict):
                errors.append({"summary": str(path), "error": "evaluation_group must be an object"})
                continue
            group_id = group.get("id", "unclassified_setup_error")
            item = groups.setdefault(group_id, {"id": group_id, "checkpoint": group.get("checkpoint"),
                                               "conditions": group.get("conditions"),
                                               "total_trials": 0, "successes": 0, "failures": 0,
                                               "setup_errors": 0, "pending": 0})
            final = summary.get("status") in FINAL_STATUSES
            started = summary.get("evaluation_started", False)
            success = summary.get("success")
            if not final:
                outcome = "pending"
            elif not started or not isinstance(success, bool):
                outcome = "setup_error"
            else:
                outcome = "success" if success else "failure"
            if outcome == "pending":
                item["pending"] += 1
            elif outcome == "setup_error":
                item["setup_errors"] += 1
            else:
                item["total_trials"] += 1
                item["successes" if success else "failures"] += 1
            rows.append({"summary": str(path), "group_id": group_id, "outcome": outcome,
                         "status": summary.get("status"),
                         "first_entry_time": evaluation.get("first_entry_time")})
        for group in groups.values():
            total = group["total_trials"]
            group["basket_success_rate"] = group["successes"] / total if total else None
            group["basket_success_rate_percent"] = 100 * group["successes"] / total if total else None
        total = sum(g["total_trials"] for g in groups.values())
        successes = sum(g["successes"] for g in groups.values())
        result = {"rule_version": RULE_VERSION, "criterion": RULE,
                  "total_trials": total, "successes": successes, "failures": total - successes,
                  "basket_success_rate": successes / total if total else None,
                  "basket_success_rate_percent": 100 * successes / total if total else None,
                  "setup_errors": sum(g["setup_errors"] for g in groups.values()),
                  "pending": sum(g["pending"] for g in groups.values()),
                  "groups": list(groups.values()), "episodes": rows,
                  "ignored_legacy_summaries": legacy, "unreadable_summaries": errors,
                  "note": "Use per-group rates for model comparison; overall may combine different conditions. Setup errors are excluded; initialization failures after evaluation_started count as failures."}
        write_json_atomic(directory / "basket_success_rate.json", result)
        return result


def publish_episode_result(output, summary, *, print_result=True):
    output = Path(output)
    write_json_atomic(output / "summary.json", summary)
    if summary.get("status") not in FINAL_STATUSES:
        return None
    try:
        totals = summarize_basket_results(output.parent)
    except (OSError, ValueError) as exc:
        summary["statistics_error"] = repr(exc)
        write_json_atomic(output / "summary.json", summary)
        print(f"[basket] Could not refresh cumulative statistics: {exc}", flush=True)
        return None
    if print_result:
        identity = summary.get("evaluation_group", {}).get("id")
        group = next((g for g in totals["groups"] if g["id"] == identity), None)
        if group and group["total_trials"]:
            print(f"[basket] this_episode={summary.get('success')} | "
                  f"same_model_and_settings={group['successes']}/{group['total_trials']} "
                  f"({group['basket_success_rate_percent']:.1f}%) | "
                  f"setup_errors={group['setup_errors']} | "
                  f"stats={output.parent / 'basket_success_rate.json'}", flush=True)
        else:
            print(f"[basket] no valid trial result; setup errors recorded in "
                  f"{output.parent / 'basket_success_rate.json'}", flush=True)
    return totals
