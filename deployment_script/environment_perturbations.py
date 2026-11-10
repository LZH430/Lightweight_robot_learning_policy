"""Optional cylinder perturbations; independent of policy actions and model inputs.

Mass/material are sampled before spawning, so PhysX computes matching inertia.
Forces use the instantaneous composer each physics step to retain world direction
as the cylinder rotates and to avoid a persistent force after a pulse ends.
"""
from __future__ import annotations

import copy
import math
import random
import secrets


def _flag(value, name):
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _bounds(value, name, *, minimum=None, positive=False):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain two bounds")
    low, high = [_number(v, name) for v in value]
    if low > high or (minimum is not None and low < minimum) or (positive and low <= 0):
        raise ValueError(f"Invalid range for {name}")
    return [low, high]


class EnvironmentPerturbations:
    """One episode's configuration, sampled physical properties and force log."""

    def __init__(self, config=None):
        if config is None:
            config = {}
        if not isinstance(config, dict):
            raise ValueError("environment_perturbations must be an object")
        unknown = set(config) - {"enabled", "seed", "position", "mass", "friction", "force_pulses"}
        if unknown:
            raise ValueError(f"Unknown environment_perturbations keys: {sorted(unknown)}")
        self.config = copy.deepcopy(config)
        self.enabled = _flag(config.get("enabled", False), "environment_perturbations.enabled")
        self.sections = {}
        fields = {
            "position": {"enabled", "x", "y"},
            "mass": {"enabled", "scale_range"},
            "friction": {"enabled", "scale_range"},
            "force_pulses": {"enabled", "force_range_newtons", "duration_seconds", "interval_seconds", "start_seconds"},
        }
        for name, allowed in fields.items():
            section = config.get(name, {})
            if not isinstance(section, dict) or set(section) - allowed:
                raise ValueError(f"Invalid or unknown {name} perturbation settings")
            _flag(section.get("enabled", False), f"{name}.enabled")
            self.sections[name] = section
        self.position_range = None
        if self._on("position"):
            self.position_range = {axis: _bounds(self.sections["position"].get(axis, [-.01, .01]), f"position.{axis}")
                                   for axis in ("x", "y")}
        self.mass_range = self.friction_range = None
        if self._on("mass"):
            self.mass_range = _bounds(self.sections["mass"].get("scale_range", [.9, 1.1]), "mass.scale_range", positive=True)
        if self._on("friction"):
            self.friction_range = _bounds(self.sections["friction"].get("scale_range", [.9, 1.1]), "friction.scale_range", minimum=0.)
        self.force_enabled = self._on("force_pulses")
        self.next_start = math.inf
        if self.force_enabled:
            section = self.sections["force_pulses"]
            self.force_range = _bounds(section.get("force_range_newtons", [.1, .3]), "force_range_newtons", minimum=0.)
            self.duration_range = _bounds(section.get("duration_seconds", [.08, .12]), "duration_seconds", positive=True)
            self.interval_range = _bounds(section.get("interval_seconds", [3., 5.]), "interval_seconds", positive=True)
            self.next_start = _number(section.get("start_seconds", 2.), "start_seconds")
            if self.next_start < 0:
                raise ValueError("start_seconds must be nonnegative")
        active = self.enabled and any(self._on(name) for name in self.sections)
        seed = config.get("seed")
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**31):
            raise ValueError("perturbation seed must be null or an integer in [0, 2**31)")
        self.seed = (secrets.randbelow(2**31) if seed is None else seed) if active else None
        self.rng = random.Random(self.seed) if active else None
        self.sampled = {}
        self.events = []
        self._active_pulse = None
        self._last_time = None
        self._configured = False
        self._force_buffer = None
        self._torque_buffer = None

    def _on(self, name):
        return self.enabled and self.sections[name].get("enabled", False)

    def configure_scene(self, cfg):
        """Called once before gym.make. Disabled mode does not touch the scene."""
        if self._configured:
            raise RuntimeError("Perturbations must be sampled only once per episode")
        self._configured = True
        if self.seed is None:
            return
        cfg.seed = self.seed
        if self.position_range is not None:
            cfg.events.reset_object.params["pose_range"] = copy.deepcopy(self.position_range)
        spawn = cfg.scene.object.spawn
        if self.mass_range is not None:
            before = _number(spawn.mass_props.mass, "base cylinder mass")
            if before <= 0:
                raise ValueError("Base cylinder mass must be positive")
            scale = self.rng.uniform(*self.mass_range)
            spawn.mass_props.mass = before * scale
            self.sampled["mass"] = {"base_kg": before, "scale": scale, "sampled_kg": spawn.mass_props.mass,
                                    "inertia": "computed by PhysX from cylinder geometry and sampled mass at spawn"}
        if self.friction_range is not None:
            material = spawn.physics_material
            before_static = _number(material.static_friction, "base static friction")
            before_dynamic = _number(material.dynamic_friction, "base dynamic friction")
            if not 0 <= before_dynamic <= before_static:
                raise ValueError("Expected 0 <= dynamic friction <= static friction")
            scale = self.rng.uniform(*self.friction_range)
            material.static_friction = before_static * scale
            material.dynamic_friction = before_dynamic * scale
            self.sampled["friction"] = {"scale": scale, "base_static": before_static, "base_dynamic": before_dynamic,
                                         "sampled_static": material.static_friction, "sampled_dynamic": material.dynamic_friction,
                                         "combine_mode": material.friction_combine_mode}

    def step_force(self, time_s, dt):
        """Return the horizontal world force for this step and boundary events.

        Times are rollout simulation times; callers do not invoke this in settling.
        Pulse durations round up to physics ticks. Intervals are idle gaps after
        a pulse. An inactive pulse returns None and never leaves a persistent force.
        """
        if not self.force_enabled:
            return None, []
        time_s = _number(time_s, "force simulation time")
        dt = _number(dt, "physics dt")
        if dt <= 0 or time_s < 0 or (self._last_time is not None and time_s < self._last_time):
            raise ValueError("Force times must be nonnegative and monotonic, with dt > 0")
        self._last_time = time_s
        changes = []
        if self._active_pulse is not None and time_s + 1e-10 >= self._active_pulse["end_s"]:
            self._active_pulse["actual_end_s"] = time_s
            changes.append(dict(self._active_pulse, kind="force_end"))
            self._active_pulse = None
        if self._active_pulse is None and time_s + 1e-10 >= self.next_start:
            magnitude = self.rng.uniform(*self.force_range)
            angle = self.rng.uniform(-math.pi, math.pi)
            duration = self.rng.uniform(*self.duration_range)
            applied_duration = max(1, math.ceil(duration / dt - 1e-10)) * dt
            pulse = {"index": len(self.events), "start_s": time_s, "end_s": time_s + applied_duration,
                     "requested_duration_s": duration, "applied_duration_s": applied_duration,
                     "force_world_newtons": [magnitude * math.cos(angle), magnitude * math.sin(angle), 0.],
                     "magnitude_newtons": magnitude, "actual_end_s": None}
            self.events.append(pulse)
            self._active_pulse = pulse
            self.next_start = pulse["end_s"] + self.rng.uniform(*self.interval_range)
            changes.append(dict(pulse, kind="force_start"))
        force = self._active_pulse["force_world_newtons"] if self._active_pulse is not None else None
        return force, changes

    def apply_force(self, cylinder, time_s, dt):
        force, changes = self.step_force(time_s, dt)
        if force is None:
            return changes
        import torch
        if cylinder.num_instances != 1 or cylinder.num_bodies != 1:
            raise ValueError("This deployment perturbation runner expects one cylinder in one environment")
        if self._force_buffer is None:
            self._force_buffer = torch.zeros((1, 1, 3), dtype=torch.float32, device=cylinder.device)
            self._torque_buffer = torch.zeros_like(self._force_buffer)
        self._force_buffer[0, 0] = self._force_buffer.new_tensor(force)
        cylinder.instantaneous_wrench_composer.set_forces_and_torques(
            forces=self._force_buffer, torques=self._torque_buffer, is_global=True)
        return changes

    def finish(self, time_s):
        """Record a pulse truncated by episode termination; no new pulse is drawn."""
        if self._active_pulse is not None:
            self._active_pulse["actual_end_s"] = float(time_s)
            self._active_pulse["truncated"] = time_s + 1e-10 < self._active_pulse["end_s"]
            event = dict(self._active_pulse, kind="force_end")
            self._active_pulse = None
            return [event]
        return []

    def metadata(self):
        return {"enabled": self.enabled, "seed": self.seed, "config": copy.deepcopy(self.config),
                "position_offset_range_m": copy.deepcopy(self.position_range),
                "sampled_properties": self.sampled, "force_pulses": self.events,
                "force_frame": "world XY, applied at cylinder center of mass; no vertical force or torque",
                "force_clock": "simulation seconds after rollout starts; none during settling",
                "force_interval": "idle gap after each pulse", "sampling": "once per deployment; seed null selects a fresh recorded seed",
                "training_note": "Deployment environment only; no model weights or offline training cache are changed"}
