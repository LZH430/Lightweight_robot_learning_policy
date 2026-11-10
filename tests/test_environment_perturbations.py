"""CPU regressions for environmental sampling and finite world-frame pulses."""
import copy
import importlib.util
import math
from pathlib import Path
import random
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[1] / "deployment_script/environment_perturbations.py"
spec = importlib.util.spec_from_file_location("environment_perturbations", MODULE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Perturbations = module.EnvironmentPerturbations


def settings():
    return {"enabled": True, "seed": 42,
            "position": {"enabled": True, "x": [-.01, .01], "y": [-.01, .01]},
            "mass": {"enabled": True, "scale_range": [.9, 1.1]},
            "friction": {"enabled": True, "scale_range": [.9, 1.1]},
            "force_pulses": {"enabled": True, "force_range_newtons": [.2, .2],
                             "duration_seconds": [.1, .1], "interval_seconds": [.3, .3], "start_seconds": 2.}}


def scene():
    return NS(seed=123, events=NS(reset_object=NS(params={"pose_range": {"x": [0., 0.], "y": [0., 0.]}})),
              scene=NS(object=NS(spawn=NS(mass_props=NS(mass=.4), physics_material=NS(
                  static_friction=1.5, dynamic_friction=1.2, friction_combine_mode="max")))))


class SamplingTests(unittest.TestCase):
    def test_absent_and_master_off_do_not_change_scene_or_apply_forces(self):
        for config in (None, dict(settings(), enabled=False)):
            p = Perturbations(config)
            cfg = scene(); before = copy.deepcopy(cfg)
            p.configure_scene(cfg)
            self.assertEqual(cfg, before)
            self.assertIsNone(p.seed)
            for t in (0., 2., 20.):
                self.assertEqual(p.apply_force(None, t, 1/120), [])

    def test_scene_samples_are_bounded_and_friction_consistent(self):
        p = Perturbations(settings()); cfg = scene(); p.configure_scene(cfg)
        self.assertEqual(cfg.seed, 42)
        self.assertEqual(cfg.events.reset_object.params["pose_range"], {"x": [-.01, .01], "y": [-.01, .01]})
        self.assertTrue(.36 <= cfg.scene.object.spawn.mass_props.mass <= .44)
        material = cfg.scene.object.spawn.physics_material
        self.assertTrue(1.35 <= material.static_friction <= 1.65)
        self.assertLessEqual(material.dynamic_friction, material.static_friction)
        self.assertAlmostEqual(material.dynamic_friction/material.static_friction, 1.2/1.5)
        self.assertEqual(material.friction_combine_mode, "max")

    def test_logged_auto_seed_replays_properties_and_pulses(self):
        cfg = settings(); cfg["seed"] = None
        with patch.object(module.secrets, "randbelow", return_value=10007):
            p = Perturbations(cfg)
        explicit = copy.deepcopy(cfg); explicit["seed"] = p.seed
        q = Perturbations(explicit)
        a,b = scene(),scene(); p.configure_scene(a); q.configure_scene(b)
        self.assertEqual(a,b)
        for k in range(400):
            self.assertEqual(p.step_force(k/120,1/120), q.step_force(k/120,1/120))
        self.assertEqual(p.events,q.events)

    def test_sampler_does_not_change_global_rng(self):
        random.seed(25); expected=random.random(); random.seed(25)
        p=Perturbations(settings()); p.configure_scene(scene()); p.step_force(2.,1/120)
        self.assertEqual(random.random(),expected)

    def test_subsystems_can_be_disabled_separately(self):
        cfg=settings(); cfg["position"]["enabled"]=False; cfg["friction"]["enabled"]=False
        p=Perturbations(cfg); obj=scene(); p.configure_scene(obj)
        self.assertEqual(obj.events.reset_object.params["pose_range"],{"x":[0.,0.],"y":[0.,0.]})
        self.assertEqual(obj.scene.object.spawn.physics_material.static_friction,1.5)
        self.assertNotEqual(obj.scene.object.spawn.mass_props.mass,.4)

    def test_repeat_configuration_cannot_compound_mass(self):
        p=Perturbations(settings()); obj=scene(); p.configure_scene(obj)
        before=copy.deepcopy(obj)
        with self.assertRaises(RuntimeError): p.configure_scene(obj)
        self.assertEqual(before,obj)

    def test_invalid_inputs_rejected(self):
        edits=[("seed", True),("seed", -1),("seed",2**31)]
        for key,value in edits:
            cfg=settings(); cfg[key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError): Perturbations(cfg)
        for section,key,value in [("position","x",[1.,-1.]),("position","y",[0.,math.nan]),
                                  ("mass","scale_range",[0.,1.]),("friction","scale_range",[-1.,1.]),
                                  ("force_pulses","duration_seconds",[0.,.1]),
                                  ("force_pulses","interval_seconds",[0.,1.]),
                                  ("force_pulses","force_range_newtons",[-1.,1.]),
                                  ("force_pulses","start_seconds",-1.)]:
            cfg=settings(); cfg[section][key]=value
            with self.subTest(section=section,key=key),self.assertRaises(ValueError): Perturbations(cfg)
        with self.assertRaises(ValueError): Perturbations(dict(settings(), unexpected=True))
        cfg=settings(); cfg["mass"]["scale_ragne"]=[.9,1.1]
        with self.assertRaises(ValueError): Perturbations(cfg)


class ForceTests(unittest.TestCase):
    def plan(self,start=2.,duration=.1):
        cfg=settings(); cfg["force_pulses"]["start_seconds"]=start
        cfg["force_pulses"]["duration_seconds"]=[duration,duration]
        p=Perturbations(cfg); p.configure_scene(scene()); return p

    def test_no_force_before_start_and_no_tail_after_pulse(self):
        p=self.plan(); self.assertEqual(p.step_force(1.999,1/120),(None,[]))
        forces=[]; kinds=[]
        for k in range(240,254):
            force,events=p.step_force(k/120,1/120); forces.append(force); kinds += [v["kind"] for v in events]
        self.assertEqual(sum(v is not None for v in forces),12)
        self.assertEqual(kinds,["force_start","force_end"])
        self.assertIsNone(forces[-1])
        self.assertAlmostEqual(p.events[0]["actual_end_s"],2.1)
        for force in [v for v in forces if v is not None]:
            self.assertEqual(force[2],0.)
            self.assertAlmostEqual(math.hypot(*force[:2]),.2)

    def test_idle_gap_is_after_previous_end(self):
        p=self.plan(start=0.)
        p.step_force(0.,1/120); p.step_force(.1,1/120)
        self.assertEqual(p.step_force(.399,1/120),(None,[]))
        force,events=p.step_force(.4,1/120)
        self.assertIsNotNone(force); self.assertEqual(events[0]["kind"],"force_start")

    def test_nonintegral_duration_rounds_up_one_tick_at_most(self):
        p=self.plan(start=0.,duration=.101)
        p.step_force(0.,1/120)
        applied=p.events[0]["applied_duration_s"]
        self.assertGreaterEqual(applied,.101)
        self.assertLess(applied-.101,1/120)

    def test_finish_records_truncation_without_new_pulse(self):
        p=self.plan(); p.step_force(2.,1/120)
        changes=p.finish(2.05)
        self.assertEqual(len(p.events),1)
        self.assertEqual(changes[0]["actual_end_s"],2.05)
        self.assertTrue(changes[0]["truncated"])
        self.assertEqual(p.finish(2.05),[])

    def test_rejects_backwards_clock_and_invalid_dt(self):
        p=self.plan(); p.step_force(1.,1/120)
        with self.assertRaises(ValueError): p.step_force(.9,1/120)
        with self.assertRaises(ValueError): p.step_force(1.,0.)

    def test_uses_only_instantaneous_world_wrench_during_active_steps(self):
        calls=[]
        class Asset:
            num_instances=1; num_bodies=1; device="cpu"
            instantaneous_wrench_composer=NS(set_forces_and_torques=lambda **kw: calls.append(
                {k:v.clone() if hasattr(v,"clone") else v for k,v in kw.items()}))
            @property
            def permanent_wrench_composer(self):
                raise AssertionError("A pulse must not leave a persistent wrench")
        asset=Asset(); p=self.plan(start=0.)
        for k in range(14): p.apply_force(asset,k/120,1/120)
        self.assertEqual(len(calls),12)
        for call in calls:
            self.assertTrue(call["is_global"])
            self.assertEqual(tuple(call["forces"].shape),(1,1,3))
            self.assertAlmostEqual(float(call["forces"].norm()),.2,places=6)
            self.assertEqual(float(call["forces"][0,0,2]),0.)
            self.assertEqual(float(call["torques"].norm()),0.)


if __name__=="__main__":
    unittest.main()
