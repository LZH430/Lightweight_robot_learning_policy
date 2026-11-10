"""One simulated closed-loop episode. No DDS or physical robot connection."""
import argparse
import faulthandler
import traceback
from datetime import datetime
from collections import deque
import json
import os
from pathlib import Path
import sys

p=argparse.ArgumentParser()
p.add_argument("--sim-repo",required=True)
p.add_argument("--config",default="configs/deploy_isaacsim.json")
p.add_argument("--output",default=None, help="Episode directory; defaults to PROJECT/runs/deployments/isaac_episode_TIMESTAMP")
p.add_argument("--keep-open", action="store_true", help="Keep GUI open after a completed episode")
p.add_argument("--seconds",type=float,default=30)
p.add_argument("--observation-hz",type=float,default=30)
p.add_argument("--action-hz",type=float,default=15)
p.add_argument("--execute-steps",type=int,default=1)
p.add_argument("--camera",default="front_camera")
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(p)
a=p.parse_args()
a.enable_cameras=True
deployment_output_root = Path(__file__).resolve().parents[1] / "runs" / "deployments"
a.output=str(Path(a.output or (deployment_output_root / ("isaac_episode_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")))).resolve())
# Preserve launch-directory path semantics before the simulator changes cwd.
policy_base_dir = Path.cwd().resolve()
a.config = str(Path(a.config).resolve())
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
root=Path(a.sim_repo).resolve();sys.path.insert(0,str(root));os.environ["PROJECT_ROOT"]=str(root)
from deployment_script.evaluate_basket_criteria import (
    RULE, RULE_VERSION, BasketEntryTracker, evaluation_group,
    publish_episode_result, write_json_atomic,
)
# Fail before creating a GUI when paths or output are invalid.
if not root.is_dir(): raise FileNotFoundError(f"Simulator repository not found: {root}")
if not Path(a.config).is_file(): raise FileNotFoundError(f"Deployment config not found: {a.config}")
if a.seconds <= 0 or a.execute_steps < 1: raise ValueError("Invalid duration/execute-steps")
out = Path(a.output)
out.mkdir(parents=True, exist_ok=True)
if any(out.iterdir()): raise ValueError(f"Choose an empty episode output directory: {out}")
fault_log = (out / "native_fault.log").open("w", encoding="utf-8")
faulthandler.enable(file=fault_log, all_threads=True)
stage = "preflight"
def mark(value):
    global stage
    stage = value
    message = f"[{datetime.now().isoformat(timespec='seconds')}] {value}"
    print(message, flush=True)
    with (out / "startup.log").open("a", encoding="utf-8") as f:
        f.write(message + "\n")

app = None

def main():
    mark("import_simulator_dependencies")
    import cv2
    import gymnasium as gym
    import numpy as np
    import torch
    import tasks
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from tools.data_convert import convert_to_gripper_range, convert_to_joint_range
    if a.seconds<=0 or a.execute_steps<1: raise ValueError("Invalid duration/execute-steps")
    mark("load_student_and_encoders")
    from deployment_script.deployment import Policy
    policy_cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    from deployment_script.gripper_calibration import GripperCalibration
    calibration = GripperCalibration(policy_cfg.get("gripper_calibration", "simulator_v2"),
        convert_to_joint_range, convert_to_gripper_range)
    convert_to_joint_range = calibration.action_to_joint
    convert_to_gripper_range = calibration.joint_to_state
    (out / "policy_config.json").write_text(json.dumps(policy_cfg, indent=2), encoding="utf-8")
    from deployment_script.environment_perturbations import EnvironmentPerturbations
    perturbations = EnvironmentPerturbations(policy_cfg.get("environment_perturbations"))
    initial_object_limits = {}
    for name, default in (("max_initial_object_tilt_degrees", 30.0),
                          ("max_initial_object_xy_displacement", None),
                          ("max_initial_object_linear_speed", None)):
        value = policy_cfg.get(name, default)
        if value is not None:
            value = float(value)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        initial_object_limits[name] = value
    policy = Policy(policy_cfg, base_dir=policy_base_dir)
    print("Student loaded in Isaac Lab process", flush=True)
    dt=1/120
    def ticks(hz):
        n=round(1/(dt*hz)) if hz>0 else 0
        if n<1 or abs(n*dt*hz-1)>1e-6: raise ValueError("Rates must divide 120 Hz exactly")
        return n
    obs_ticks,action_ticks=ticks(a.observation_hz),ticks(a.action_hz)
    mark("parse_scene_config")
    cfg=parse_env_cfg("Isaac-PickPlace-Cylinder-G129-Dex1-Joint",device=a.device,num_envs=1)
    cfg.env_name="Isaac-PickPlace-Cylinder-G129-Dex1-Joint"
    cfg.sim.dt=dt;cfg.decimation=1;cfg.sim.render_interval=1
    cfg.actions.joint_pos.use_default_offset=False
    # Table surface ~0.794 m plus cylinder half-height 0.175 m and ~1 mm clearance.
    object_spawn_height = policy_cfg.get("object_spawn_height")
    if object_spawn_height is not None:
        object_spawn_height = float(object_spawn_height)
        if not np.isfinite(object_spawn_height) or object_spawn_height <= 0:
            raise ValueError("object_spawn_height must be finite and positive")
        xyz = list(cfg.scene.object.init_state.pos)
        xyz[2] = object_spawn_height
        cfg.scene.object.init_state.pos = tuple(xyz)
        print(f"Initial cylinder center height: {object_spawn_height:.3f} m", flush=True)
    native_object_reset_range = dict(cfg.events.reset_object.params["pose_range"])
    object_reset_range = policy_cfg.get("object_reset_pose_range")
    if object_reset_range is not None:
        if not isinstance(object_reset_range, dict) or set(object_reset_range) != {"x", "y"}:
            raise ValueError("object_reset_pose_range must specify x and y offsets")
        validated_range = {}
        for axis, bounds in object_reset_range.items():
            bounds = np.asarray(bounds, dtype=float)
            if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] > bounds[1]:
                raise ValueError(f"Invalid object reset offset range for {axis}")
            validated_range[axis] = bounds.tolist()
        cfg.events.reset_object.params["pose_range"] = validated_range
        print(f"Object reset offsets (m): {validated_range}", flush=True)
    perturbations.configure_scene(cfg)
    if perturbations.seed is not None:
        print(f"Environment perturbations: {json.dumps(perturbations.metadata(), ensure_ascii=False)}", flush=True)
        print("Effective object reset offsets (m):", cfg.events.reset_object.params["pose_range"], flush=True)
    gripper_joint_friction = float(policy_cfg.get("gripper_joint_friction", 0.0))
    if not np.isfinite(gripper_joint_friction) or gripper_joint_friction < 0:
        raise ValueError("gripper_joint_friction must be finite and nonnegative")
    cfg.scene.robot.actuators["hands"].friction = gripper_joint_friction
    print(f"Gripper joint friction: {gripper_joint_friction}", flush=True)
    # The upstream 'success' is an out-of-bounds reset, not task success.
    # Avoid automatic resets during this bounded-duration scored episode.
    cfg.terminations.success=None
    getattr(cfg.scene, a.camera).update_period=0.0
    arms=[side+"_"+joint+"_joint" for side in ("left","right") for joint in
          ("shoulder_pitch","shoulder_roll","shoulder_yaw","elbow","wrist_roll","wrist_pitch","wrist_yaw")]
    fingers=["left_hand_Joint1_1","left_hand_Joint2_1","right_hand_Joint1_1","right_hand_Joint2_1"]
    initial_state = policy_cfg.get("initial_state")
    # Measured joint positions differ from the PD targets that hold them under gravity.
    initial_arm_target = policy_cfg.get("initial_arm_target")
    if initial_arm_target is not None:
        initial_arm_target = np.asarray(initial_arm_target, dtype=np.float32)
        if initial_arm_target.shape != (14,) or not np.isfinite(initial_arm_target).all():
            raise ValueError("initial_arm_target must contain 14 finite arm targets in dataset order")
    settling_seconds = float(policy_cfg.get("settling_seconds", 0.0))
    if not np.isfinite(settling_seconds) or not 0 <= settling_seconds <= 5:
        raise ValueError("settling_seconds must be between 0 and 5")
    if initial_state is not None:
        initial_state = np.asarray(initial_state, dtype=np.float32)
        if initial_state.shape != (16,) or not np.isfinite(initial_state).all():
            raise ValueError("initial_state must contain 16 finite values in dataset order")
        if np.any(initial_state[14:] < 0) or np.any(initial_state[14:] > calibration.state_max):
            raise ValueError(f"Initial gripper state must be between 0 and {calibration.state_max}")
        positions = dict(cfg.scene.robot.init_state.joint_pos)
        positions.update(zip(arms, map(float, initial_state[:14])))
        for side, value in enumerate(initial_state[14:]):
            for name in fingers[side*2:side*2+2]:
                positions[name] = calibration.initial_joint_from_state(float(value))
        cfg.scene.robot.init_state.joint_pos = positions
    mark("create_scene")
    env=gym.make("Isaac-PickPlace-Cylinder-G129-Dex1-Joint",cfg=cfg).unwrapped
    writer=None
    perturbation_log = None
    rollout_steps = 0
    live_log = (out / "trajectory.jsonl").open("w", encoding="utf-8", buffering=1)
    log=[]
    summary={"status":"started", "success":None, "evaluation_started":False,
             "success_note":"Any cylinder part inside basket interior at any rollout step; no release, dwell or full containment required",
             "basket_evaluation":{"rule_version":RULE_VERSION, "criterion":RULE},
             "arguments":vars(a),
             "environment_perturbations":perturbations.metadata()}
    published = False
    try:
        mark("reset_scene")
        env.reset();robot=env.scene["robot"]
        reset_object_state = env.scene["object"].data.root_state_w[0].detach().cpu().tolist()
        if perturbations.seed is not None:
            view = env.scene["object"].root_physx_view
            readback = {"masses_kg": view.get_masses().detach().cpu().tolist(),
                        "inertias_kg_m2": view.get_inertias().detach().cpu().tolist(),
                        "materials_static_dynamic_restitution": view.get_material_properties().detach().cpu().tolist()}
            summary["environment_perturbations"]["physics_readback"] = readback
            perturbation_log = (out / "perturbation_events.jsonl").open("w", encoding="utf-8", buffering=1)
        mark("prepare_basket_evaluation")
        import omni.usd
        from deployment_script.basket_scene_geometry import basket_interior_from_usd
        basket_geometry = basket_interior_from_usd(omni.usd.get_context().get_stage())
        cylinder_axis = {"X": (1.,0.,0.), "Y": (0.,1.,0.), "Z": (0.,0.,1.)}[
            str(getattr(cfg.scene.object.spawn, "axis", "Z")).upper()]
        basket_tracker = BasketEntryTracker(basket_geometry,
            length=float(cfg.scene.object.spawn.height), radius=float(cfg.scene.object.spawn.radius),
            local_axis=cylinder_axis)
        summary.update(evaluation_started=True, success=False,
            basket_evaluation=basket_tracker.metadata(),
            evaluation_group=evaluation_group(policy_cfg, policy.cfg["checkpoint"],
                policy.cfg["export_config"], vars(a), basket_tracker))
        write_json_atomic(out / "summary.json", summary)
        names=robot.data.joint_names
        arm_idx=[names.index(x) for x in arms];finger_idx=[names.index(x) for x in fingers]
        target=robot.data.default_joint_pos.clone()
        command_limits = robot.data.soft_joint_pos_limits[0].clone()
        command_limits[finger_idx] = robot.data.joint_pos_limits[0, finger_idx]
        calibration.validate_physical_limits(command_limits[finger_idx].detach().cpu().tolist())
        summary["gripper_calibration"] = calibration.metadata()
        if initial_arm_target is not None:
            target[0, arm_idx] = torch.as_tensor(initial_arm_target, device=env.device)
            target[0, arm_idx] = torch.maximum(torch.minimum(target[0, arm_idx], command_limits[arm_idx, 1]), command_limits[arm_idx, 0])
        mark("write_initial_robot_state" if initial_state is not None else "use_scene_initial_robot_state")
        authored_joint_positions = robot.data.joint_pos[0].detach().cpu().tolist()
        # This task resets the object only. Write joints only for an explicit pose;
        # otherwise retain the scene's robot state and hold its default targets.
        with torch.inference_mode():
            if initial_state is not None:
                robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(),
                                               robot.data.default_joint_vel.clone())
            robot.set_joint_position_target(target)
            robot.set_joint_velocity_target(torch.zeros_like(target))
            env.scene.write_data_to_sim()
            env.sim.forward()
            env.scene.update(0.0)
        written_joint_positions = robot.data.joint_pos[0].detach().cpu().tolist()
        summary["gripper_control"] = {"joint_friction": gripper_joint_friction,
            "limit_mode": "physical_limits_for_fingers_soft_limits_for_other_joints",
            "command_source": "model",
            "finger_limits": command_limits[finger_idx].detach().cpu().tolist()}
        summary["initialization"] = {
            "requested_state": initial_state.tolist() if initial_state is not None else None,
            "joint_state_written": initial_state is not None,
            "joint_positions_before_write": authored_joint_positions,
            "joint_positions_after_write": written_joint_positions,
            "arm_hold_target": target[0, arm_idx].detach().cpu().tolist(),
            "settling_seconds": settling_seconds,
            "reference": policy_cfg.get("initial_state_reference"),
            "object_spawn_height": object_spawn_height,
            "native_object_reset_pose_range": native_object_reset_range,
            "object_reset_pose_range": cfg.events.reset_object.params["pose_range"],
            "object_reset_reference": ("environment_perturbations.position: uniform XY reset offsets"
                if perturbations.position_range is not None else policy_cfg.get("object_reset_reference")),
            "object_after_reset": reset_object_state,
        }
        if settling_seconds:
            mark("settle_initial_pose")
            with torch.inference_mode():
                for _ in range(round(settling_seconds / dt)):
                    if not app.is_running():
                        raise RuntimeError("Simulation closed during initialization")
                    env.step(target)
        summary["initialization"]["actual_joint_positions"] = robot.data.joint_pos[0].detach().cpu().tolist()
        initial_object_state = env.scene["object"].data.root_state_w[0].detach().cpu().tolist()
        summary["initialization"]["object_after_settling"] = initial_object_state
        if policy_cfg.get("require_upright_start", False):
            mark("validate_initial_object")
            quat = np.asarray(initial_object_state[3:7], dtype=float)
            norm = float(np.linalg.norm(quat))
            if not np.isfinite(initial_object_state).all() or not np.isfinite(norm) or norm < 1e-12:
                raise RuntimeError("Invalid cylinder state before the first policy action")
            quat /= norm
            axis_z = abs(1.0 - 2.0 * (quat[1]**2 + quat[2]**2))
            tilt = float(np.degrees(np.arccos(np.clip(axis_z, 0.0, 1.0))))
            xy_drift = float(np.linalg.norm(np.asarray(initial_object_state[:2]) - reset_object_state[:2]))
            speed = float(np.linalg.norm(initial_object_state[7:10]))
            summary["initialization"].update(object_tilt_degrees=tilt,
                object_xy_displacement=xy_drift, object_linear_speed=speed,
                acceptance_limits=initial_object_limits)
            violations = []
            for name, value in (("max_initial_object_tilt_degrees", tilt),
                                ("max_initial_object_xy_displacement", xy_drift),
                                ("max_initial_object_linear_speed", speed)):
                limit = initial_object_limits[name]
                if limit is not None and value > limit:
                    violations.append(f"{name}: measured {value:.6g}, limit {limit:.6g}")
            if violations:
                raise RuntimeError("Cylinder disturbed before the first policy action: " + "; ".join(violations)
                    + "; inspect initialization in summary.json")
        history=deque(maxlen=2);pending=deque()
        mark("rollout")
        interrupted = False
        for step in range(round(a.seconds/dt)):
            if not app.is_running():
                interrupted = True
                break
            if step%obs_ticks==0:
                env.sim.render()
                image=env.scene[a.camera].data.output["rgb"][0,:,:,:3].detach().cpu().numpy().astype(np.uint8)
                history.append(image.copy())
                if step % 120 == 0:
                    cv2.imwrite(str(out / "latest_rgb.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                if writer is None:
                    writer=cv2.VideoWriter(str(out/"episode.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),a.observation_hz,(image.shape[1],image.shape[0]))
                    if not writer.isOpened(): raise RuntimeError("Cannot create episode.mp4")
                writer.write(cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
            if step%action_ticks==0 and len(history)==2:
                object_state = env.scene["object"].data.root_state_w[0].detach().cpu().tolist()
                q=robot.data.joint_pos[0]
                state=q[arm_idx].detach().cpu().tolist()+[convert_to_gripper_range(q[finger_idx[0]].item()),convert_to_gripper_range(q[finger_idx[2]].item())]
                if not pending:
                    if not log: mark("first_policy_prediction")
                    actions=policy.predict(list(history), state)
                    if not log: mark("rollout")
                    if actions.ndim!=2 or actions.shape[0]<1 or actions.shape[1]!=16 or not np.isfinite(actions).all(): raise ValueError("Invalid policy actions")
                    pending.extend(action.copy() for action in actions[:a.execute_steps])
                command = pending.popleft()
                applied_left = float(command[14])
                target[0,arm_idx]=torch.as_tensor(command[:14],device=env.device)
                for offset,value in ((0,command[14]),(2,command[15])):
                    target[0,finger_idx[offset:offset+2]]=convert_to_joint_range(float(value))
                limits=command_limits
                target[0]=torch.maximum(torch.minimum(target[0],limits[:,1]),limits[:,0])
                log.append({"simulation_time":step*dt,"state":state,"requested_action":command.tolist(),"finger_joint_positions":q[finger_idx].detach().cpu().tolist(),"applied_left_gripper":applied_left,"joint_target":target[0].detach().cpu().tolist(),"object_position":object_state[:3],"object_orientation_wxyz":object_state[3:7],"object_linear_velocity":object_state[7:10],"object_angular_velocity":object_state[10:13]})
                live_log.write(json.dumps(log[-1]) + "\n")
                if len(log) == 1 or step % 120 == 0:
                    obj = log[-1]["object_position"]
                    print(f"[control] t={step*dt:.2f}s object={obj} gripper_state={state[14:]} gripper_command={command[14:].tolist()} applied_left={applied_left:.3f}", flush=True)
            changes = perturbations.apply_force(env.scene["object"], step * dt, dt)
            for event in changes:
                print("[perturbation]", json.dumps(event), flush=True)
                if perturbation_log is not None:
                    perturbation_log.write(json.dumps(event) + "\n")
            if changes:
                write_json_atomic(out / "summary.json", summary)
            with torch.inference_mode(): env.step(target)
            rollout_steps = step + 1
            # Score the post-physics pose, including the final step; observation only.
            if not basket_tracker.success:
                pose = env.scene["object"].data.root_state_w[0,:7].detach().cpu().tolist()
                if basket_tracker.update(pose, (step + 1) * dt):
                    summary.update(success=True, basket_evaluation=basket_tracker.metadata())
                    write_json_atomic(out / "summary.json", summary)
                    print(f"[basket] entered at {(step + 1)*dt:.3f}s; success latched", flush=True)
        summary.update(status="interrupted" if interrupted else "completed", policy_steps=len(log))
        mark(f"episode_{summary['status']}: {len(log)} policy steps; output={out}")
        (out/"trajectory.json").write_text(json.dumps(log),encoding="utf-8")
        if writer is not None:
            writer.release()
            writer = None
        if a.keep_open and not a.headless and not interrupted:
            publish_episode_result(out, summary)
            published = True
            mark("episode_finished_close_window_to_exit")
            while app.is_running(): app.update()
    except Exception as exc:
        summary.update(status="failed",error=repr(exc));raise
    finally:
        end_events = perturbations.finish(rollout_steps * dt)
        if perturbation_log is not None:
            for event in end_events:
                perturbation_log.write(json.dumps(event) + "\n")
            perturbation_log.close()
        live_log.close()
        if writer is not None: writer.release()
        summary["policy_steps"] = len(log)
        publish_episode_result(out, summary, print_result=not published)
        (out/"trajectory.json").write_text(json.dumps(log),encoding="utf-8")
        env.close()

try:
    os.chdir(root)
    mark("import_pinocchio_before_isaac")
    # Match the official sim_main.py import order.
    import pinocchio
    mark("launch_isaac")
    launcher = AppLauncher(a)
    app = launcher.app
    main()
except BaseException as exc:
    trace = traceback.format_exc()
    (out / "error.log").write_text(trace, encoding="utf-8")
    report_path = out / "summary.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    report.update(status="failed", stage=stage, error=repr(exc), arguments=vars(a))
    report.setdefault("evaluation_started", False)
    report.setdefault("success", None)
    report.setdefault("basket_evaluation", {"rule_version":RULE_VERSION, "criterion":RULE})
    publish_episode_result(out, report, print_result=False)
    print(f"Deployment failed during {stage}. See {out / 'error.log'}", file=sys.stderr, flush=True)
    raise
finally:
    try:
        if app is not None: app.close()
    finally:
        faulthandler.disable()
        fault_log.close()
