# Single-process G1 Dex1 student deployment on Linux

Launch from the student repository in your working Isaac Lab Python environment. The script starts the scene and loads the student and frozen image/text encoders in that same process. No HTTP server or second terminal is needed.

## Deployment files

The deployment Python files are grouped in the project-root
`deployment_script/` package:

- `run_isaacsim.py`: Isaac Lab episode launcher.
- `deployment.py`: shared in-process policy inference.
- `deploy_server.py`: optional HTTP adapter.
- `evaluate_basket.py`: repeated independent trials and basket-entry success-rate reporting.
- `evaluate_basket_criteria.py`: entry tracking, per-trial results and cumulative statistics.
- `basket_scene_geometry.py`: physical basket interior measured from the live USD colliders.
- `basket_intersection.py`: finite-cylinder/interior intersection test.

`__init__.py` marks the folder as a Python package. Installed-package imports use
`deployment_script.deployment`; the former `student_policy.deployment` and
`student_policy.deploy_server` module paths have been removed. The episode launcher is `deployment_script/run_isaacsim.py`.

## Configuration

Edit configs/deploy_isaacsim.json: checkpoint (your student best.ckpt), dataset_name, and instruction (exact training/export instruction). export_config supplies the original encoder architecture/weights, schema and normalization statistics. All relative paths are resolved against the launch working directory, including paths inside the export configuration. Absolute Linux paths are recommended for external files.

Supports action_only_v1 and dynamics_v2 checkpoints, including gated/adaptive dynamics variants. Frozen encoders remain necessary; deployment forces enable_dynamics=False and does not run the full WMA teacher sampler. The student checkpoint alone does not include these external encoders/statistics.

## Environment and launch

Use the environment that already runs your Isaac Lab scene. It must also provide torchvision, NumPy, OpenCV, OmegaConf, OpenCLIP and the dependencies imported by the WMA encoders and normalization utilities. Keep Isaac Lab's compatible PyTorch/CUDA installation; do not blindly replace it with the teacher environment's pinned dependencies. The script adds the student src directory automatically; the teacher repo path comes from export_config.

```bash
cd /home/lzh/unifolm_student_policy
python deployment_script/run_isaacsim.py \
  --sim-repo /home/lzh/unitree_sim_isaaclab \
  --config configs/deploy_isaacsim.json \
  --seconds 30 --observation-hz 30 --action-hz 15 --execute-steps 1
```

The checkpoint and instruction placeholders must be replaced before running. Simulator and policy share the process and GPU; 4090 combined memory usage has not been measured. deployment_script/deploy_server.py remains an optional HTTP adapter. Run it from the project root with `PYTHONPATH=src python -m deployment_script.deploy_server --config configs/deploy_isaacsim.json`.

Replace sim-repo with the actual Linux path. D:\unitree_sim_isaaclab is only the Windows location. Do not start sim_main.py simultaneously; the client creates the scene itself. Headless rendering is an explicit option and needs separate validation; use GUI when comparing with interactive deployment. Scene assets must already be installed.

Outputs: episode.mp4 (simulation-time video), trajectory.json (observed state, requested actions, clipped joint targets and object positions), summary.json. Slow inference pauses simulation progression; video does not represent wall-clock throughput. `summary.json` records basket-entry success, the first entry time and pose, scoring geometry, initialization and final status. The upstream term named success only checks object bounds; it is disabled together with automatic resets for this bounded-duration episode. The episode's parent directory also receives `basket_success_rate.json` with cumulative results.

## Basket-entry success rate

Success means that any part of the cylinder has positive overlap with the physical basket interior at any rollout step. The detector uses the cylinder's position, orientation, radius and finite length, and derives the basket's interior from its five USD floor/wall colliders. A cylinder centre outside the basket can still count when a tip or side enters. The interior is inset by 0.1 mm to distinguish entry from wall/rim contact; penetration shallower than that numerical tolerance can be rejected.

The first entry latches success and records its simulation time and cylinder pose. Release, dwell and full containment are not required. The detector checks the complete rollout at 120 Hz, including the final physics step. Scoring does not change model actions or stop an episode early.

Run 50 GUI trials from the student repository in the Isaac Lab environment, matching direct deployment:

```bash
cd /home/lzh/unifolm_student_policy
python deployment_script/evaluate_basket.py \
  --sim-repo /home/lzh/unitree_sim_isaaclab \
  --config configs/deploy_isaacsim.json \
  --episodes 50 --gui \
  --seconds 30 --observation-hz 30 --action-hz 15 --execute-steps 1 \
  --output runs/deployments/basket_eval_50
```

Choose a new empty output directory. Each trial runs in its own subprocess, reloads the simulator, student and encoders, and performs a fresh randomized reset. This can take substantial wall-clock time. The batch snapshots the policy configuration and writes `evaluation_config.json`, `episode_0001/summary.json` and subsequent episode directories, plus `console_0001.log` and subsequent trial console logs. Each episode also retains its video, trajectory and startup diagnostics. The batch directory's `basket_success_rate.json` contains successes, scored trials, failures, setup errors and per-condition groups.

The rate is successful scored trials divided by all scored trials. Setup failures before evaluation is ready are reported separately and excluded from that denominator. Once evaluation is ready, startup initialization failures count as unsuccessful trials. Failed or interrupted scored trials also count; an entry already recorded remains successful if a later error occurs. Task failures are not retried. Different model weights, policy configurations and execution settings receive separate groups; use those group rates for comparisons. Older summaries without the current basket scoring rule are ignored. Recounting results does not add duplicate trials.

Rebuild the statistics from existing scored summaries without launching Isaac Sim:

```bash
cd /home/lzh/unifolm_student_policy
python deployment_script/evaluate_basket.py \
  --summarize runs/deployments/basket_eval_50
```

## Mapping and assumptions to verify

Arm joints follow schema: left shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw; then the right arm; then left/right gripper. Exact names/order must pass schema validation. State and actions are normalized/denormalized with upstream WMA functions, not new statistics. The adapter holds uncommanded joints at defaults and configures absolute joint targets (use_default_offset=False). Physical limits clip targets; requested and applied targets are recorded separately.

For the published August 2025 G1 Dex1 cylinder dataset, `configs/deploy_isaacsim.json` selects `gripper_calibration: dataset_v1`. Historical DDS conversion used `q_target = .03 - .05 * clip(action, 0, 5.6) / 5.6` and measured feedback `5.6 * (.03 - q_actual) / .05`, without rounding. The runner still clips commanded targets to the physical finger limits `[-.02, .0245]`. These measured-state units differ from the current simulator's `5.4/.024` conversion; at the closed physical limit the dataset reports approximately `.616`, and the open limit reports `5.6`. Initial measured gripper states are inverted using this same calibration, so initial `.616` reconstructs physical `.0245`. Arm initialization and model action selection are unchanged. Other configs default to `simulator_v2`, which delegates to the current `tools/data_convert.py`; choose the profile matching the dataset rather than changing a gripper timing rule. The selected profile and raw finger positions are recorded in the episode outputs. See `GRIPPER_CALIBRATION.md` for source history and regression evidence. Front camera is selected by default; choose --camera only to match the data camera. Image processing matches resize_center_crop then [-1,1]. Two observations are captured at 30Hz and actions executed at 15Hz by default, consistent only with a 30Hz source and stride 2. Confirm original source rate; teacher conditioning fps alone is not proof. Physics runs at 120Hz for exact timing. One action is executed before replanning by default; execution may be shorter than the trained horizon.

Verify idle posture, camera view, joint directions and gripper mapping before comparing policies. The reported metric measures basket entry under the recorded initialization and execution settings; it does not require a completed placement or a released cylinder.

## Startup diagnostics

The launcher now imports pinocchio before creating AppLauncher, matching the official sim_main.py. Without --output it creates a timestamped subdirectory under PROJECT/runs/deployments, regardless of the launch directory. An explicitly supplied nonempty directory is rejected before GUI startup.

startup.log records startup stages; error.log and summary.json record caught Python failures, including model loading and scene construction. native_fault.log captures supported fatal signals via Python faulthandler; OS kills and some GPU/native failures may still require terminal logs.

The default episode ends after 30 simulated seconds and closes the app normally. Add --keep-open to keep the GUI open after completion (no further policy actions). This does not recover an initialization failure. Use python -u -X faulthandler and tee to retain full Isaac terminal logs.
