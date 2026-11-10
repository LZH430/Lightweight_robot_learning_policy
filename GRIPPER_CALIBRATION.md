# Dex1 dataset conversion provenance

Read-only investigation of local primary source history, 2026-10-27. No simulator or inference was launched for this investigation.

## Historical executable conversion

Simulator checkout: `/home/lzh/unitree_sim_isaaclab`.

The parent of conversion-change commit `b075304e7891ec11fd6f142d4816b3321e5dcc63` is `9101b548334b9cbfbb30f208e70ef73b75b3a2cd` (2025-09-02). Its `dds/gripper_dds.py` contains actual inline methods:

- Lines 237–264, `convert_to_joint_range`: input range 0..5.6; physical PD target range .03..−.02.
- Lines 266–293, `convert_to_gripper_range`: physical position range −.02...03; reported state range 5.6..0; no rounding.
- `_update_gripper_state` calls that feedback method before publishing DDS state.

The executable formulas are:

```python
q_target = 0.03 - 0.05 * clip(action, 0.0, 5.6) / 5.6
state = 5.6 * (0.03 - clip(q_actual, -0.02, 0.03)) / 0.05
```

`b075304`, dated 2025-09-04 17:42:48 +08:00, removes the inline methods, introduces `tools/data_convert.py`, and changes executable constants to 5.4/.024, with position rounding to three decimal places and feedback rounding. The old constants remaining in current docstrings are not the evidence for this conclusion; the historical executable methods and diff are.

Reproduce without checking out or changing files:

```bash
git -C /home/lzh/unitree_sim_isaaclab show 9101b548:dds/gripper_dds.py
git -C /home/lzh/unitree_sim_isaaclab show b075304 -- dds/gripper_dds.py tools/data_convert.py
```

## Dataset chronology and unchanged data

Dataset checkout: `/home/lzh/G1_Dex1_PickPlaceCylinder_Dataset_Sim`.

- Original upload: `1df63c16eb0665a706dfeb60a5e7040a22f91bfc`, 2025-08-28 03:15:36 UTC.
- README update: `6228802dcc43fcc317c57ce22c2a47141c0ee7e4`, 2025-08-28 03:20:18 UTC. It explicitly identifies `unitree_sim_isaaclab` and task `Isaac-PickPlace-Cylinder-G129-Dex1-Joint`.
- The dataset upload predates the executable mapping change by seven days.
- `data/chunk-000/episode_000047.parquet` SHA256 matches its original upload commit's Git LFS object: `e37e03abd0f5b114a5ab1ab14a56195497d7d7f118ea17b8e554ef9dc5f6e0b8` (79,903 bytes). The local example is therefore the original uploaded object.
- Dataset metadata specifies LeRobot `codebase_version: v2.1`, but does not pin the exact simulator commit.

The local `unitree_deploy` recorder path preserves received DDS gripper observations: `robot_devices/endeffector/gripper.py:115` copies message `q` unchanged; `read_current_endeffector_q` at lines 194–197 returns that value; `robot/robot.py` `capture_observation` concatenates it into `observation.state`.

## Physical endpoint consistency

The loaded Dex1 physical finger limits are approximately `[-0.019999999553, 0.024499999359]` metres. Historical conversion explains the recorded action and measured-state ranges together:

| Quantity | Historical conversion result |
| --- | --- |
| Action 0 | PD target q=.03, clipped by physical upper limit |
| Actual q=.024499999359 | State .616000071764, matching the recorded float32 .616000056267 floor |
| Action 5.4 | PD target q=−.0182142857143; corresponding feedback 5.4 |
| Actual q=−.02 | State 5.6 |

An initial measured state of .616 therefore reconstructs physical q=.0245. It is distinct from a fresh action 0, whose historical PD target remains .03 before the runtime physical-limit clamp.

## Limits of the conclusion

The executable historical code, original upload chronology, unchanged data object and endpoint correspondence strongly support the dataset's legacy conversion convention. An exact collection simulator commit is not pinned. This evidence establishes conversion provenance; it does not by itself establish a new success rate or prove that calibration explains every release failure. Model output remains the source of arm and gripper decisions.

## Deployment integration

`configs/deploy_isaacsim.json` selects `dataset_v1`. Other configurations retain the previous conversion unless that profile is explicitly selected. `deployment_script/gripper_calibration.py` keeps action conversion, measured state conversion and initialization inversion explicit; it validates the loaded Dex1 physical finger limits. The existing physical clipping, policy inference rate, first-action execution, arm targets and randomized reset are retained. No timer, forced opening, object-state control rule or action-chunk change is introduced.

The raw 16D model command remains the source of grasp/release decisions. `summary.json` records the conversion profile/provenance and `trajectory.json[l]` additionally records all four raw finger joint positions. `dataset_v1` removes the newer feedback quantization so inputs use the precision of the recorded observations. The calibration profile is part of policy configuration, so evaluation groups separate it from older runs.

## Regression validation (2026-10-27)

Twelve calibration unit tests pass. With the original finetuned checkpoint, the latest evaluation episode 28 object reset pose was reproduced exactly: `[-0.39630162715911865, 0.4303821325302124, .8399999737739563]`. The simulator_v2 control completed 30 seconds without any left command >=4 after 3 seconds and without basket entry. The dataset_v1 run produced a model opening command at 8.8 seconds and first basket entry at 9.858333 seconds. This is a controlled failure regression at one object position, not a population success-rate estimate. Finger initialization follows each tested calibration; arm initialization, object reset pose, checkpoint and control cadence are unchanged.

An isolated five-epoch first-step gripper-loss experiment was also tested with the existing cached split. It retained the same 2/49 held-out first-left-opening misses, so its checkpoint and training changes were not selected for deployment. The installed calibration repair continues to use the original checkpoint.

A second earlier stalled position (latest evaluation episode 23) was tested with the same original checkpoint. The calibrated model issued a post-grasp opening command and first entered the basket interior at 5.225 seconds. Its subsequent trajectory left the basket and fell outside; the existing user-selected criterion counts any cylinder part entering at any rollout step, so this remains an entry success, not evidence of stable placement. These two release regressions do not establish an overall task success rate.
