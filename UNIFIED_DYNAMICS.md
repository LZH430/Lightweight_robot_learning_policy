# Unified WMA future distillation (dynamics_v2)

Five distinct experiments share the same cache samples and encoder features:

- student.json: pure action supervision, no future branch, fusion or distillation. Uses train.py and outputs runs/action_only_student.
- dynamics_distill.json: fixed-weight WMA dynamics distillation and ungated fusion.
- gated_future.json: dynamics distillation with gated fusion.
- adaptive_distill.json: adaptive dynamics distillation with ungated fusion.
- adaptive_gated.json: adaptive dynamics distillation with gated fusion.

Commands:

python -m student_policy.train --config configs/student.json
python -m student_policy.train_dynamics --config configs/dynamics_distill.json
python -m student_policy.train_dynamics --config configs/gated_future.json
python -m student_policy.train_dynamics --config configs/adaptive_distill.json
python -m student_policy.train_dynamics --config configs/adaptive_gated.json

Among the four dynamics models, the switches are model.gated_future and adaptive_dynamics. dynamics_distill.json is no longer an alias of student.json. The pure action baseline uses only masked demonstration action MSE, logs action_mse, and ignores cached teacher/future labels. No cache re-export is needed. It still reuses frozen encoder features; it is not a randomly initialized end-to-end visual baseline. Existing runs/wma_student results belong to the OLD dynamics baseline, not the new action-only model. Start a fresh run; old dynamics checkpoints cannot resume action-only training. Parameter counts differ, so this is a system-level ablation, not a parameter-matched control.

The following describes the four dynamics models only:

Policy: current inputs -> initial action A0 -> predicted future conditioned on A0 -> attention fusion -> final action A1. Without gating the fusion weight is one; with gating it is learned. Both initial and final actions have real demonstration supervision. Future inputs at policy inference are predictions only.

A separate forward pass predicts dynamics conditioned on the cached dynamics_action, and matches teacher_future for exactly that action. Never match an A0-conditioned future to teacher labels for a different action.

Loss: final masked action MSE + 0.5 * initial masked action MSE + 0.1 * dynamics KD. Adaptive versions weight KD by exp(-teacher_vs_real_future_MSE/temperature), with detached weights and no division by their sum. Real future labels and identical demonstration conditioning actions are required. This is a reliability heuristic, not calibrated confidence. Validation KD is unweighted in every experiment; best.ckpt uses final action_mse.

The inspected upstream WMAData uses actions at start+i*stride and future video frames at start+(i+1)*stride. Teacher terminal frame must follow that same convention. Recheck this alignment when changing the data loader or teacher model. Mixed fps/frame_stride cache conditions are rejected.

Old configs are copied to archive/pre_unified_dynamics. Old dynamics_v1 weights/runs cannot resume this changed architecture; use new output paths. No real WMA or GPU performance is established by synthetic tests. A single refinement does not guarantee A1 has the same predicted outcome as A0, and this is not a planner.
