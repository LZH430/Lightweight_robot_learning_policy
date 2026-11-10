# Gripper decision fine-tuning

The configuration preserves every original training setting and loss weight from
`configs/backups/gripper_finetune_64clips.before_left_grasp_20261029_202008_434872.json`,
except the independent output directory. The only added setting is
`left_grasp_action_weight: 0.5`.

Original weights remain: first action 0.5, gripper actions 0.25, opening/closing
transitions 0.5, initial-action branch 0.5, and adaptive dynamics teacher 0.1.
All original full-task targets and all policy parameters remain in training.

The extra loss approximately emphasizes the left approach and closing target,
using only cached measured hand states and demonstration actions. An open measured
left hand with a closure in the target chunk marks the prefix through that closing
target (inclusive). Extra supervision applies to left-arm coordinates 0..6 and
left-gripper coordinate 14. It does not extend into subsequent holding/lifting.
Receiver closure before left closure or a possible held right hand excludes handoff.
The original losses still cover holding, lifting, release, handoff and placement.

This is an approximation, not a cylinder-contact label. Approach earlier than the
available chunk and partially closed hands can be missed; empty grasps may be
misclassified. No phase file, raw-data lookup, new export step or runtime helper
is used. Setting the additional weight to zero gives the original objective.

```bash
cd /home/lzh/unifolm_student_policy
PYTHONPATH=src:. python -u -m student_policy.train_dynamics \
  --config configs/gripper_finetune_64clips.json
```

Initialization: `runs/wma_adaptive_gated_64clips/best.ckpt`.
Defaults: 20 epochs, learning rate 1e-5, batch size 4.
Output: `runs/wma_gated_grasp_finetune_64clips`.
Previous checkpoints are retained; no trained model is automatically deployed.
