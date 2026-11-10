# 16-dimensional action and state configuration

Aligned with upstream configs/train/config.yaml (agent_action_dim=16,
agent_state_dim=16) and configs/train/meta.json (action and agent_pos shape [16]).
All student training configs now use action_dim=16 and state_dim=16.
The horizon remains 16: output shape is [batch, 16 time steps, 16 action dimensions].

New exports: cache/teacher_model_16d/{train,val}.
New training outputs use the _16d suffix. Existing caches and runs are untouched.
Use a matching 16-dimensional teacher checkpoint and teacher architecture config.
The checkpoint path is unchanged; replace it with your new checkpoint path.

Existing dataset paths are retained. Before exporting, ensure their action/state
arrays, ordered info.json names and normalization statistics actually describe
16 dimensions. The exporter intentionally rejects old 26-dimensional schemas;
it does not truncate or infer a dexterous-hand-to-gripper conversion.
No checkpoint weights or source datasets were converted.

On the Linux teacher repository, also set agent_action_dim and agent_state_dim
to 16 in configs/train/config.yaml, and both agent_pos/action shapes to [16] in
configs/train/meta.json. Sync these changes before running the export command.

Run in the teacher environment:

    python -m student_policy.action_schema --config configs/export.json
    python -m student_policy.export_cache --config configs/export.json

Then train from scratch using the appropriate original or adaptive student entry
point and its updated config. Do not resume a 26-dimensional student checkpoint.
