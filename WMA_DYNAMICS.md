> Current four-model commands and architecture: [UNIFIED_DYNAMICS.md](UNIFIED_DYNAMICS.md). Earlier commands below describe prior experiments.

# Action-conditioned WMA export

configs/export.json enables teacher.enable_dynamics and writes a new cache at cache/cylinder_wma_dynamics_16d. BaseEncoders reuses the full frozen teacher encoders (no duplicate encoder instance), loads the full checkpoint strictly, and samples with action_projector conditions and sim_mode=True following the upstream interaction script.

Set actual teacher checkpoint/model_config/repo paths. The architecture, action width and observation history must match the checkpoint. Run the normal export command in the WMA dependency environment:

python -m student_policy.export_cache --config configs/export.json

New fields: teacher_future (predicted terminal visual features), dynamics_action (the exact normalized conditioning actions), dynamics_fs/type (sampling rate condition), dynamics_target_frame (terminal frame index). Existing future remains the real-data target. Only observations, state, instruction and specified actions enter WMA prediction; real future frames are never passed to prediction.

This path loads the full frozen WMA and decodes predicted video before extracting terminal features. Single-4090 feasibility and real checkpoint predictions have NOT been verified. Disabling teacher.enable_dynamics restores encoder-only loading. Use an empty separate output for each export. This is stochastic sampling despite eta=0 because initial noise is random; export seeds are recorded in the manifest.

Current student losses do not yet consume teacher_future or dynamics_action. This change supplies teacher dynamics targets; an action-conditioned student predictor and dynamics distillation loss remain the next step. Check temporal alignment in the source dataset before supervising real dynamic outcomes. Predicted future features do not establish task success.

## Student dynamics distillation (implemented)

Run: python -m student_policy.train_dynamics --config configs/dynamics_distill.json

New model_dynamics.py shares visual/text/state projections with the policy, and adds action-position embeddings and a terminal-feature decoder. predict_dynamics(visual,text,state,actions) compares specified action sequences without future labels. Policy forward/predict still uses current inputs only. This is joint representation learning, not yet planning or gated dynamic rollout selection.

The loss is real-action masked MSE plus dynamics_teacher_weight times terminal WMA feature MSE. Best checkpoint selection remains validation action_mse; dynamics_teacher_mse is logged independently. Optional dynamics_real_weight defaults to zero pending source temporal alignment verification; enabling it requires identical demonstration and conditioning actions.

Set dynamics_fs and dynamics_fs_type to the actual cache values (fps versus frame_stride follows the teacher YAML). Defaults are null: infer one fixed condition from the first training cache, save it in checkpoint config, and validate all training/validation samples against it. Mixed sampling rates are rejected. Existing encoder-only caches are rejected. Use fresh runs; original policy checkpoints have incompatible architecture. Outputs are best.ckpt, last.ckpt and metrics.jsonl. This implementation supersedes the earlier note that student training does not consume teacher_future. Real checkpoint/GPU/task validation remains outstanding.
