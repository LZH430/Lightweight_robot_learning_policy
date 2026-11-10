# Cylinder data + frozen Base encoders + student training

The standard export.json, student.json and gated_future.json now use the Base encoder workflow. Existing caches and runs are unchanged.
No WMA diffusion backbone, VAE or action head is instantiated in base_encoders mode.
Only embedder and cond_stage_model are instantiated and strictly loaded from Base.
Missing encoder weights cause an error; random or downloaded fallback is forbidden.
The checkpoint is read on CPU (mmap when supported). CPU RAM is still needed;
OpenCLIP constructors temporarily allocate unused components on CPU.
GPU holds frozen encoders only; image_batch_size=1 limits extraction activations.
Student training uses caches and loads neither encoder nor teacher.

## Preparation

Download G1_Dex1_PickPlaceCylinder_Dataset_Sim and convert using the WMA preparation
script, choosing unitree_g1_pickplace_cylinder as the local dataset directory/name.
Keep the original ordered 16D action/state metadata and matching normalization.
Verify actual metadata before proceeding; dimensions alone do not establish semantics.
Update configs/export.json: teacher.repo, teacher.model_config,
teacher.checkpoint (actual Base file), data_dir, transition_dir and schema_sources.
The model YAML provides encoder architecture and text layer (normally penultimate).
Use the same encoder weights, preprocessing and normalization for deployment.
No real cylinder data or Base checkpoint has been downloaded by this change.

## Commands (from repository root)

```bash
python -m student_policy.data_split --csv /path/to/trainingdata/unitree_g1_pickplace_cylinder.csv --dataset-name unitree_g1_pickplace_cylinder
python -m student_policy.action_schema --config configs/export.json
python -m student_policy.export_cache --config configs/export.json
python -m student_policy.train --config configs/student.json
python -m student_policy.train_adaptive --config configs/gated_future.json
```

Split/check only require Python. Export requires the existing WMA dependency
environment (OpenCLIP, Kornia, OmegaConf and WMAData dependencies). Training uses
the student environment. Switch environments between export and training.

Training exports 8 random clips per episode; validation exports 1. Clips may
repeat or overlap within a split; this is increased sampling, not exhaustive
coverage. Episode splits remain separate. Do not reuse caches after changing
encoder weights or normalization. Export refuses a nonempty output directory.

Teacher actions and adaptive KD are disabled. Targets are dataset actions and
features of the real future image. Outputs: runs/cylinder_student and
runs/cylinder_gated_future, with best.ckpt / last.ckpt. The predicted horizon is
16 time steps, each action has 16 dimensions. This does not transfer the Base
world model dynamics into the student: the future branch learns from data.

Actual 4090 memory, real Base key compatibility and task success need validation
with the user's files. This change does not yet implement Isaac Sim deployment
or success counting. Held-out validation is not an independent final test set.
