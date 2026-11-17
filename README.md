# Lightweight Robot Learning Policy

Description: **Lightweight learning policy training for G1 + Dex1 (16dof) and IsaacSim deployment (Cylinder pick & place task)**

Algorithm: **Real action supervision + WMA future feature distillation**. 

Technical configuration: we use Nvidia 4090 24GB to train it.

***Three main models***:

Action-only baseline model (See `action_only_baseline.json`): This model does not use WMA future prediction. 

Distillation baseline model (See `distillation_baseline.json`): This model uses WMA distillation with constant distillated ratio and does not have future gated function.

Adaptive gated model (See `adaptive_gated.json`): This model has both dynamic distillation ratio and future gated function.

***Two external models***:

Gated model (See `gated_future.json`): This model has future gated function without dynamic distillation function.

Adaptive distillation model (See `adaptive_distillation.json`): This model has dynamic distillation function but does not have future gated function.

| Model | Running | Real action supervision | Future distillation | Gated function | Adaptive |
|---|---|---|---|---|---|
| action_only_baseline.json | lightweight_policy.train | Yes | No | No | No |
| distillation_baseline.json | lightweight_policy.train_dynamics | Yes | Yes | No | No |
| gated_future.json | lightweight_policy.train_dynamics | Yes | Yes | Yes | No |
| adaptive_distillation.json | lightweight_policy.train_dynamics | Yes | Yes | No | Yes |
| adaptive_gated.json | lightweight_policy.train_dynamics | Yes | Yes | Yes | Yes |

## 1. Environment Installation

### Teacher Model Running Environment Installation

```bash
conda create -n teacher_model python==3.10.18
conda activate teacher_model
conda install pinocchio=3.2.0 -c conda-forge -y
conda install ffmpeg=7.1.1 -c conda-forge
# Cloning the repo [Unitree UnifoLM-WMA-0 (https://github.com/unitreerobotics/unifolm-world-model-action)]:
git clone --recurse-submodules https://github.com/unitreerobotics/unifolm-world-model-action.git
cd unifolm-world-model-action
git submodule update --init --recursive
pip install -e .
cd external/dlimp
pip install -e .
```

### Student Model Running Environment Installation

```bash
conda create -n lightweight_model python=3.10 pip -y
conda activate lightweight_model
python -m pip install -e .
python -m pip check
python -c "import torch, lightweight_policy; print(torch.__version__, torch.cuda.is_available())"
```

## 2. Data Processing 

- Degree of freedom 16: Robotic arms (7 joints) + left gripper + right gripper.

First step: Download base checkpoint and dataset

- Base checkpoint

| Model | Link|
|---------|------|
|$\text{UnifoLM-WMA-0}_{Dual}$ | [HuggingFace](https://huggingface.co/unitreerobotics/UnifoLM-WMA-0-Dual)|

- Dataset

| Dataset | Robot | Link |
|---------|-------|------|
|metal_part_sort_v5|[Unitree G1](https://www.unitree.com/g1)|[Huggingface](https://huggingface.co/datasets/unitreerobotics/G1_Dex1_PickPlaceCylinder_Dataset_Sim)|

Note: Replace "your path" in export.json with your own path.

Second step: using `prepare_training_data.py` in Unitree UnifoLM-WMA-0 (https://github.com/unitreerobotics/unifolm-world-model-action) to convert data format.

```bash
conda activate teacher_model
cd unifolm-world-model-action
cd prepare_data
python prepare_training_data.py \
    --source_dir /path/to/your/source_dir \
    --target_dir /path/to/save/the/converted/data \
    --dataset_name "dataset1_name" \
    --robot_name "Unitree G1 Robot with Gripper"
```

Third step: If the dataset includes multiple views, keep main-view camera's values from the `data_dir` column in the CSV file.

Fourth step: Generate corresponded data format to prepare training.

```bash
git clone --recurse-submodules https://github.com/LZH430/Lightweight_robot_learning_policy.git
python -m lightweight_policy.data_split \
  --csv /PATH/TO/CONVERTED_G1.csv \
  --output data/splits \
  --dataset-name unitree_g1_pickplace_cylinder
```
Note: Currently, we set 64 training datasets and 4 verification datasets. 

## 3. Pre-training

```bash
python -m lightweight_policy.action_schema --config configs/export.json
python -m lightweight_policy.export_cache --config configs/export.json
```

Note: Currently, we set 64 training datasets and 4 verification datasets. So, it will spend some times

## 4. Training

In our test, we use the model which contains dynamic distillation and gated future functions.

First step: running the following code to generate 1st version checkpoint. 

```bash
conda activate lightweight_model
python -m lightweight_policy.train_dynamics --config configs/adaptive_gated.json
```

If you want to do another models. Therefore, running the orders shown in below:

```bash
python -m lightweight_policy.train --config configs/action_only_baseline.json
python -m lightweight_policy.train_dynamics --config configs/distillation_baseline.json
python -m lightweight_policy.train_dynamics --config configs/gated_future.json
python -m lightweight_policy.train_dynamics --config configs/adaptive_distillation.json
```

The output package contains two checkpoints. One is best.ckpt and another one is last.ckpt.

## 4. Finetune training

This step is going to increase success rate in deployment stage. The finetune process focus on gripper opening and closing processes.

Running the following code to do finetune work:

```bash
python -m lightweight_policy.train_dynamics --config configs/adaptive_gated_finetune.json
```

## 5. Training criteria

| Criteria | Meaning |
|---|---|
| action_mse | Error between final predicted actions and real actions |
| initial_action_mse | Initial action error before fusing future information |
| dynamics_teacher_mse | Error between student future characteristics and teacher predictions |
| teacher_future_reliability | Mean of adaptive distillation weights |
| future_gate | Mean of gate value |
| control_score | Combined score of fine-tuning experiments |
| peak_allocated_gib | Peak value of PyTorch memory allocation |

## 6. Deployment

First step:

Install 'unitree_isaac_sim' required environment. See: [unitree_sim_isaaclab](https://github.com/unitreerobotics/unitree_sim_isaaclab)

Second step:

Enter checkpoint、export_config、dataset_name and instruction in [configs/deploy_isaacsim.json](configs/deploy_isaacsim.json).

Third step: single-round evaluation and summary

Run

```bash
conda activate unitree_sim_env
```

```bash
python deployment_script/run_isaacsim.py \
  --sim-repo /Your/PATH/TO/unitree_sim_isaaclab \
  --config configs/deploy_isaacsim.json \
  --seconds 30 --keep-open
```

Fourth step: Multiple-round evaluation and summary (for 50 epochs)

Run

```bash
python deployment_script/evaluate_basket.py \
  --sim-repo /PATH/TO/unitree_sim_isaaclab \
  --config configs/deploy_isaacsim.json \
  --episodes 50 --gui --output runs/deployments/basket_eval_50
```

If you want to test success rate in perturbations, you need to do following changes in [configs/deploy_isaacsim.json](configs/deploy_isaacsim.json)

```bash
  "environment_perturbations": {
    "enabled": true,
```
## 7. Environment perturbations setup

| Perturbations | Set | Description |
|---|---|---|
| Inital position of cylinder | [-0.01, 0.01] m | X is located randomly from -0.36 to -0.34 and Y is located randomly from 0.39 to 0.41 |
| Cylinder mass | Original mass times [0.9,1.1] | Cylinder mass is generated randomly between 0.36kg and 0.44 kg based |
| Cylindrical material friction | Static/dynamic friction multiplied by [0.9,1.1] | efault 1.5 changed to 1.35～1.65 |
| External force pulse | Random in the horizontal direction, resultant force 0.1～0.3 N | Applied to the center of mass of the cylinder, no vertical force and additional torque |
