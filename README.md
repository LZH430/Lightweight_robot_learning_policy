# Lightweight Robot Learning Policy

Description: **Lightweight learning policy training for G1 + Dex1 (16dof) and IsaacSim deployment (Cylinder pick & place task)**

Algorithm: **Real action supervision + WMA future feature distillation**. 

***Three main models***:

Action-only baseline model (See `action_only_baseline.json`): This model does not use WMA future prediction. 

Distillation baseline model (See `distillation_baseline.json`): This model uses WMA distillation with constant distillated ratio and does not have future gated function.

Adaptive gated model (See `adaptive_gated.json`): This model has both dynamic distillation ratio and future gated function.

***Two external models***:

Gated model (See `gated_future.json`): This model has future gated function without dynamic distillation function.

Adaptive distillation model (See `adaptive_distillation.json`): This model has dynamic distillation function but does not have future gated function.

## 2. Environment Installation

# Teacher Model Running Environment Installation



# Student Model Running Environment Installation

```bash
conda create -n lightweight_model python=3.10 pip -y
conda activate lightweight_model
python -m pip install -e .
python -m pip check
python -c "import torch, lightweight_policy; print(torch.__version__, torch.cuda.is_available())"
```

## 2. Data Processing 

- Degree of freedom 16: Robotic arms (7 joints) + left gripper + right gripper.



## 2. Environment Installation

### 学生训练

```bash
conda create -n unifolm-student python=3.10 pip -y
conda activate unifolm-student
python -m pip install -e .
python -m pip check
python -c "import torch, lightweight_policy; print(torch.__version__, torch.cuda.is_available())"
```

GPU 训练需要与本机驱动兼容的 CUDA 版 PyTorch。当前 pyproject 的核心依赖为 PyTorch；不要将成功安装包等同于全部导出、仿真依赖已安装。

### 教师导出与仿真

- **导出环境**：需要原 WMA 仓库及其编码器、采样器、数据加载器依赖；同时安装本仓库。
- **Isaac Lab 环境**：需要已能启动的 Unitree 仿真场景，以及冻结编码器使用的推理依赖。
- 不要将教师环境的整套依赖版本直接覆盖到 Isaac 环境，尤其是 PyTorch/CUDA。
- 源码包名已变更，已有环境请重新执行 `python -m pip install -e .`。不要在同一环境混装另一个包含同名包的工作副本。

## 3. 数据准备和缓存导出

先将 LeRobot 原始数据转换为原 WMA 数据加载器要求的 CSV、视频、transition 文件及统计量。以下划分命令接收的是**转换后的 CSV**：

```bash
python -m lightweight_policy.data_split \
  --csv /PATH/TO/CONVERTED_G1.csv \
  --output data/splits \
  --dataset-name unitree_g1_pickplace_cylinder
```

按完整 episode 划分，避免同一轨迹或不同相机视角跨训练/验证集合。已正确划分的数据可复用。

编辑 [configs/export.json](configs/export.json)：

| 字段 | 用途 |
|---|---|
| teacher.repo / model_config / checkpoint | WMA 代码、结构配置与权重 |
| datasets.train / val 的 meta_path | 训练、验证 CSV |
| data_dir / transition_dir | 转换数据根目录、统计量目录 |
| schema_sources | 原始数据 meta/info.json，用于检查动作名称与顺序 |
| output | 输出缓存目录，必须与训练配置的 train_cache / val_cache 对应 |
| clips_per_split | 每个数据条目的 clip 数量，不是整个划分的总量 |

当前配置是 train=64、val=4，每个条目分别采样相应数量的窗口。数据较多时导出量会很大；先小规模测量时间与显存。随机窗口可能重叠，不保证全部独立。

```bash
python -m lightweight_policy.action_schema --config configs/export.json
python -m lightweight_policy.export_cache --config configs/export.json
```

导出目录必须为空，避免混合缓存版本。当前配置开启 `teacher.enable_dynamics=true`，关闭 `export_teacher_actions`：教师提供未来特征目标，动作标签仍来自示范数据。

典型单样本字段：

| 字段 | 形状/含义 |
|---|---|
| visual | [2, 64, 1280]，当前两帧视觉特征 |
| text | [1024]，文本特征 |
| state | [16]，当前归一化状态 |
| action / action_mask | [16, 16]，真实动作及有效维度 |
| future | [64, 1280]，真实未来图像特征 |
| teacher_future | 教师预测的未来特征 |
| dynamics_action | 生成教师未来目标时使用的动作条件 |
| dynamics_fs / dynamics_fs_type | 时间条件 |

学生动态分支使用与教师目标一致的动作条件训练。真实未来特征只用于训练目标或可靠性评估，不作为部署时输入。

## 4. Training

| 配置 | 启动模块 | 真实动作监督 | 未来蒸馏 | 门控 | 自适应 |
|---|---|---|---|---|---|
| action_only_baseline.json | lightweight_policy.train | 是 | 否 | 否 | 否 |
| distillation_baseline.json | lightweight_policy.train_dynamics | 是 | 是 | 否 | 否 |
| gated_future.json | lightweight_policy.train_dynamics | 是 | 是 | 是 | 否 |
| adaptive_distillation.json | lightweight_policy.train_dynamics | 是 | 是 | 否 | 是 |
| adaptive_gated.json | lightweight_policy.train_dynamics | 是 | 是 | 是 | 是 |

**当前配置注意事项：** `adaptive_distillation.json` 和 `gated_future.json` 的 output 都指向 `runs/wma_gated_future_64clips`。启动对照实验前，为其中一个设置独立目录，例如 `runs/wma_adaptive_distillation_64clips`。不要根据这个旧目录名判断模型种类；以配置内容为准。本次文档整理未移动已有结果或修改实验配置。

```bash
python -m lightweight_policy.train --config configs/action_only_baseline.json
python -m lightweight_policy.train_dynamics --config configs/distillation_baseline.json
python -m lightweight_policy.train_dynamics --config configs/gated_future.json
python -m lightweight_policy.train_dynamics --config configs/adaptive_distillation.json
python -m lightweight_policy.train_dynamics --config configs/adaptive_gated.json
```

固定蒸馏使用示范动作损失、初始动作损失和教师未来特征损失。自适应权重根据教师未来特征与真实未来特征的误差计算；门控调节预测未来信息参与动作融合的程度。它们分别控制“如何学习教师”和“如何使用未来信息”。

`teacher_weight=0` 表示不做旧版动作蒸馏，不能据此判断未来蒸馏关闭。`dynamics_teacher_weight` 才控制当前未来蒸馏；`model.future=false` 关闭的是旧观察条件未来分支，`dynamics_v2` 有自己的动作条件动态分支。旧 `train_adaptive.py` 不是上表配置的入口。

### 续训与微调

每个输出目录包含 `metrics.jsonl`、`best.ckpt` 和 `last.ckpt`。

```bash
python -m lightweight_policy.train_dynamics \
  --config configs/adaptive_gated.json \
  --resume /PATH/TO/last.ckpt
```

`--resume` 恢复模型和优化器等状态，应沿用一致的训练目标。`--init-from` 只加载兼容模型权重，开始新实验；不能用来直接跨越不同模型结构或动作维度。

`adaptive_gated_finetune.json` 是额外的 G1 抓取辅助监督实验，不属于五组基础对照。它通过 `init_from` 读取已有 adaptive_gated checkpoint，并使用 control_score 选模，应单独报告。

## 5. 指标如何理解

| 指标 | 含义 |
|---|---|
| action_mse | 最终动作与真实示范动作的误差，越低越好 |
| initial_action_mse | 融合未来信息前的初始动作误差 |
| dynamics_teacher_mse | 学生未来特征与教师预测的误差，不是真实未来误差 |
| teacher_future_reliability | 自适应蒸馏权重均值，不是预测准确率 |
| future_gate | 门值均值，不是“越大越好” |
| control_score | 微调实验的组合评分，具体组成见配置与 gripper_finetune.py |
| peak_allocated_gib | PyTorch 分配显存峰值，不是整卡或完整部署显存 |

比较模型时固定验证缓存，并区分最佳单轮、最后若干轮均值和多随机种子结果。离线 MSE 不等于任务成功率。

## 6. Deployment

详细说明见 [ISAAC_DEPLOY.md](ISAAC_DEPLOY.md)。

先填写 [configs/deploy_isaacsim.json](configs/deploy_isaacsim.json) 的 checkpoint、export_config、dataset_name 和 instruction。文本、编码器、归一化及夹爪校准需与训练数据一致。

在 Linux 的 Isaac Lab 环境、仓库根目录运行：

```bash
python deployment_script/run_isaacsim.py \
  --sim-repo /PATH/TO/unitree_sim_isaaclab \
  --config configs/deploy_isaacsim.json \
  --seconds 30 --keep-open
```

脚本在同一进程创建场景并调用学生模型，无需额外 HTTP 服务或另开官方 sim_main.py。场景为 `Isaac-PickPlace-Cylinder-G129-Dex1-Joint`。

输出包含视频、轨迹、summary.json 和启动/异常诊断。计时基于仿真时间，录制视频不能用来推断实时推理速度。

多轮评估与重新汇总：

```bash
python deployment_script/evaluate_basket.py \
  --sim-repo /PATH/TO/unitree_sim_isaaclab \
  --config configs/deploy_isaacsim.json \
  --episodes 50 --gui --output runs/deployments/basket_eval_50

python deployment_script/evaluate_basket.py \
  --summarize runs/deployments/basket_eval_50
```

**当前成功定义是“圆柱任意部分进入篮子内部”，不要求松开、稳定保持或完整放入。** 因而应称为篮子进入率，而不是未经限定的完整放置成功率。初始化异常、有效试验的分母和不同配置分组规则见 ISAAC_DEPLOY.md；不要将所有失败日志一律算作抓取失败。

夹爪校准见 [GRIPPER_CALIBRATION.md](GRIPPER_CALIBRATION.md)，环境扰动见 [ENVIRONMENT_PERTURBATIONS.md](ENVIRONMENT_PERTURBATIONS.md)。实际部署性能需由本机记录验证，文档命令不是成功率结果。

## 7. tests 与 deployment_script 的区别

`deployment_script` 实现实际推理、控制和统计；`tests` 用已知输入验证这些逻辑是否正确。测试不是重复部署，也不在普通训练时自动运行。

| 测试文件 | 检查内容 |
|---|---|
| test_gripper_finetune.py | G1 辅助损失、有效维度和梯度 |
| test_gripper_calibration.py | 夹爪映射与校准 |
| test_environment_perturbations.py | 环境扰动逻辑 |
| test_basket_intersection.py | 圆柱/篮子几何相交 |
| test_basket_reporting.py | 结果汇总、分组、异常和重复统计 |

安装本仓库且环境具备 PyTorch、NumPy 后：

```bash
python -m unittest discover -s tests -p "test_*.py"
```

测试主要使用合成数据和模拟运行结果，不需要真实机器人，不能替代 Isaac 场景和真实 checkpoint 的闭环验证。

## 8. 目录导航

```text
configs/                    数据导出、训练及部署配置
data/splits/                按 episode 划分的 CSV
src/lightweight_policy/
  action_schema.py          动作/状态 schema 验证
  data_split.py             数据划分
  teacher.py                原 WMA 模型加载和采样
  base_encoders.py           编码器与动作条件未来导出
  export_cache.py            离线缓存生成
  model.py / train.py        纯动作基线与缓存读取
  model_dynamics.py         动作条件未来预测和门控融合
  train_dynamics.py         动态蒸馏训练
  gripper_finetune.py        G1 辅助监督与选模评分
scripts/losses_dynamics.py   当前动态蒸馏损失
deployment_script/          部署、校准、环境扰动及成功率统计
tests/                      回归测试
cache/                      导出的训练/验证缓存
runs/                       checkpoint、指标和部署结果
```

根目录中关于旧 student.json、旧 adaptive-data 或历史 26 维实验的说明保留作历史参考；当前入口以本 README 和实际配置为准。
