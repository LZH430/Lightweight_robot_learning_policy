"""Real action supervision and frozen WMA terminal-feature distillation."""
import math
import torch
from lightweight_policy.train import loss_and_metrics as baseline_loss
from lightweight_policy.model import masked_mse
from lightweight_policy.gripper_finetune import auxiliary_action_loss, validate_supervision_config, left_grasp_loss


def validate_data_config(cfg):
    validate_supervision_config(cfg)
    if cfg.get("stage", "joint") != "joint" or cfg.get("teacher_weight", 0):
        raise ValueError("Dynamics training requires joint stage and teacher_weight=0")
    for key in ("dynamics_teacher_weight", "dynamics_real_weight", "initial_action_weight"):
        value = cfg.get(key, 0.)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid {key}")
    temperature = cfg.get("dynamics_temperature", .05)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("dynamics_temperature must be finite and positive")
    if cfg.get("adaptive_data", False) or cfg.get("adaptive_distillation", False):
        raise ValueError("Use adaptive_dynamics, not legacy adaptive losses")
    if cfg.get("dynamics_teacher_weight", 0.) <= 0:
        raise ValueError("Set positive dynamics_teacher_weight to enable WMA distillation")


def loss_and_metrics(model, b, cfg, training, result=None):
    if result is None:
        result = model(b["visual"], b["text"], b["state"])
    loss, metrics = baseline_loss(model, b, cfg, training, result)
    initial = masked_mse(result["initial_action"], b["action"], b["action_mask"])
    loss = loss + cfg.get("initial_action_weight", .5) * initial
    metrics["initial_action_mse"] = (initial.detach().item(), int(b["action_mask"].bool().sum()))
    extra, control_metrics = auxiliary_action_loss(result["action"], b["action"], b["action_mask"], cfg, state=b["state"])
    initial_extra, _ = auxiliary_action_loss(result["initial_action"], b["action"], b["action_mask"], cfg, state=b["state"])
    loss = loss + extra + cfg.get("initial_action_weight", .5) * initial_extra
    metrics.update(control_metrics)
    if model.gated_future:
        metrics["future_gate"] = (result["future_gate"].detach().mean().item(), result["future_gate"].numel())
    predicted = model.predict_dynamics(b["visual"], b["text"], b["state"], b["dynamics_action"])
    teacher = b["teacher_future"].detach().float()
    kd = (predicted.float() - teacher).square().mean()
    weighted = kd
    if cfg.get("adaptive_dynamics", False):
        if not torch.equal(b["dynamics_action"], b["action"]):
            raise ValueError("Adaptive reliability requires demonstration-conditioned teacher predictions")
        with torch.no_grad():
            error = (teacher - b["future"].float()).square().flatten(1).mean(1)
            weights = torch.exp(-error / cfg.get("dynamics_temperature", .05))
        weighted = ((predicted.float()-teacher).square().flatten(1).mean(1) * weights).mean()
        metrics["teacher_future_reliability"] = (weights.mean().item(), weights.numel())
    loss = loss + cfg["dynamics_teacher_weight"] * (weighted if training else kd)
    if training and cfg.get("adaptive_dynamics", False):
        metrics["weighted_dynamics_mse"] = (weighted.detach().item(), teacher.numel())
    metrics["dynamics_teacher_mse"] = (kd.detach().item(), teacher.numel())
    # Only demonstration-conditioned predictions have a real observed outcome.
    if cfg.get("dynamics_real_weight", 0.):
        if not torch.equal(b["dynamics_action"], b["action"]):
            raise ValueError("Real dynamics supervision requires demonstration actions")
        real = (predicted.float() - b["future"].float()).square().mean()
        loss = loss + cfg["dynamics_real_weight"] * real
        metrics["dynamics_real_mse"] = (real.detach().item(), b["future"].numel())
    grasp_weight = cfg.get("left_grasp_action_weight", 0.0)
    if grasp_weight:
        grasp, count = left_grasp_loss(result["action"], b["action"], b["action_mask"], b["state"])
        if count:
            loss = loss + grasp_weight * grasp
            metrics["left_grasp_action_mse"] = (grasp.detach().item(), count)
    return loss, metrics
