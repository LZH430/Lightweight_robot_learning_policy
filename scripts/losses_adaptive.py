"""Adaptive demonstration supervision using real action labels."""
import math
import torch
from lightweight_policy.train import loss_and_metrics as original_loss


def validate_data_config(cfg):
    if cfg.get("adaptive_distillation", False) or cfg.get("teacher_weight", 0):
        raise ValueError("Teacher distillation is retired in train_adaptive; use adaptive_data with teacher_weight=0")
    for key, default in (("data_weight", .1), ("data_temperature", .05)):
        value = cfg.get(key, default)
        if not math.isfinite(value) or value < 0 or (key == "data_temperature" and value == 0):
            raise ValueError(f"Invalid {key}")


def adaptive_data_loss(prediction, target, mask, temperature):
    """Self-paced demonstration weighting; errors are NOT data quality scores."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("data_temperature must be finite and positive")
    valid = mask.bool().expand_as(prediction)
    counts = valid.flatten(1).sum(1)
    if (counts == 0).any():
        raise ValueError("Each sample needs valid action entries")
    p = prediction.float().masked_fill(~valid, 0)
    y = target.detach().float().masked_fill(~valid, 0)
    squared_sum = (p-y).square().flatten(1).sum(1)
    error = squared_sum / counts
    weights = torch.exp(-error.detach() / temperature)
    return (weights * squared_sum).sum() / counts.sum(), weights


def loss_and_metrics(model, b, cfg, training, result=None):
    validate_data_config(cfg)
    if result is None:
        result = model(b["visual"], b["text"], b["state"])
    loss, metrics = original_loss(model, b, cfg, training, result)
    if cfg.get("adaptive_data", False) and training and "action" in result:
        data_loss, weights = adaptive_data_loss(result["action"], b["action"],
                                                b["action_mask"], cfg.get("data_temperature", .05))
        loss = loss + cfg.get("data_weight", .1) * data_loss
        metrics.update(adaptive_data_mse=(data_loss.detach().item(), int(b["action_mask"].bool().sum())),
                       data_weight_mean=(weights.mean().item(), weights.numel()))
    if "future_gate" in result:
        gate = result["future_gate"].detach().float()
        metrics["future_gate"] = (gate.mean().item(), gate.numel())
    return loss, metrics
