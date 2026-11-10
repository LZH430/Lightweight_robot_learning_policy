import math

import torch
from torch.nn import functional as F


WEIGHTS = {
    "first_action_weight": "first_action_mse",
    "gripper_action_weight": "gripper_action_mse",
    "gripper_transition_weight": "gripper_transition_mse",
}

LEFT_GRASP_DIMS = (0, 1, 2, 3, 4, 5, 6, 14)


def left_grasp_loss(prediction, target, mask, state):
    """Extra approach/closure supervision from cached state and action targets only."""
    valid = mask.bool().expand_as(target)
    left, right = target[..., 14], target[..., 15]
    frames = torch.arange(left.shape[1], device=left.device).expand_as(left)
    horizon = left.shape[1]
    left_close = torch.zeros_like(left, dtype=torch.bool)
    left_close[:, 1:] = (valid[:, 1:, 14] & valid[:, :-1, 14]
                        & (left[:, 1:] < 0) & (left[:, :-1] >= 0))
    left_open = state[:, 14] >= .5
    left_close[:, 0] = valid[:, 0, 14] & left_open & (left[:, 0] < 0)
    first_close = torch.where(left_close, frames, horizon).amin(dim=1)
    # Do not add holding/lifting weight. Only an open measured left hand
    # with an upcoming target closure can enter this extra supervision window.
    right_holding = (state[:, 15] > -.8) & (state[:, 15] < .5) & (right[:, 0] < 0)
    eligible = left_open & (first_close < horizon) & ~right_holding
    handoff = torch.zeros_like(right, dtype=torch.bool)
    handoff[:, 1:] = (valid[:, 1:, 15] & valid[:, :-1, 15]
                     & (right[:, 1:] < 0) & (right[:, :-1] >= 0))
    handoff[:, 0] = valid[:, 0, 15] & (state[:, 15] >= .5) & (right[:, 0] < 0)
    receiver_close = torch.where(handoff, frames, horizon).amin(dim=1)
    eligible &= first_close < receiver_close
    # The prefix includes approach and the closing target, never later lifting.
    phase = eligible[:, None] & (frames <= first_close[:, None])
    selected = torch.zeros_like(valid)
    selected[..., list(LEFT_GRASP_DIMS)] = phase.unsqueeze(-1)
    selected &= valid
    count = int(selected.sum())
    if not count:
        return prediction.new_zeros(()), 0
    return (prediction.float()[selected] - target.float()[selected]).square().mean(), count


def validate_supervision_config(cfg):
    if cfg.get("finetune_scope", "full") != "full":
        raise ValueError("Restricted fine-tuning was removed; use the full-task configuration")
    grasp_weight = cfg.get("left_grasp_action_weight", 0.0)
    if (isinstance(grasp_weight, bool) or not isinstance(grasp_weight, (int, float))
            or not math.isfinite(grasp_weight) or grasp_weight < 0):
        raise ValueError("left_grasp_action_weight must be finite and nonnegative")
    if grasp_weight and cfg["model"].get("action_dim", 16) != 16:
        raise ValueError("Grasp supervision requires 16D actions")
    for key in WEIGHTS:
        value = cfg.get(key, 0.0)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if cfg.get("checkpoint_metric", "action_mse") not in ("action_mse", "control_score"):
        raise ValueError("checkpoint_metric must be action_mse or control_score")
    if type(cfg.get("first_step_gripper_transitions", False)) is not bool:
        raise ValueError("first_step_gripper_transitions must be boolean")
    enabled = any(cfg.get(key, 0.0) for key in WEIGHTS)
    if enabled and cfg["model"].get("action_dim", 16) != 16:
        raise ValueError("Gripper supervision requires the G1 14-arm + 2-gripper schema")
    if cfg.get("checkpoint_metric") == "control_score" and not enabled:
        raise ValueError("control_score requires positive auxiliary supervision weights")


def auxiliary_action_loss(prediction, target, mask, cfg, state=None):
    """Inputs use the existing per-dimension min-max normalization to [-1, 1].

    Gripper indices 14 and 15 follow the verified G1 cache schema. Transitions
    cross the action-range midpoint, in either direction; the crossing frame
    and its two neighbors receive extra supervision. The optional first-step
    mode also supervises measured-hand-state to first-action changes. No object state or timer
    is used, and this function never participates in deployment inference.
    """
    zero = prediction.new_zeros((), dtype=torch.float32)
    if not any(cfg.get(key, 0.0) for key in WEIGHTS):
        return zero, {}
    if prediction.ndim != 3 or prediction.shape[-1] != 16 or prediction.shape != target.shape:
        raise ValueError("Expected matching [batch, horizon, 16] actions")
    valid = mask.bool().expand_as(prediction)
    target = target.float()
    prediction = prediction.float()
    metrics = {}
    loss = zero

    def add(name, predicted, expected, selected, weight):
        nonlocal loss
        count = int(selected.sum())
        if count:
            mse = (predicted[selected] - expected[selected]).square().mean()
            metrics[name] = (mse.detach().item(), count)
            loss = loss + weight * mse

    add("first_action_mse", prediction[:, 0], target[:, 0], valid[:, 0],
        cfg.get("first_action_weight", 0.0))
    add("arm_action_mse", prediction[..., :14], target[..., :14], valid[..., :14], 0.0)
    gp, gt, gv = prediction[..., 14:16], target[..., 14:16], valid[..., 14:16]
    add("gripper_action_mse", gp, gt, gv, cfg.get("gripper_action_weight", 0.0))
    crossings = torch.zeros_like(gv)
    crossings[:, 1:] = (gv[:, 1:] & gv[:, :-1]
                       & ((gt[:, 1:] >= 0) != (gt[:, :-1] >= 0)))
    if cfg.get("first_step_gripper_transitions", False):
        if (state is None or state.shape != (prediction.shape[0], 16)
                or not torch.isfinite(state).all()):
            raise ValueError("First-step gripper supervision requires finite normalized [batch,16] state")
        # A measured holding/closed hand is below the open threshold; the two
        # modalities have separate min-max ranges, so use a coarse open class.
        first_crossing = gv[:, 0] & ((gt[:, 0] >= .5) != (state[:, 14:16] >= .5))
        crossings[:, 0] = first_crossing
        add("first_gripper_transition_mse", gp[:, 0], gt[:, 0], first_crossing, 0.0)
    transitions = F.max_pool1d(crossings.float().transpose(1, 2), 3, stride=1,
                             padding=1).transpose(1, 2).bool() & gv
    add("gripper_transition_mse", gp, gt, transitions,
        cfg.get("gripper_transition_weight", 0.0))

    # Report both directions: reducing missed opening must not cause early release.
    for side, index in (("left", 0), ("right", 1)):
        for direction, selected, wrong in (
            ("open", gv[:, 0, index] & (gt[:, 0, index] >= 0.5), gp[:, 0, index] < 0),
            ("close", gv[:, 0, index] & (gt[:, 0, index] <= -0.5), gp[:, 0, index] > 0),
        ):
            count = int(selected.sum())
            if count:
                metrics[f"first_{side}_{direction}_miss_rate"] = (
                    wrong[selected].float().mean().item(), count)
    return loss, metrics


def checkpoint_score(values, cfg):
    score = values["action_mse"]
    if cfg.get("checkpoint_metric", "action_mse") == "control_score":
        for weight_key, metric in WEIGHTS.items():
            if cfg.get(weight_key, 0.0):
                # A split may contain no midpoint-crossing samples.
                if metric == "gripper_transition_mse" and metric not in values:
                    continue
                score += cfg[weight_key] * values[metric]
        if cfg.get("left_grasp_action_weight", 0.0):
            score += cfg["left_grasp_action_weight"] * values.get("left_grasp_action_mse", 0.0)
    return score
