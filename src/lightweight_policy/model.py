import torch
from torch import nn


class StudentPolicy(nn.Module):
    """Cached spatial features -> action chunk; optional predicted subgoal."""

    def __init__(self, visual_dim, text_dim, state_dim=16, action_dim=16,
                 horizon=16, tokens=64, obs_steps=2, width=256, layers=4,
                 heads=8, future=False):
        super().__init__()
        self.stage = "joint"
        self.future = future
        self.obs_steps, self.tokens = obs_steps, tokens
        self.visual = nn.Linear(visual_dim, width)
        self.text = nn.Linear(text_dim, width)
        self.state = nn.Linear(state_dim, width)
        self.position = nn.Parameter(torch.randn(1, obs_steps * tokens, width) * .02)
        self.action_query = nn.Parameter(torch.randn(1, horizon, width) * .02)
        def decoder():
            layer = nn.TransformerDecoderLayer(width, heads, width * 4,
                                               dropout=0.0, batch_first=True,
                                               norm_first=True)
            return nn.TransformerDecoder(layer, layers, nn.LayerNorm(width))
        self.action_decoder = decoder()
        self.action_out = nn.Linear(width, action_dim)
        if future:
            self.future_query = nn.Parameter(torch.randn(1, tokens, width) * .02)
            self.future_decoder = decoder()
            self.future_out = nn.Linear(width, visual_dim)

    def set_stage(self, stage):
        if stage not in ("joint", "future", "action"):
            raise ValueError("stage must be joint, future or action")
        if stage != "joint" and not self.future:
            raise ValueError("Staged subgoal training requires model.future=true")
        self.stage = stage
        for name, parameter in self.named_parameters():
            action = name.startswith(("action_query", "action_decoder.", "action_out."))
            parameter.requires_grad_(stage == "joint" or (action if stage == "action" else not action))

    @torch.no_grad()
    def predict(self, visual, text, state):
        """Feature-level inference without target features; preserves training state."""
        stage, training = self.stage, self.training
        try:
            self.stage = "joint"
            self.eval()
            return self(visual, text, state)
        finally:
            self.stage = stage
            self.train(training)

    def forward(self, visual, text, state):
        # visual: [B, observation_steps, spatial_tokens, frozen_encoder_dim]
        if visual.shape[1:3] != (self.obs_steps, self.tokens):
            raise ValueError("Cached observation steps/tokens disagree with model config")
        batch = visual.shape[0]
        memory = self.visual(visual.flatten(1, 2)) + self.position
        memory = torch.cat((memory, self.text(text).unsqueeze(1),
                            self.state(state).unsqueeze(1)), dim=1)
        result = {}
        if self.future:
            hidden = self.future_decoder(self.future_query.expand(batch, -1, -1), memory)
            predicted = self.future_out(hidden)
            result["future"] = predicted
            # Actions explicitly consume the predicted feature, not the ground truth.
            memory = torch.cat((memory, self.visual(predicted)), dim=1)
        if self.stage == "future":
            return result
        hidden = self.action_decoder(self.action_query.expand(batch, -1, -1), memory)
        result["action"] = self.action_out(hidden)
        return result


def masked_mse(prediction, target, mask):
    # Select valid entries before subtracting (padding may contain NaNs).
    valid = mask.bool().expand_as(prediction)
    if not valid.any():
        raise ValueError("Batch has no valid action targets")
    return (prediction.float()[valid] - target.float()[valid]).square().mean()
