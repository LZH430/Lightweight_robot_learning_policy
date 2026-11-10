"""Policy plus action-conditioned terminal feature predictor."""
import math
import torch
from torch import nn
from .model import StudentPolicy


class DynamicsStudentPolicy(StudentPolicy):
    def __init__(self, *args, gated_future=False, gate_init=0.1, **kwargs):
        if kwargs.get("future", False):
            raise ValueError("Dynamics uses its own action-conditioned branch; future must be false")
        super().__init__(*args, **kwargs)
        width = self.action_query.shape[-1]
        self.dynamics_action = nn.Linear(self.action_out.out_features, width)
        self.dynamics_position = nn.Parameter(torch.randn_like(self.action_query) * .02)
        self.dynamics_query = nn.Parameter(torch.randn(1, self.tokens, width) * .02)
        layer = nn.TransformerDecoderLayer(width, self.action_decoder.layers[0].multihead_attn.num_heads,
                                          width*4, dropout=0., batch_first=True, norm_first=True)
        self.dynamics_decoder = nn.TransformerDecoder(layer, 2, nn.LayerNorm(width))
        self.dynamics_out = nn.Linear(width, self.visual.in_features)
        self.gated_future = gated_future
        self.refine_attention = nn.MultiheadAttention(width, self.action_decoder.layers[0].multihead_attn.num_heads, batch_first=True)
        self.refine_norm = nn.LayerNorm(width)
        self.refine_out = nn.Linear(width, self.action_out.out_features)
        if gated_future:
            if not math.isfinite(gate_init) or not 0 < gate_init < 1:
                raise ValueError("gate_init must be between 0 and 1")
            self.refine_gate = nn.Linear(width, 1)
            nn.init.zeros_(self.refine_gate.weight)
            nn.init.constant_(self.refine_gate.bias, math.log(gate_init/(1-gate_init)))

    def forward(self, visual, text, state):
        if visual.shape[1:3] != (self.obs_steps, self.tokens):
            raise ValueError("Observation shape disagrees with model config")
        memory = self.visual(visual.flatten(1, 2)) + self.position
        memory = torch.cat((memory, self.text(text).unsqueeze(1), self.state(state).unsqueeze(1)), dim=1)
        hidden = self.action_decoder(self.action_query.expand(visual.shape[0], -1, -1), memory)
        initial = self.action_out(hidden)
        future = self.predict_dynamics(visual, text, state, initial)
        projected = self.visual(future)
        context, _ = self.refine_attention(hidden, projected, projected, need_weights=False)
        gate = torch.sigmoid(self.refine_gate(hidden)) if self.gated_future else torch.ones_like(hidden[..., :1])
        refined = hidden + gate * self.refine_norm(context)
        return {"action": self.refine_out(refined), "initial_action": initial,
                "policy_future": future, "future_gate": gate}


    def predict_dynamics(self, visual, text, state, actions):
        expected = (visual.shape[0], self.action_query.shape[1], self.action_out.out_features)
        if tuple(actions.shape) != expected or not torch.isfinite(actions).all():
            raise ValueError(f"Finite normalized actions of shape {expected} required")
        memory = self.visual(visual.flatten(1, 2)) + self.position
        memory = torch.cat((memory, self.text(text).unsqueeze(1), self.state(state).unsqueeze(1),
                            self.dynamics_action(actions) + self.dynamics_position), dim=1)
        hidden = self.dynamics_decoder(self.dynamics_query.expand(visual.shape[0], -1, -1), memory)
        return self.dynamics_out(hidden)
