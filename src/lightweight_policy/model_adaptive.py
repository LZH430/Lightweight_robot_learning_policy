"""Additive gated-future policy; original StudentPolicy is unchanged."""
import math
import torch
from torch import nn
from .model import StudentPolicy


class AdaptiveStudentPolicy(StudentPolicy):
    def __init__(self, *args, gated_future=False, gate_init=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.gated_future = gated_future
        if gated_future:
            if not self.future:
                raise ValueError("gated_future requires future=true")
            if not math.isfinite(gate_init) or not 0 < gate_init < 1:
                raise ValueError("gate_init must lie strictly between 0 and 1")
            width = self.action_query.shape[-1]
            heads = self.action_decoder.layers[0].multihead_attn.num_heads
            self.action_future_attention = nn.MultiheadAttention(width, heads, batch_first=True)
            self.action_future_norm = nn.LayerNorm(width)
            self.action_future_gate = nn.Linear(width, 1)
            nn.init.zeros_(self.action_future_gate.weight)
            nn.init.constant_(self.action_future_gate.bias, math.log(gate_init / (1-gate_init)))

    def set_stage(self, stage):
        super().set_stage(stage)
        if self.gated_future:
            for name, parameter in self.named_parameters():
                if name.startswith("action_future_"):
                    parameter.requires_grad_(stage in ("joint", "action"))

    def forward(self, visual, text, state):
        if not self.gated_future:
            return super().forward(visual, text, state)
        if visual.shape[1:3] != (self.obs_steps, self.tokens):
            raise ValueError("Cached observation steps or tokens disagree with model config")
        batch = visual.shape[0]
        memory = self.visual(visual.flatten(1, 2)) + self.position
        memory = torch.cat((memory, self.text(text).unsqueeze(1),
                            self.state(state).unsqueeze(1)), dim=1)
        future_hidden = self.future_decoder(self.future_query.expand(batch, -1, -1), memory)
        predicted = self.future_out(future_hidden)
        result = {"future": predicted}
        if self.stage == "future":
            return result
        # Current observations remain an independent action path.
        hidden = self.action_decoder(self.action_query.expand(batch, -1, -1), memory)
        future_memory = self.visual(predicted)
        context, _ = self.action_future_attention(hidden, future_memory, future_memory,
                                                   need_weights=False)
        gate = torch.sigmoid(self.action_future_gate(hidden))
        hidden = hidden + gate * self.action_future_norm(context)
        result.update(action=self.action_out(hidden), future_gate=gate)
        return result
