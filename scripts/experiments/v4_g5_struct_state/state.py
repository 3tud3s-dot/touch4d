"""Per-Anchor recurrent memory; signed information stays per physical hand."""
import torch
from torch import nn
from scripts.training.model import masked


class AnchorStateMemory(nn.Module):
    input_width = 301
    width = 128

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.LayerNorm(301), nn.Linear(301, 128), nn.GELU())
        self.cell = nn.GRUCell(128, 128)
        self.readout = nn.Linear(128, 128)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, state_input, previous_z, anchor_valid):
        if state_input.shape[-1] != self.input_width or previous_z.shape != (*state_input.shape[:2], self.width):
            raise ValueError('Expected Anchor state input [B,A,301], Z [B,A,128]')
        encoded = self.encoder(masked(state_input, anchor_valid))
        b, a, _ = encoded.shape
        z = self.cell(encoded.reshape(b*a, 128), masked(previous_z, anchor_valid).reshape(b*a, 128))
        z = masked(z.reshape(b, a, 128), anchor_valid)
        return z, masked(self.readout(z), anchor_valid)
