# app/modules/net.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class Layers(nn.Module):

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float
    ):
        super().__init__()
        self.dropout = dropout
        # TODO

    def forward(self, batch: torch.Tensor):
        # TODO
        return
