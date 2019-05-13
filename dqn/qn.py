"""
File: qn
Date: 5/6/19 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List

def to_sate_tensor(s, device):
    """ converts a numpy array to a Tensor suitable for passing through DQNs """
    return torch.from_numpy(s).to(device=device, dtype=torch.float32)

class StateEncoder(nn.Module):
    def __init__(self, state_size: int, layer_sizes: List[int],
                 p_dropout=0, device=None):
        super(StateEncoder, self).__init__()

        self._layers = list()
        prev_size = state_size
        for size in layer_sizes:
            layer = nn.Linear(prev_size, size, bias=True).to(device)
            self._layers.append(layer)
            prev_size = size

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=p_dropout).to(device)
        self.device = device

    def forward(self, s) -> torch.Tensor:
        if isinstance(s, np.ndarray):
            s = to_sate_tensor(s, self.device)

        activations = s
        for layer in self._layers:
            z = layer(activations)
            activations = self.dropout(self.relu(z))

        return activations


class DuelingDQN(nn.Module):
    def __init__(self, state_size, action_size, layer_sizes: List[int],
                 p_dropout=0, device=None):
        super(DuelingDQN, self).__init__()

        self.encoder = StateEncoder(state_size, layer_sizes, p_dropout=p_dropout,
                                    device=device)

        self.encoder_out_size = layer_sizes[-1]
        self.V = nn.Linear(self.encoder_out_size, 1, bias=True).to(device)
        self.A = nn.Linear(self.encoder_out_size, action_size, bias=True).to(device)

        self.device = device


    def forward(self, s) -> torch.Tensor:
        if isinstance(s, np.ndarray):
            s = to_sate_tensor(s, self.device)

        encoding = self.encoder(s)
        v = self.V(encoding)
        a = self.A(encoding)

        Q = v + a - torch.mean(a)
        return Q



class DQN(nn.Module):
    def __init__(self, state_size, action_size, layer_sizes: List[int],
                 p_dropout=0, device=None):
        super(DQN, self).__init__()

        self.encoder = StateEncoder(state_size, layer_sizes,
                                    p_dropout=p_dropout, device=device)

        last_layer_size = layer_sizes[-1] if layer_sizes else state_size
        self.last_layer = nn.Linear(last_layer_size, action_size, bias=True).to(device)
        self.device = device

    def forward(self, s) -> torch.Tensor:
        if isinstance(s, np.ndarray):
            s = to_sate_tensor(s, self.device)

        encoding = self.encoder(s)
        return self.last_layer(encoding)
