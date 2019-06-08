"""
File: qn
Date: 5/6/19 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple

def to_sate_tensor(s, device):
    """ converts a numpy array to a Tensor suitable for passing through DQNs """
    return torch.from_numpy(s).to(device=device, dtype=torch.float32)

class StateEncoder(nn.Module):
    def __init__(self, state_shape: Tuple, layer_sizes: List[int], p_dropout=0, device=None):
        super(StateEncoder, self).__init__()
        if len(state_shape) > 1:
            raise ValueError("State must be a flat vector")

        self._layers = list()
        prev_size = state_shape[0]
        for size in layer_sizes:
            layer = nn.Linear(prev_size, size, bias=True).to(device)
            self._layers.append(layer)
            prev_size = size

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=p_dropout).to(device)

        self.output_shape = (layer_sizes[-1], )
        self.device = device

    def forward(self, s) -> torch.Tensor:
        if isinstance(s, np.ndarray):
            s = to_sate_tensor(s, self.device)

        activations = s
        for layer in self._layers:
            z = layer(activations)
            activations = self.dropout(self.relu(z))

        return activations

class ConvEncoder(nn.Module):
    def __init__(self, state_shape: Tuple, device=None):
        super(ConvEncoder, self).__init__()
        in_channels = state_shape[0]

        nc = 32
        self.conv1 = nn.Conv2d(in_channels, nc, (8, 8), stride=(4, 4)).to(device)
        self.conv2 = nn.Conv2d(nc, 2 * nc, (4, 4), stride=(2, 2)).to(device)
        self.conv3 = nn.Conv2d(2 * nc, 2 * nc, (3, 3), stride=(1, 1)).to(device)
        self.relu = nn.ReLU().to(device)

        self.state_shape = state_shape

    def forward(self, s) -> torch.Tensor:
        if isinstance(s, np.ndarray):
            s = to_sate_tensor(s, self.device)
        a = self.relu(self.conv1(s))
        a = self.relu(self.conv2(a))
        a = self.relu(self.conv3(a))
        return torch.flatten(a, 1)

    @property
    def output_shape(self):
        """ the output shape of this CNN encoder.
        :return: tuple of output shape
        """
        return 64, 12, 12

    def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
        """ Utility function for computing output of convolutions
        takes a tuple of (h,w) and returns a tuple of (h,w)
        """
        from math import floor
        if type(kernel_size) is not tuple:
            kernel_size = (kernel_size, kernel_size)
        h = floor(((h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1)
        w = floor(((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1)
        return h, w

class DuelingDQN(nn.Module):
    def __init__(self, encoder, action_size, device=None):
        super(DuelingDQN, self).__init__()

        self.encoder = encoder
        enc_out_size = np.prod(encoder.output_shape)
        self.V = nn.Linear(enc_out_size, 1, bias=True).to(device)
        self.A = nn.Linear(enc_out_size, action_size, bias=True).to(device)

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
    def __init__(self, encoder, action_size, device=None):
        super(DQN, self).__init__()

        self.encoder = encoder
        enc_out_size = np.prod(encoder.output_shape)

        self.output_layer = nn.Linear(enc_out_size, action_size, bias=True).to(device)
        self.device = device

    def forward(self, s) -> torch.Tensor:
        if isinstance(s, np.ndarray):
            s = to_sate_tensor(s, self.device)

        encoding = self.encoder(s)
        return self.output_layer(encoding)
