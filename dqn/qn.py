"""
File: qn
Date: 5/6/19 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import torch
import torch.nn as nn
import numpy as np

def _to_state_tensor(s, device):
    """ converts a numpy array to a Tensor suitable for passing through DQNs """
    return torch.from_numpy(s).to(device=device, dtype=torch.float32)

class QN(nn.Module):
    """ Simple 1 layer Q-Network with dropout """
    def __init__(self, state_size, action_size, p_dropout=0, device=None):
        super(QN, self).__init__()
        self.linear = nn.Linear(state_size, action_size, bias=True).to(device)

        self.dropout = nn.Dropout(p=p_dropout).to(device)
        self.device = device

    def forward(self, s) -> torch.Tensor:
        if isinstance(s, np.ndarray):
            s = _to_state_tensor(s, self.device)

        lin = self.linear(s)
        return self.dropout(lin)


class DQN(nn.Module):
    def __init__(self, state_size, action_size, p_dropout=0, hidden_size=16, device=None):
        super(DQN, self).__init__()
        self.linear1 = nn.Linear(state_size, hidden_size, bias=True).to(device)
        self.linear2 = nn.Linear(hidden_size, hidden_size, bias=True).to(device)
        self.linear3 = nn.Linear(hidden_size, action_size, bias=True).to(device)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=p_dropout).to(device)

        self.device = device

    def forward(self, s) -> torch.Tensor:
        if isinstance(s, np.ndarray):
            s = _to_state_tensor(s, self.device)

        l1 = self.dropout(self.relu(self.linear1(s)))
        l2 = self.dropout(self.relu(self.linear2(l1)))
        return self.dropout(self.linear3(l2))
