import numpy as np
import torch
import torch.nn as nn

class MLP_onehot(nn.Module):
  def __init__(self):
    super().__init__()
    self.linears = nn.ModuleList([nn.Linear(22,1) for i in range(31)])
    self.hc1     = nn.Linear(31,256)
    self.hc2     = nn.Linear(256,256)
    self.hc3     = nn.Linear(256,256)
    self.out     = nn.Linear(256, 3)

  def forward(self, x):
    # for i,l in enumerate(self.linears):
    x = torch.cat([ l(x[:,i,:]) for i,l in enumerate(self.linears)], dim=1) # should be N x 31
    x = nn.functional.relu(self.hc1(x))
    x = nn.functional.relu(self.hc2(x))
    x = nn.functional.relu(self.hc3(x))
    x = self.out(x)

    return x


class MLP_onehot_flat(nn.Module):
  def __init__(self):
    super().__init__()
    self.l1      = nn.Linear(31*22, 256)
    self.l2      = nn.Linear(256,256)
    self.l3      = nn.Linear(256,256)
    self.out     = nn.Linear(256, 3)

  def forward(self, x):
    x = nn.functional.relu(self.l1(x))
    x = nn.functional.relu(self.l2(x))
    x = nn.functional.relu(self.l3(x))
    x = self.out(x)

    return x
