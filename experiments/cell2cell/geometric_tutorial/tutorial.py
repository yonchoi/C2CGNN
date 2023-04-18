import os
from time import time
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy.sparse.linalg import eigs


USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0')
print("CUDA:", USE_CUDA, DEVICE)

from tensorboardX import SummaryWriter
sw = SummaryWriter(logdir='.', flush_secs=5)

import math
from typing import Optional, List, Union

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.typing import OptTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.transforms import LaplacianLambdaMax
from torch_geometric.utils import remove_self_loops, add_self_loops, get_laplacian
from torch_geometric.utils import to_dense_adj
from torch_scatter import scatter_add

## =============================================================================

from torch_geometric_temporal import ChickenpoxDatasetLoader
from torch_geometric_temporal import temporal_signal_split

train, test = temporal_signal_split(dataset, train_ratio=0.9)

import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import DCRNN

class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, filters):
        super(RecurrentGCN, self).__init__()
        self.recurrent = DCRNN(node_features, filters, 1)
        self.linear = torch.nn.Linear(filters, 1)
    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = F.dropout(h, training=self.training)
        h = self.linear(h)
        return h

model = RecurrentGCN(node_features=4, filters=32)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()

for epoch in range(200):
    cost = 0
    for time, snapshot in enumerate(train):
        y_hat = model(snapshot.x,
            snapshot.edge_index,
            snapshot.edge_attr)
        cost = cost + torch.mean((y_hat-snapshot.y)**2)
        cost = cost / (time+1)
        cost.detach()
        cost.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()
