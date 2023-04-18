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

# Load samples and adjacency matrix
samples
adjs

#### Select one well
adj_i = 0
idx_well = np.where(samples.adj_idx == adj_i)[0]
samples_in = samples[idx_well]
adj_in = adjs[adj_i]
adj_mx = adj_in

#### Select all wells
idx_well = np.where(np.isin(samples.adj_idx,np.arange(20)))[0]
samples_in = samples[idx_well]
# samples_in = samples
adjs = adjs

# adj_in = np.zeros(adj_in.shape)
# np.fill_diagonal(adj_in,1)

def adj2edge(adj,DEVICE):
    rows, cols = np.where(adj == 1)
    edges = zip(rows.tolist(), cols.tolist())
    edge_index_data = torch.LongTensor(np.array([rows, cols])).to(DEVICE)
    return edge_index_data

edge_index_data_list = [adj2edge(adj,DEVICE) for adj in adjs]

batch_size = 4

dataout = create_data_loaders(samples_in, batch_size,
                              train_size=0.8, train_val_size=0.8,
                              shuffle=True, DEVICE = DEVICE)

train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, _mean, _std = dataout

# ==============================================================================
from torch_geometric_temporal.nn.attention import ASTGCN   # For information about the architecture check the source code

num_of_vertices = len(adj_in)
nb_block = 4
in_channels = 1
K = 5
nb_chev_filter = 128
nb_time_filter = 128
time_strides = 1
num_for_predict = 10
len_input = 10
#L_tilde = scaled_Laplacian(adj_mx)
#cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor).to(DEVICE) for i in cheb_polynomial(L_tilde, K)]
net = ASTGCN(nb_block,
             in_channels,
             K,
             nb_chev_filter,
             nb_time_filter,
             time_strides,
             num_for_predict,
             len_input,
             num_of_vertices).to(DEVICE)

print(net)

#------------------------------------------------------
learning_rate = 0.0001
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

print('Net\'s state_dict:')
total_param = 0
for param_tensor in net.state_dict():
    print(param_tensor, '\t', net.state_dict()[param_tensor].size(), '\t', net.state_dict()[param_tensor].device)
    total_param += np.prod(net.state_dict()[param_tensor].size())

print('Net\'s total params:', total_param)

#--------------------------------------------------
print('Optimizer\'s state_dict:')
for var_name in optimizer.state_dict():
    print(var_name, '\t', optimizer.state_dict()[var_name])

def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

masked_flag=0
criterion = nn.L1Loss().to(DEVICE)
criterion_masked = masked_mae
loss_function = 'mse'

metric_method = 'unmask'
missing_value=0.0

if loss_function=='masked_mse':
    criterion_masked = masked_mse         #nn.MSELoss().to(DEVICE)
    masked_flag=1
elif loss_function=='masked_mae':
    criterion_masked = masked_mae
    masked_flag = 1
elif loss_function == 'mae':
    criterion = nn.L1Loss().to(DEVICE)
    masked_flag = 0
elif loss_function == 'rmse':
    criterion = nn.MSELoss().to(DEVICE)
    masked_flag= 0

def compute_val_loss_mstgcn(net, val_loader, criterion,  masked_flag,missing_value,sw, epoch, edge_index_data, limit=None):
    '''
    for rnn, compute mean loss on validation set
    :param net: model
    :param val_loader: torch.utils.data.utils.DataLoader
    :param criterion: torch.nn.MSELoss
    :param sw: tensorboardX.SummaryWriter
    :param global_step: int, current global_step
    :param limit: int,
    :return: val_loss
    '''
    net.train(False)  # ensure dropout layers are in evaluation mode
    with torch.no_grad():
        val_loader_length = len(val_loader)  # nb of batch
        tmp = []  # batch loss
        for batch_index, batch_data in enumerate(val_loader):
            encoder_inputs, labels, adj_idx = batch_data
            edge_index_data = edge_index_data_list[adj_idx.cpu().detach().numpy().astype('int')[0]]
            outputs = net(encoder_inputs, edge_index_data)
            if masked_flag:
                loss = criterion(outputs, labels, missing_value)
            else:
                loss = criterion(outputs, labels)
            tmp.append(loss.item())
            if batch_index % 100 == 0:
                print('validation batch %s / %s, loss: %.2f' % (batch_index + 1, val_loader_length, loss.item()))
            if (limit is not None) and batch_index >= limit:
                break
#
        validation_loss = sum(tmp) / len(tmp)
        sw.add_scalar('validation_loss', validation_loss, epoch)
    return validation_loss

global_step = 0
best_epoch = 0
best_val_loss = np.inf
start_time= time()

# train model
for epoch in range(10):
    params_filename = os.path.join('./', 'epoch_%s.params' % epoch)
    masked_flag = 1
    if masked_flag:
        val_loss = compute_val_loss_mstgcn(net, val_loader, criterion_masked, masked_flag,missing_value, sw, epoch,edge_index_data)
    else:
        val_loss = compute_val_loss_mstgcn(net, val_loader, criterion, masked_flag, missing_value, sw, epoch,edge_index_data)
#
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        # torch.save(net.state_dict(), params_filename)
        print('save parameters to file: %s' % params_filename)
#
    net.train()  # ensure dropout layers are in train mode
#
    for batch_index, batch_data in enumerate(train_loader):
        encoder_inputs, labels, adj_idx = batch_data   # encoder_inputs torch.Size([32, 307, 1, 12])  label torch.Size([32, 307, 12])
        edge_index_data = edge_index_data_list[adj_idx.cpu().detach().numpy().astype('int')[0]]
        optimizer.zero_grad()
        outputs = net(encoder_inputs, edge_index_data) # torch.Size([B, N, T])
#
        if masked_flag:
            loss = criterion_masked(outputs, labels,missing_value)
        else :
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        training_loss = loss.item()
        global_step += 1
        sw.add_scalar('training_loss', training_loss, global_step)
#
        if global_step % 200 == 0:
            print('global step: %s, training loss: %.2f, time: %.2fs' % (global_step, training_loss, time() - start_time))

#### Validation

net.train(False)  # ensure dropout layers are in evaluation mode
with torch.no_grad():
    test_loader_length = len(test_loader)  # nb of batch
    tmp = []  # batch loss
    for batch_index, batch_data in enumerate(test_loader):
        encoder_inputs, labels, adj_idx = batch_data
        edge_index_data = edge_index_data_list[adj_idx.cpu().detach().numpy().astype('int')[0]]
        outputs = net(encoder_inputs, edge_index_data)
        loss = criterion(outputs, labels)
        tmp.append(loss.item())
        if batch_index % 100 == 0:
            print('test_loss batch %s / %s, loss: %.2f' % (batch_index + 1, test_loader_length, loss.item()))
    test_loss = sum(tmp) / len(tmp)
    sw.add_scalar('test_loss', test_loss, epoch)

print(test_loss)

sample_input  = encoder_inputs[0][:,0]  # input
sample_output = outputs[0]  # prediction
sample_labels = labels[0] # truth
print(sample_output.shape, sample_labels.shape)

from matplotlib import pyplot as plt

plt.figure(figsize=(30,4), dpi=80)
idx = np.linspace(150,len(sample_output)-1,20).astype('int')
for j,i in enumerate(idx):
    new_i = j * 20
    _ = plt.plot(range(10+new_i,20+new_i),sample_output[i].detach().cpu().numpy(), color = 'red')
    _ = plt.plot(range(10+new_i,20+new_i),sample_labels[i].cpu().numpy(), color='blue')
    _ = plt.plot(range(0+new_i,10+new_i),sample_input[i].cpu().numpy(), color='green')

plt.savefig(os.path.join(outdir,"pred.svg"))
plt.close()
