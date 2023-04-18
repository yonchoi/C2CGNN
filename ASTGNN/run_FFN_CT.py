#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
from time import time
import shutil
import argparse
import configparser
from matplotlib import pyplot as plt

from tensorboardX import SummaryWriter
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

# ASTGNN
from create_dataset import load_npz, create_data_loaders_channels
from modified_classes import make_model, predict_main, norm_Adj
from modified_classes_3channel import make_model, compute_val_loss

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--STdatadir', type=str,
                    help='Directory to the ST data file in fomr of .npz')

parser.add_argument('--outdir', type=str, default='NA',
                    help='Directory where results will be saved')

parser.add_argument('--channel_target', type=int, default=0,
                    help='Index of heldout channel to predict')

parser.add_argument('--input', type=str, default='ERK',
                    help='Input type (ERK,ERK_P, \
                                      ERK-meta, \
                                      ERK-ERK_P, \
                                      ERK-meta-ERK_P)')

parser.add_argument('--use_cuda', type=str2bool, default=True,
                    help='Use cuda')

args = parser.parse_args()

STdatadir     = args.STdatadir
outdir         = args.outdir
input          = args.input
use_cuda       = args.use_cuda
channel_target = args.channel_target

# outdir = '../out/sensor2sensor_conv/exp1'
# samples_dir = '../data/Devan/3channel/data_processed/all/scale-True/samples.npz'

samples = load_npz(STdatadir)

def flatten_samples(samples):
    """
    Flatten samples from n_sample x n_cell to n_sample*n_cell
    Input
    custom dataset with attributes
        .input = (n_sample x n_cell x n_time x n_channel)
        .adj_idx = (n_sample)
        .metdata = (n_sample)
    return
        flattened: (n_sample*n_cell x n_time x n_channel)
        adj_idx: (n_sample*n_cell)
        metadta: (n_sample*n_cell)
    """
    flattened = samples.input.reshape(-1,*samples.input.shape[-2:])
    adj_idx  = np.repeat(samples.adj_idx,samples.input.shape[1])
    metadata = np.repeat(samples.metadata,samples.input.shape[1])
    return flattened, adj_idx, metadata

flattened, adj_idx, metadata = flatten_samples(samples)
cellmeta = adj_idx.reshape(-1,1)

use_cuda = True
batch_size = 64
n_channel = flattened.shape[2]-1

Y = flattened[:,:,channel_target]
channel_input = np.arange(flattened.shape[-1]) != channel_target
X = flattened[:,:,channel_input] # n_cells x n_time x n_channel
X = np.swapaxes(X,1,2) # n_cells x n_channel x n_time
X = X.reshape(len(X),-1) # n_cell x n_channel * n_time

# placeholder although not used currently
M = np.zeros([len(X),1])
P = np.zeros([len(X),1])

X = torch.tensor(X).float()
Y = torch.tensor(Y).float()
M = torch.tensor(M).float()
P = torch.tensor(P).float()

device = torch.device("cuda" if use_cuda else "cpu")

if use_cuda:
    X,Y,M,P = X.to(device), Y.to(device), M.to(device), P.to(device)


# ==============================================================================
# Train model
# ==============================================================================
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from src.model.regressor.FFN import Net, mixedNet, CustomDataset
from src.model.regressor.linear import LinearRegressor
from experiments.ETG_prediction.evaluate import evaluate_model

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5)
split_meta = cellmeta.astype('str')

for k, (train_idx, test_idx) in enumerate(skf.split(X,split_meta)):
#
    X_train, X_final = X[train_idx], X[test_idx]
    Y_train, Y_final = Y[train_idx], Y[test_idx]
    M_train, M_final = M[train_idx], M[test_idx]
    P_train, P_final = P[train_idx], P[test_idx]
    split_meta_train = split_meta[train_idx]
#
    X_train, X_test, Y_train, Y_test, M_train, M_test, P_train, P_test = train_test_split(X_train,
                                                                                          Y_train,
                                                                                          M_train,
                                                                                          P_train,
                                                                                          train_size=0.8,
                                                                                          stratify=split_meta_train)
#
    # Create datasets
    trainset = CustomDataset(X_train,Y_train,M_train,P_train)
    testset  = CustomDataset(X_test,Y_test,M_test,P_test)
    finalset  = CustomDataset(X_final,Y_final,M_final,P_final)

    # Create dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)

    finalloader = torch.utils.data.DataLoader(finalset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)

    dataloaders = {'train': trainloader,
                   'test' : testloader,
                   'final': finalloader}

    def set_n_hidden(n_input, n_output):
        if n_input > 200:
            n_hiddens = (128,64,64)
        if n_input > 100:
            n_hiddens = (64,64)
        else:
            n_hiddens = (16,16)
        return n_hiddens

    model_types = ['flatten','conv']

    for model_type in model_types:

        # Set model directory
        outdir_model = os.path.join(outdir,model_type,'kfold-'+str(k))
        os.makedirs(outdir_model,exist_ok=True)

        # Split model arguments
        model_type = model_type.split('-')
        if len(model_type) == 1:
            mixed_model=False
        else:
            mixed_model=True

        model_type = model_type[0]

        # Writer will output to ./runs/ directory by default
        writer = SummaryWriter(outdir_model)

        #### Train the neural networks
        #set up model, loss, and optimizer (L2 applied for all weights)
        n_input  = X.shape[-1]
        n_output = Y.shape[-1]
        n_input_meta = M.shape[-1]

        n_hiddens = set_n_hidden(n_input,n_output)

        ## Create model

        if mixed_model:
            model = mixedNet
            metanet_kwargs = {'n_input_meta': n_input_meta,
                              'n_hiddens_meta': [32,16]
                              }
        else:
            model = Net
            metanet_kwargs = {}

        convnet_kwargs = {'kernel_size': 16,
                          'kernel_size_pool':2,
                          'conv_channels': [8,16]}

        net = model(n_input = n_input,
                    n_output = n_output,
                    activation = F.leaky_relu,
                    n_hiddens = n_hiddens,
                    mode=model_type,
                    dropout=0.2,
                    batchnorm=False,
                    n_channel=n_channel,
                    **metanet_kwargs,
                    **convnet_kwargs)
        net.to(device)

        ## Set functions for initializing weights
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        net.apply(init_weights) # custom initialization

        lr = 0.0001
        weight_decay=1e-4
        criterion = F.mse_loss
        optimizer = optim.AdamW(net.parameters(),
                                lr=lr,
                                weight_decay=weight_decay)

        net.train(trainloader=trainloader,
                  testloader=testloader,
                  writer = writer,
                  epsilon=0,
                  max_count=20,
                  optimizer=optimizer,
                  epoch=200,
                  criterion=criterion)

        # optimizer = optim.AdamW(net.parameters(),
        #                         lr=lr/10,
        #                         weight_decay=weight_decay)
        #
        # net.train(trainloader=trainloader,
        #           testloader=testloader,
        #           writer = writer,
        #           epsilon=0,
        #           max_count=20,
        #           optimizer=optimizer,
        #           epoch=200,
        #           criterion=criterion)

        # Save model and evaluation
        torch.save(net,
                   os.path.join(outdir_model,'best_model.pt'))

        df_eval = evaluate_model(net,dataloaders,
                                 input_keys=['x','m'],
                                 do_plot=False,
                                 do_plot_time=True,
                                 do_attribution=True,
                                 outdir=outdir_model,
                                 model_type=mixed_model,
                                 device=device)

        df_eval.to_csv(os.path.join(outdir_model,'Eval.csv'))

    raise Exception
        # # Attribution
        # if k == 0:
