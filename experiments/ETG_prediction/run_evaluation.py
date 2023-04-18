import os,gc

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import argparse

import scanpy as sc

from experiments.ETG_prediction.evaluate import evaluate_model
from experiments.ETG_prediction.util import data

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from src.model.regressor.FFN import Net, mixedNet, CustomDataset
from src.model.regressor.linear import LinearRegressor

from util.invnorm import rank_INT

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

parser.add_argument('--anndatadir', type=str,
                    help='Directory to the anndata file')

parser.add_argument('--outdir', type=str, default='NA',
                    help='Directory where results will be saved')

parser.add_argument('--treatment', type=str, default='all',
                    help='Type of treatment')

parser.add_argument('--input', type=str, default='ERK',
                    help='Input type (ERK,ERK_P, \
                                      ERK-meta, \
                                      ERK-ERK_P, \
                                      ERK-meta-ERK_P)')

parser.add_argument('--input_subset', type=str, default='all',
                    help='Input type')

parser.add_argument('--output', type=str, default='ETG',
                    help='Input type (ERK,ERK_P, \
                                      ERK-meta, \
                                      ERK-ERK_P, \
                                      ERK-meta-ERK_P, \
                                      ETG)')

parser.add_argument('--target', type=str, default='all',
                    help='Target gene, individual gene or all')

parser.add_argument('--use_cuda', type=str2bool, default=True,
                    help='Use cuda')

parser.add_argument('--transform_y', type=str, default='scale',
                    help='Define transformation to use on Y')

parser.add_argument('--remove_outlier', type=float, default=0,
                    help='Top and bottom percentile to remove')

#### Set up experiments
parser.add_argument('--do_attribution', type=str2bool, default=False,
                    help='Top and bottom percentile to remove')

args = parser.parse_args()

anndatadir  = args.anndatadir
outdir      = args.outdir

treatment      = args.treatment
remove_outlier = args.remove_outlier

input        = args.input
input_subset = args.input_subset

output      = args.output
target      = args.target
t_type      = args.transform_y

use_cuda    = args.use_cuda

# evaluation setting
do_attribution = args.do_attribution

# Overwrite input args (remove)
# anndatadir = "data/Ram/ERK_ETGs_Replicate1/combined_data_raw.h5ad"
# anndatadir = "data/Devan/3channel/combined_data_raw.h5ad"
# anndatadir="data/Ram/All_Replicates_dataset/combined_data_raw_subset.h5ad"
# anndatadir="data/Ram/All_Replicates_dataset/combined_data_raw.h5ad"

# outdir = 'out/exp1'

if outdir == 'NA':
    outdir = os.path.join(os.path.dirname(anndatadir),'out')

print('Saving to ', outdir)
# outdir = os.path.join(outdir,treatment,input,target)

# ==============================================================================
# Load data
# ==============================================================================
adata = sc.read_h5ad(anndatadir)
adata = data.filter_cells(adata,treatment=treatment)

Y = data.get_data(adata, output,
                  t_type=t_type,
                  subset=target)

adata,Y = data.remove_outlier(adata,Y,remove_outlier)

X = data.get_data(adata, input,
                  t_type='raw',
                  subset=input_subset)
M, split_meta = data.get_meta(adata)
C = data.get_cell(adata)
P = data.get_param(adata)

X = torch.tensor(X).float()
Y = torch.tensor(Y).float()
M = torch.tensor(M).float()
P = torch.tensor(P).float()

n_channel = data.get_n_channel(adata)

device = torch.device("cuda" if use_cuda and torch.else "cpu")

if use_cuda:
    X,Y,M,P = X.to(device), Y.to(device), M.to(device), P.to(device)

# ==============================================================================
# Train model
# ==============================================================================

batch_size = 32

# ------------------------------------------------------------------------------
# Split data

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5,
                      random_state=0)

for k, (train_idx, test_idx) in enumerate(skf.split(X,split_meta)):
#
    X_train, X_final = X[train_idx], X[test_idx]
    Y_train, Y_final = Y[train_idx], Y[test_idx]
    M_train, M_final = M[train_idx], M[test_idx]
    P_train, P_final = P[train_idx], P[test_idx]
    split_meta_train = split_meta[train_idx]
    C_train, C_final = C[train_idx], C[test_idx]
#
    X_train, X_test, Y_train, Y_test, M_train, M_test, P_train, P_test, C_train, C_test = train_test_split(X_train,
                                                                                                           Y_train,
                                                                                                           M_train,
                                                                                                           P_train,
                                                                                                           C_train,
                                                                                                           train_size=0.8,
                                                                                                           stratify=split_meta_train,
                                                                                                           random_state=0)
#
    # Create datasets
    trainset = CustomDataset(X_train,Y_train,M_train,P_train,C_train)
    testset  = CustomDataset(X_test,Y_test,M_test,P_test,C_test)
    finalset  = CustomDataset(X_final,Y_final,M_final,P_final,C_final)
#
    # Create dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)
#
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)
#
    finalloader = torch.utils.data.DataLoader(finalset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)
#
    dataloaders = {'train': trainloader,
                   'test' : testloader,
                   'final': finalloader}
#
    dataset_dict = dataloaders['test'].dataset[:]
    # print(dataset_dict['Y'][:10])
#
    # ------------------------------------------------------------------------------
    # Neural network

    def set_n_hidden(n_input, n_output):
        if n_input > 200:
            n_hiddens = (128,64,64)
        if n_input > 100:
            n_hiddens = (64,64)
        else:
            n_hiddens = (16,16)
        return n_hiddens

    if input == 'ERK':
        # Input = sensor activity
        if n_channel == 1:
            # model_types = ['flatten-mixed','conv-mixed','flatten','conv']
            # model_types = ['conv-mixed','flatten','conv']
            model_types = ['conv','flatten']
        else:
            model_types = ['conv-mixed','conv']
    else:
        # Input = calculated parameters
        model_types = ['flatten']

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
                          'kernel_size_pool':3,
                          'conv_channels': [16,16]}

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

        lr = 0.001
        weight_decay=1e-2
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
                  epoch=100,
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

        do_attribution_ = do_attribution and k==0 # Perform attribution only on 1 fold

        df_eval = evaluate_model(net,dataloaders,
                                 input_keys=['x','m'],
                                 outdir=os.path.join(outdir_model,'FFN'),
                                 model_type=mixed_model,
                                 do_plot=True,
                                 do_attribution=do_attribution_,
                                 do_plot_range=True,
                                 device=device,
                                 n_channel=n_channel)
        df_eval.to_csv(os.path.join(outdir_model,'Eval.csv'))

    # ------------------------------------------------------------------------------
    # Linear

    model_types = ['linear','ridge','lasso']

    for model_type in model_types:
        #
        # Set model directory
        outdir_model = os.path.join(outdir,model_type,'kfold-'+str(k))
        os.makedirs(outdir_model,exist_ok=True)
        #
        model = LinearRegressor(model_type)
        #
        model.fit_dataloader(trainloader,
                             testloader)
                             #
        df_eval = evaluate_model(model,dataloaders,
                                 outdir=os.path.join(outdir_model,model_type),
                                 do_plot=True)
        df_eval.to_csv(os.path.join(outdir_model,'Eval.csv'))
        print(df_eval)

    # raise Exception()
