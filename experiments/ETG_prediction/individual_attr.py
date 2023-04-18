import os,gc

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import argparse

import scanpy as sc

from experiments.ETG_prediction.evaluate import evaluate_model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from src.model.regressor.FFN import Net, mixedNet, CustomDataset
from src.model.regressor.linear import LinearRegressor
from src.util.util import series2colors

import torch
from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
)

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

parser.add_argument('--target', type=str, default='all',
                    help='Target gene, individual gene or all')

parser.add_argument('--use_cuda', type=str2bool, default=True,
                    help='Use cuda')

args = parser.parse_args()

anndatadir  = args.anndatadir
treatment   = args.treatment
outdir      = args.outdir
input       = args.input
target      = args.target
use_cuda    = args.use_cuda

# Overwrite input args (remove)
## Overwrite
anndatadir = "data/Ram/ERK_ETGs_Replicate1/combined_data_raw.h5ad"
treatment = 'all'
input  = 'ERK'
target = ''
use_cuda = True
outdir = f'out/exp4_scaled-128/plot/{treatment}/{input}'

targets = ['pERK','FRA1','EGR1','pRB','cMyc','cFos','pcFos','cJun']

attr_models = []

adata_orig = sc.read_h5ad(anndatadir)

attr_method = 'IG'

# if outdir == 'NA':
#     outdir = os.path.join(os.path.dirname(anndatadir),'out')
#
# outdir = os.path.join(outdir,treatment,input,target)

os.makedirs(outdir,exist_ok=True)

for target in targets:

    # ==============================================================================
    # Load data
    # ==============================================================================
    from sklearn.preprocessing import scale

    adata = sc.read_h5ad(anndatadir)

    # Proecss
    adata = adata[np.invert(np.isnan(adata.X).any(1))]

    if treatment != 'all':
        adata = adata[adata.obs.Treatment == treatment]

    assert len(adata) != 0, 'No cells left after initial filtering'

    ## Set X, input
    inputs = input.split("-")

    Xs = []
    for inp in inputs:
        if inp == 'ERK':
            X = adata.X
        elif inp == 'ERK_P':
            X = scale(adata.obsm['erk_param'],axis=0)
        else:
            raise Exception('Input argument must be start with either ERK or ERK_P')
        Xs.append(X)

    X = np.concatenate(Xs,axis=1)


    ## Set Y, target genes
    if target == 'all':
        Y = adata.obsm["Y"]
    else:
        target_names = adata.uns["Y"]
        Y = adata.obsm["Y"][:,target==target_names]

    # Scale output per gene
    from sklearn.preprocessing import scale
    Y = scale(Y,axis=0)

    ## Set M, meta
    from sklearn.preprocessing import OneHotEncoder

    cellmeta = adata.obs

    enc = OneHotEncoder(handle_unknown='ignore')
    M_well = enc.fit_transform(cellmeta[['Well of Origin']].values).toarray()
    # M_size = cellmeta[['Size']].values

    # M = np.concatenate([M_well,M_size],axis=1)
    M = np.concatenate([M_well],axis=1)
    M = scale(M,axis=0)

    ## Set P if required
    P = adata.obsm["erk_param"]
    P = scale(P,axis=0)

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

    batch_size = 128

    # ------------------------------------------------------------------------------
    # Split data

    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=5)
    split_meta = cellmeta.astype('str').sum(1)

    for k, (train_idx, test_idx) in enumerate(skf.split(X,split_meta)):
    #
        X_train, X_final = X[train_idx], X[test_idx]
        Y_train, Y_final = Y[train_idx], Y[test_idx]
        M_train, M_final = M[train_idx], M[test_idx]
        P_train, P_final = P[train_idx], P[test_idx]
        split_meta_train = split_meta[train_idx]
        split_meta_final = split_meta[test_idx]
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

    # ------------------------------------------------------------------------------
    # Neural network

    def set_n_hidden(n_input, n_output):
        if n_input > 100:
            n_hiddens = (64,64)
        else:
            n_hiddens = (16,16)
        return n_hiddens


    model_type = 'conv-mixed'

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
                **metanet_kwargs,
                **convnet_kwargs)
    net.to(device)

    ## Set functions for initializing weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    net.apply(init_weights) # custom initialization

    criterion = F.mse_loss
    optimizer = optim.AdamW(net.parameters(),
                            lr=0.001,
                            weight_decay=1e-3)

    net.train(trainloader=trainloader,
              testloader=testloader,
              writer = writer,
              epsilon=0,
              max_count=10,
              optimizer=optimizer,
              epoch=200,
              criterion=criterion)

    # Save model and evaluation
    torch.save(net,
               os.path.join(outdir_model,'best_model.pt'))

    df_eval = evaluate_model(net,dataloaders,input_keys=['x','m'])
    df_eval.to_csv(os.path.join(outdir_model,'Eval.csv'))

    ## Loss
    input_keys = ['x','m']
    dfs = []
    #
    train_type = 'final'
    dataloader = dataloaders[train_type]
    #
    dataset_dict = dataloader.dataset[:]
    #
    X_in = X_final
    M_in = M_final
    Y_in = Y_final

    X_in,_,M_in,_,Y_in,_,split_meta_in,_ = train_test_split(X_in,M_in,Y_in,split_meta_final,
                                                            train_size = 0.05,
                                                            stratify=split_meta_final)

    targets_ =Y_in.cpu().detach().numpy()
    #
    inputs = {'x':X_in,'m':M_in}
    #
    with torch.no_grad():
        outputs = net(**inputs)
        #
        if torch.is_tensor(outputs):
            outputs = outputs.cpu().detach().numpy()
    #
    y = outputs.reshape(len(targets_),-1)
    x = targets_
    errors = np.square(x - y)
    print(errors.mean())
    #
    ## Attribution
    input_attr = (inputs['x'], inputs['m'])
    baseline = (torch.zeros(input_attr[0].shape).to(device),
                torch.zeros(input_attr[1].shape).to(device))
                #
    ig = IntegratedGradients(net)
    with torch.no_grad():
        attributions, delta = ig.attribute(input_attr, baseline, target=0, return_convergence_delta=True)

    attr_new = np.concatenate([attr.cpu().detach().numpy() for attr in attributions],axis=1)
    attr = attr_new

    df_attr = pd.DataFrame(attr).transpose() # n_feature x n_cell

    def filter_pd100nm(x):
        if 'PD100nM' in x:
            s = 'PD100nM'
        else:
            if 'EGF' in x :
                if 'EGF0' not in x:
                    s = 'EGF-high'
                else:
                    s = 'EGF-low'
            else:
                s = x
        return s

    from scipy.stats import rankdata

    treatment_shorthand = split_meta_in.map(lambda x: x.split(" ")[0])
    treatment_shorthand = treatment_shorthand.map(lambda x: x.split(".")[0])
    treatment_shorthand = treatment_shorthand.map(lambda x: x.split("ng")[0])
    treatment_shorthand = treatment_shorthand.map(filter_pd100nm)
    col_colors = []
    col_color,lut2 = series2colors(treatment_shorthand,
                                   return_lut=True)
    col_colors.append(col_color.values)
    col_color,lut3 = series2colors(Y_in.cpu().detach().numpy().reshape(-1),
                                   palette=sns.cubehelix_palette(as_cmap=False),
                                   return_lut=True)
    col_colors.append(col_color.values)
    col_color, lut = series2colors(errors.reshape(-1),
                                   palette=sns.cubehelix_palette(as_cmap=False),
                                   return_lut=True)
    col_colors.append(col_color.values)


    row_colors = ['blue'] * 191 + ['red'] * 83

    g = sns.clustermap(data=df_attr,
                       row_cluster=False,
                       col_cluster=True,
                       row_colors=row_colors,
                       col_colors=col_colors,
                       cmap='RdBu',center=0
                       # cmap=sns.color_palette("rocket_r", as_cmap=True)
                       )

    plt.savefig(os.path.join(outdir_model,f'attr-{attr_method}-{target}-notscaled.svg'))
    plt.close()

    fig, ax = plt.subplots()

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=lut[name]) for name in lut]

    leg = plt.legend(handles, lut, title='Error')
    ax.add_artist(leg)

    handles = [Patch(facecolor=lut2[name]) for name in lut2]
    leg = plt.legend(handles, lut2, title='Treatment',loc='lower left')
    ax.add_artist(leg)

    handles = [Patch(facecolor=lut3[name]) for name in lut3]
    plt.legend(handles, lut3, title='Expression',loc='lower right')

    plt.savefig(os.path.join(outdir_model,f'attr-{attr_method}-{target}-notscaled-legends.svg'))
    plt.close()

    #### Plot ERK activity as time series for cells with high/low attribution
    #### and high/low gene expression
    exp = Y_in.cpu().detach().numpy().reshape(-1)
    exp_level = np.repeat('None',len(exp))
    exp_level[exp > np.percentile(exp,95)] = 'High'
    exp_level[exp < np.percentile(exp,50)] = 'Low'
    #
    attr_sum = np.abs(attr).sum(1)
    attr_level = np.repeat('None',len(attr_sum))
    attr_level[attr_sum > np.percentile(attr_sum,80)] = 'High'
    attr_level[attr_sum < np.percentile(attr_sum,10)] = 'Low'
    #
    np.where(np.all([(exp_level == 'High'),(attr_level == 'High')],axis=0))[0][:3]
    np.where(np.all([(exp_level == 'High'),(attr_level == 'Low')],axis=0))[0][:3]
    np.where(np.all([(exp_level == 'Low'),(attr_level == 'High')],axis=0))[0][:3]
    np.where(np.all([(exp_level == 'Low'),(attr_level == 'Low')],axis=0))[0][:3]
    #
    erk = inputs['x'].cpu().detach().numpy()
    #
    combos = [("High","High"),
              ("High","Low"),
              ("Low","High"),
              ("Low","Low")]
    colors = sns.color_palette()
    for color,(exp_cond,attr_cond) in zip(colors,combos):
        idx = np.where(np.all([(exp_level == exp_cond),(attr_level == attr_cond)],axis=0))[0][-1:]
        for i in idx:
            plt.plot(erk[i],color=color)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir_model,f'ERK_activity-{target}.svg'))
    plt.close()
    lut = dict(zip([f"ERK_{c[0]}-EXPR_{c[1]}" for c in combos],colors))
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=lut[name]) for name in lut]
    leg = plt.legend(handles, lut,bbox_to_anchor=(1.04,1), title='ERK-EXPR')
    plt.savefig(os.path.join(outdir_model,f'ERK_activity-{target}-legend.svg'))
    plt.close()
