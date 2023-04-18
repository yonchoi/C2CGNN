import os,gc

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import argparse

import scanpy as sc

from create_dataset import sample_time_window, TemporalDataLoader, combine_dataloader, generate_adj

from util.data_handler.multichannel_adata import DataHandler

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()

parser.add_argument('--anndatadir', type=str,
                    help='Directory to the anndata file')
parser.add_argument('--outdir', type=str, default='NA',
                    help='Directory where results will be saved')
parser.add_argument('--treatment', type=str, default='all',
                    help='Directory to the anndata file')
parser.add_argument('--shift', type=str2bool, default=False,
                    help='If True, generate shifted data')

args = parser.parse_args()

anndatadir = args.anndatadir
treatment  = args.treatment
outdir     = args.outdir
shift      = args.shift

# Overwrite input arguments
anndatadir = "../data/Devan/3channel/combined_data_raw.h5ad"
anndatadir = "./data/Nick/16HBE14nEKARSpreadTest/combined_data_raw.h5ad"
anndatadir = "data/Alex/S1T4/20171030 S1 T4 Plate 2/activity_interpolated.h5ad"
# treatment = 'EGF-high'
# treatment = 'Imaging Media'
treatment = 'all'
scale_mode = 'time'

if outdir == 'NA':
    outdir = os.path.join(os.path.dirname(anndatadir),'data_processed',treatment,f'scale-{scale_mode}')

def find_local_cells(X,Y,xrange,yrange):
    cells_X = np.all([X >= xrange[0], X <= xrange[1]],axis=0)
    cells_Y = np.all([Y >= yrange[0], Y <= yrange[1]],axis=0)
    cells = np.all([cells_X,cells_Y],axis=0)
    return cells

def generate_shifted_cells(adata, n_cell=10, n_repeat=20, n_shift=1, n_neighbor=2):
    np.random.seed(0)
    sample_cell_idx = np.random.choice(np.arange(len(adata)),n_cell,replace=False)
    adata = adata[sample_cell_idx]
    length = adata.shape[-1] - n_repeat*n_shift
    adata_shifted = []
    for t in np.arange(n_repeat):
        adata_ = adata[:,(t*n_shift):(t*n_shift+length)]
        adata_.var.index = np.arange(adata_.shape[1])
        adata_shifted.append(adata_)
    adata_shifted = sc.concat(adata_shifted,axis=0)
    adata_shifted.obsm['x'] = adata_shifted.obsm['x'][:,:length]
    adata_shifted.obsm['y'] = adata_shifted.obsm['y'][:,:length]
    # Rearrange so the cells are indexed together
    n_node = len(adata_shifted)
    which_cell  = (np.arange(n_node) / n_repeat).astype('int')
    which_shift = np.arange(n_node) % n_repeat
    idx_rearrange = which_cell + which_shift * n_cell
    adata_shifted = adata_shifted[idx_rearrange]
    adj_settings = [f'TimeShift-{t}' for t in range(n_neighbor+1)]
    # Set adjacency matrices
    n_adj  = len(adj_settings)
    adj = np.zeros([n_adj, n_node, n_node]) # (n_adj, n_node, n_node)
    for i_adj in np.arange(n_adj):
        for i_node in np.arange(n_node):
            i_cell = int(i_node/n_repeat)
            i_start = max(i_cell*n_repeat,i_node - i_adj)
            i_end   = min(i_node + i_adj+1,(i_cell+1)*n_repeat)
            adj[i_adj][i_node][i_start:i_end] = 1
    return adata_shifted, adj, adj_settings

def plot_treatment(adata,filename="plot.svg",scale='cell-channel'):
    treatments = adata.obs.Treatment.unique()
    channels = adata.var.Channel.unique()
    fig,axes = plt.subplots(len(treatments),
                            figsize=(20,4*len(treatments)),
                            squeeze=False)
    axes = axes.flatten()
    for t,ax in zip(treatments,axes):
        adata_t = adata[adata.obs.Treatment==t]
        signals = adata_t.X.reshape(adata_t.X.shape[0],len(channels),-1) # (cell,n_channel,n_timepoint)
        signals = scale_3d(signals,axis=scale) # (cell,n_channel,n_timepoint)
        signals = signals[0] # (n_channel,n_timepoint)
        colors = ['red','blue','green']
        for channel,signal,color in zip(channels,signals,colors):
            ax.plot(signal,label=channel,color=color)
        ax.legend()
        ax.set_title(t)
    plt.savefig(filename)
    plt.close()

def plot_treatment_avg(adata,meta,filename="plot.svg",scale='cell-channel',use_meta=None,n_max=100):
    treatments = adata.obs.Treatment.unique()
    channels = adata.var.Channel.unique()
    fig,axes = plt.subplots(len(channels),len(treatments),
                            figsize=(20*len(treatments),4*len(channels)),
                            squeeze=False)
    # axes = axes.flatten()
    # Plot average
    X = adata.X # (n_cell, n_channel * n_timepoint)
    # #1
    X = X.reshape(len(X), len(channels), -1) # (n_cell, n_channel,n_timepoint)
    # #2
    # X = X.reshape(len(X), -1, len(channels)) # (n_cell, n_timepoint, n_channel)
    # X = np.moveaxis(X,1,2) # (n_cell, n_channel,n_timepoint)
    X = scale_3d_meta(X,meta=use_meta,axis=scale)
    for t,axes2 in zip(treatments,axes.transpose()):
        signals = X[t == meta]
        signals = np.moveaxis(signals,0,1) # (channel, cells, timepoint)
        colors = ['red','blue','green']
        for ax,channel,signal,color in zip(axes2,channels,signals,colors):
            # Plot individual lines
            signal_mean = signal.mean(0)
            _ = ax.plot(signal_mean,label=channel,color=color)
            # ax.legend()
            for s in signal[:n_max]:
                # signal = signal.mean(0)
                _ = ax.plot(s,color=color,alpha=0.1)
            _ = ax.set_title(t)
    plt.savefig(filename)
    plt.close()

def scale_3d_meta(x, meta, axis='cell-channel',**kwargs):
    """
    Perform scale_3d but split per meta
    Input:
        Inputs to scale_3d
        meta: 1d array used to subset x before performing scaling
    """
    scaled_data = np.zeros(x.shape)
    if meta is None:
        meta = np.zeros(len(x))
    assert len(meta) == len(x)
    for m in np.unique(meta):
        idx_meta = meta == m
        scaled_data[idx_meta] = scale_3d(x[idx_meta], axis=axis, **kwargs)
    return scaled_data

def scale_3d(x,axis='cell-channel',**kwargs):
    """
    Input: (n_cell,n_channel,n_time)
    axis: time then across timepoints, per cell/sensor combo
          cell then across
    """
    from sklearn.preprocessing import scale
    if axis == 'None':
        pass # return input
    else:
        sig_shape = x.shape
        if axis == 'cell-channel':
            x = x.reshape(-1,x.shape[-1]) # (cell * n_channel, time)
            x = scale(x,1) # scale for last input axis
            x = x.reshape(*sig_shape)
        # if axis == 'cell-channel':
        #     pass
        elif axis == 'time':
            x = x.reshape(-1,x.shape[-1]) # (cell * n_channel, time)
            x = scale(x,0) # scale for first input
        elif axis == 'time-channel':
            x = x.reshape(x.shape[0],-1) # (cell, n_channel * time)
            x = scale(x,0) # scale for first input
        elif axis == 'None':
            pass
        else:
            raise Exception('Input valid axis value [cell-channel, time, time-sensor]')
        x = x.reshape(*sig_shape)
    return x

def plot_corr(adata,filename):
    channels = adata.var.Channel.unique()
    signals = adata.X.reshape(adata.X.shape[0],len(channels),-1)
    sig_shape = signals.shape
    from sklearn.preprocessing import scale
    signals = signals.reshape(-1,signals.shape[-1])
    signals = scale(signals,1) # scale per node
    signals = signals.reshape(*sig_shape)
    signals = signals.reshape(signals.shape[0],-1)
#
    adata.X = signals
#
    treatments = adata.obs.Treatment.unique()
#
    from scipy.stats import rankdata
    corr = np.array([np.corrcoef(rankdata(c,axis=1)) for c in signals]) # n_cell x n_tx x n_tx
    ncols = 5
    nrows = int(np.ceil(len(treatments)/ncols))
    fig,axes=plt.subplots(nrows=nrows,ncols=ncols,figsize=(5*ncols,5*nrows))
    axes = axes.flatten()
#
    for t,ax in zip(treatments,axes):
        corr_t = np.abs(corr[adata.obs.Treatment == t]).mean(0)
        ax = sns.heatmap(corr_t,cmap='rocket_r',ax=ax, annot=True, fmt=".2g")
        _ = ax.set_title(t)
#
    plt.savefig(filename)
    plt.close()

## Setup directory
os.makedirs(outdir,exist_ok=True)

## Load anndata
adata = sc.read_h5ad(anndatadir)

## Filter out cells with NA values
idx_keep = np.invert(np.isnan(adata.X).any(1))
adata = adata[idx_keep]

# use_meta=None
# use_meta = adata.obs.Treatment
use_meta = adata.obs['Well of Origin']
use_meta = adata.obs['CellType'] +

meta = adata.obs.Treatment
use_meta = adata.obs.Treatment
metatype='Treatment'

for scale_ in ['cell-channel','time','time-channel','None']:
    plot_treatment(adata,filename=os.path.join(outdir,f'signal_treatment-{scale_}.svg'),scale=scale_)
    plot_treatment_avg(adata,meta,filename=os.path.join(outdir,f'SignalTreatmentAvg_{scale_}_meta-{metatype}.svg'),scale=scale_,use_meta=use_meta,n_max=100)

# plot_corr(adata,os.path.join(outdir,'corr_spearman_treatments.svg'))

## Filter by treatment
if treatment == 'all':
    adata_t = adata
elif treatment == 'EGF':
    adata_t = adata[adata.obs.Treatment.str.startswith('EGF')]
else:
    adata_t = adata[adata.obs.Treatment == treatment]

## Set the variables.
erk = adata_t.X

cellmeta = adata_t.obs

wells = adata_t.obs[['Treatment','Well of Origin']]
wells.index = wells["Well of Origin"]
wells_name = wells.drop_duplicates()
wells_name.index = wells_name["Well of Origin"]


# Filter wells based on min_cell count to keep n_nodes constant across wells
min_cell = 50
wells_count = wells.index.value_counts()
wells_keep = wells_count[wells_count > min_cell].index
wells_name = wells_name.loc[wells_keep]

time_start = 0 # index where stimulus is applied

adj_settings = ['dist-0','dist-60', 'dist-120', 'neighbor-5', 'neighbor-10']
adj_settings = ['dist-0','dist-60', 'dist-120', 'dist-240', 'neighbor-5', 'neighbor-10', 'neighbor-20']
# adj_settings = ['dist-0','dist-60', 'dist-120', 'dist-240', 'neighbor-5', 'neighbor-10', 'neighbor-20']
# adj_settings = ['dist-0','dist-60', 'dist-120', 'dist-240', 'neighbor-5', 'neighbor-10', 'neighbor-20']
adj_settings = ['identity','dist-60','dist-120']

sensors = adata.var['Channel'].unique()

adjs = []
samples = []

for ii, (name, (t,w)) in enumerate(wells_name.iterrows()):
    # Load coord
    adata_w = adata_t[wells.index == name]
    X = adata_w.obsm['x']
    Y = adata_w.obsm['y']
    center = np.array(((X.max() + X.min())/2,
                       (Y.max() + Y.min())/2))
    dist_from_center = np.sqrt(np.square(np.array((X - center[0],Y-center[1]))).sum(0))
    dist_from_center_avg = dist_from_center.mean(1)
    cell_keep = np.argsort(-dist_from_center_avg)[:min_cell]
    # filter X,Y
    adata_w = adata_w[cell_keep]
#
    X = adata_w.obsm['x']
    Y = adata_w.obsm['y']
    signal = adata_w.X
#
    adj = np.array([generate_adj(X,Y,s) for s in adj_settings])
    # Only keep time points after stimulus
    adjs.append(adj[:,time_start])
#
    ## Save dataset
    # adata_w = adata_w[:,time_start:]
    signal = signal.reshape(len(signal),len(sensors),-1) # n_cell x n_channel x n_timepoints
    seq = np.array(signal[:,:,time_start:])
    seq = scale_3d(seq,axis=scale_mode) # scale per node
#
    # Create samples
    # seq_in: (n_batch, n_node, n_time, n_feature)
    # seq_out: (n_batch, n_node, n_time, n_feature)
    seq_in = np.expand_dims(seq,0) # (n_batch, n_node, n_feature, n_timeponit)
    seq_in = np.swapaxes(seq_in,2,3) # (n_batch, n_node, n_timepoint, n_feature)
    seq_out = seq_in[:,:,:,0] # (n_batch, n_node, n_time)
    adj_idx = np.repeat(ii,len(seq_out)) # (n_batch)
    metadata = np.arange(len(seq_out)) # time point reference
    #
    sample = TemporalDataLoader(seq_in,seq_out,adj_idx,metadata)
    samples.append(sample)

samples = combine_dataloader(samples)
adjs = np.array(adjs)
samples.setAdjMtx(adjs,adj_settings)
samples.write(os.path.join(outdir,'samples.npz'))

## Save sample adjs
## TODO
