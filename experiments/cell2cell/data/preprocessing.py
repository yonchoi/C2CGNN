import os,gc

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import argparse

import scanpy as sc

from create_dataset import sample_time_window, TemporalDataLoader, combine_dataloader, generate_adj

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
anndatadir = "../data/Ram/ERK_ETGs_Replicate1/combined_data_raw.h5ad"
# treatment = 'EGF-high'
treatment = 'Imaging Media'
shift=True

if outdir == 'NA':
    outdir = os.path.join(os.path.dirname(anndatadir),'data_processed',treatment,f'shift-{shift}')

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


## Setup directory
os.makedirs(outdir,exist_ok=True)

## Load anndata
adata = sc.read_h5ad(anndatadir)

## Filter out cells with NA values
idx_keep = np.invert(np.isnan(adata.X).any(1))
adata = adata[idx_keep]

## Filter by treatment
if treatment == 'all':
    adata_t = adata
elif treatment == 'EGF-high':
    treatments = ['EGF1ng/ml','EGF3.164ng/ml','EGF10ng/ml','EGF31.64ng/ml']
    adata_t = adata[np.isin(adata.obs.Treatment,treatments)]
else:
    adata_t = adata[adata.obs.Treatment == treatment]

## Set the variables
erk = adata_t.X
exp = adata_t.obsm['Y']
from sklearn.preprocessing import scale
exp = scale(exp,axis=0)

cellmeta = adata_t.obs
exp_columns = adata_t.uns['Y']

wells = adata_t.obs[['Treatment','Well of Origin']]
wells.index = wells["Well of Origin"]
wells_name = wells.drop_duplicates()
wells_name.index = wells_name["Well of Origin"]


# Filter wells based on min_cell count to keep n_nodes constant across wells
min_cell = 200
wells_count = wells.index.value_counts()
wells_keep = wells_count[wells_count > min_cell].index
wells_name = wells_name.loc[wells_keep]

time_start = 35 # index where stimulus is applied

adj_settings = ['dist-0','dist-60', 'dist-120', 'neighbor-5', 'neighbor-10']
adj_settings = ['dist-0','dist-60', 'dist-120', 'dist-240', 'neighbor-5', 'neighbor-10', 'neighbor-20']
# adj_settings = ['dist-0','dist-60', 'dist-120', 'dist-240', 'neighbor-5', 'neighbor-10', 'neighbor-20']
# adj_settings = ['dist-0','dist-60', 'dist-120', 'dist-240', 'neighbor-5', 'neighbor-10', 'neighbor-20']
adj_settings = ['identity','dist-0']

adjs = []
samples = []
for ii, (name, (t,w)) in enumerate(wells_name.iterrows()):
    # Load coord
    adata_w = adata_t[wells.index == name][:,:-1]
    X = adata_w.obsm['x']
    Y = adata_w.obsm['y']
    center = np.array(((X.max() + X.min())/2,
                       (Y.max() + Y.min())/2))
    dist_from_center = np.sqrt(np.square(np.array((X - center[0],Y-center[1]))).sum(0))
    dist_from_center_avg = dist_from_center.mean(1)
    cell_keep = np.argsort(-dist_from_center_avg)[:min_cell]
    # filter X,Y
    adata_w = adata_w[cell_keep]
    if shift:
        adata_w,adj,adj_settings = generate_shifted_cells(adata_w,
                                                          n_cell=10,
                                                          n_repeat=30,
                                                          n_shift=1)
    X = adata_w.obsm['x']
    Y = adata_w.obsm['y']
    signal = adata_w.X
#
    if shift:
        adj = np.broadcast_to(adj,(adata_w.shape[1],*adj.shape))
        adj = np.swapaxes(adj,0,1) # (n_time, n_adj, n_node, n_node)
        adjs.append(adj[:,time_start])
    else:
        adj = np.array([generate_adj(X,Y,s) for s in adj_settings])
        # Only keep time points after stimulus
        adjs.append(adj[:,time_start])
#
    ## Save dataset
    # adata_w = adata_w[:,time_start:]
    seq = np.array(signal[:,time_start:])
    from sklearn.preprocessing import scale
    seq = scale(seq,1) # scale per node
#
    # Create samples
    # seq_in: (n_batch, n_node, n_time, n_feature)
    # seq_out: (n_batch, n_node, n_time, n_feature)
    seq_in,seq_out = sample_time_window(seq, window_in=10, window_out=10)
    seq_out = seq_out[:,:,:,0] # (n_batch, n_node, n_time)
    adj_idx = np.repeat(ii,len(seq_out)) # (n_batch)
    metadata = np.arange(len(seq_out)) # time point reference
    #
    sample = TemporalDataLoader(seq_in,seq_out,adj_idx,metadata)
    samples.append(sample)
#
    # ## Plot change in adjacency matrix
    # deltaAdj = (np.abs(adj[:-1] - adj[1:]).sum(axis=1).sum(axis=1)) / 2 / adj.sum(1).sum(1).mean(0)
    # sns.lineplot(y=deltaAdj,
    #              x=np.arange(len(deltaAdj)))
    # plt.savefig(os.path.join(outdir,'deltaAdj_per_step.svg'))
    # plt.close()
    # deltaAdj = np.abs(adj[0] - adj).sum(axis=1).sum(axis=1) / 2 / adj.sum(1).sum(1).mean(0)
    # sns.lineplot(y=deltaAdj,
    #              x=np.arange(len(deltaAdj)))
    # plt.savefig(os.path.join(outdir,'deltaAdj_per_ref.svg'))
    # plt.close()
#
    ## Plot analysis for selected wells
    if ii in [0]:
        ## Plot snapshots of sampled cell and its neighbor
        coord = np.array([X,Y]) # n_coord x n_cell x n_time
        coord = np.swapaxes(coord,0,2) # n_time x n_cell x n_coord
        sample_time = np.round(np.linspace(0, len(coord) - 1, 5)).astype(int)
        # Choose one node to plot
        idx_adj = 1
        idx_time = 0
        for idx_node in [1,50,100]:
            idx_neighborhood = np.where(adj[idx_adj][idx_time][idx_node])[0]
            for i in sample_time:
                coord_t = coord[i] # n_node x n_coord
                adj_t = adj[idx_adj][i] # n_node x n_node
                df_plot = pd.DataFrame(coord_t,columns=['x','y'])
                idx_neighborhood_curr = np.where(adj_t[idx_node])[0]
                df_plot['Type'] = 'None'
                df_plot['Type'][idx_neighborhood] = 'Neighbor_old'
                df_plot['Type'][idx_neighborhood_curr] = 'Neighbor'
                df_plot['Type'][idx_node] = 'Target'
                palette = {'None'         : 'gray',
                           'Neighbor'     : 'orange',
                           'Neighbor_old' : 'green',
                           'Target'       : 'red'}
                sns.scatterplot(data=df_plot,
                                x = 'x',
                                y = 'y',
                                hue = 'Type',
                                palette = palette)
                plt.savefig(os.path.join(outdir,f'NeighborhoodPlot-time{i}-node{idx_node}.svg'))
                plt.close()
        # Plot ERK activities
        nodes = np.linspace(0,len(signal)-1,20).astype('int')
        fig,axes = plt.subplots(len(nodes),figsize=(20,4*len(nodes)),squeeze=False)
        axes = axes.flatten()
        for idx_node,ax in zip(nodes,axes):
            idx_neighborhood = np.where(adj[idx_adj][idx_time][idx_node])[0]
            # Assume (dist0,dist60,dist120,dist240)
            color_dict = dict(zip(adj_settings,['red','orange','green']))
            for i_, (adj_,setting) in enumerate(zip(adj,adj_settings[:3])):
                if i_ == 0:
                    adj_unique = adj_
                else:
                    adj_unique = (adj_ - adj_prev) == 1
                adj_prev = adj_
                idx_neighborhood = np.where(adj_unique[idx_time][idx_node])[0]
                signal_neighbors = signal[idx_neighborhood]
                if setting == 'dist-0' or (setting.startswith('TimeShift')):
                    for signal_node in signal_neighbors:
                        _ = ax.plot(range(len(signal_node)),
                                signal_node,
                                color=color_dict[setting],
                                label=setting)
                else:
                    signal_average = signal_neighbors.mean(0)
                    signal_std = signal_neighbors.std(0)
                    _ = ax.plot(range(len(signal_average)),
                            signal_average,
                            color=color_dict[setting],
                            ls="--",
                            label=setting)
                    _ = ax.fill_between(range(len(signal_average)),
                            signal_average - signal_std,
                            signal_average + signal_std,
                            facecolor=color_dict[setting],
                            alpha=0.05)
                ax.legend()
        #
        plt.savefig(os.path.join(outdir,f'NeighborhoodPlot-node-ERK.svg'))
        plt.close()
        #
    #     # ## Plot the correlation between nodes
    #     corr = np.abs(np.corrcoef(adata_w.X))
    #     adj_true  = adj[:,time_start].copy()
    #     adj_false = 1 - adj_true
    #     for i in range(len(adj_true)):
    #         np.fill_diagonal(adj_true[i],0)
    #     masked_corr_true = (adj_true * corr).transpose(1,2,0).sum(0).sum(0) / adj_true.sum(-1).sum(-1)
    #     masked_corr_true = np.nan_to_num(masked_corr_true,0)
    #     masked_corr_true
    #     masked_corr_false = (adj_false * corr).transpose(1,2,0).sum(0).sum(0) / adj_false.sum(-1).sum(-1)
    #     masked_corr_false = np.nan_to_num(masked_corr_false,0)
    #     masked_corr_false
    # #
    #     # ## Heatmap of local signal
    #     signal = adata_w.X[:,:-1]
    #     n_length = 20
    #     filter_size = 1
    #     heatmap_size = n_length - filter_size + 1
    #     xs = np.array(np.linspace(X.min(),X.max(),n_length+1))
    #     ys = np.array(np.linspace(Y.min(),Y.max(),n_length+1))
    #     signal_max,signal_min = signal.max(),signal.min()
    #     signal_local = -np.Inf * np.ones((heatmap_size,heatmap_size,signal.shape[-1],))
    #     for i in range(heatmap_size):
    #         for j in range(heatmap_size):
    #             xrange = (xs[i],xs[i+filter_size])
    #             yrange = (ys[j],ys[j+filter_size])
    #             cells_local_mask = find_local_cells(X,Y,xrange,yrange)
    #             signal_local[i,j] = np.sum(signal * cells_local_mask, axis=0) / np.sum(cells_local_mask,axis=0) # total signal / # cells
    #     signal_local = np.nan_to_num(signal_local,nan=-np.Inf)
    #     # Sample across time points than plot
    #     for t in range(signal_local.shape[-1]):
    #         signal_plot = signal_local[:,:,t]
    #         signal_plot = signal_plot.transpose() # row = y, columns = x
    #         # signal_plot = np.flip(signal_plot,axis=0)
    #         g = sns.heatmap(signal_plot,
    #                         vmin=signal_min,vmax=signal_max,
    #                         cmap=sns.color_palette("rocket_r", as_cmap=True))
    #         _ = g.invert_yaxis()
    #         plt.savefig(os.path.join(outdir,f'LocalSignalHeatmap-{t}.svg'))
    #         plt.close()
    #     # Plot x,y cell with signal as hue across time
    #     for t in range(signal_local.shape[-1]):
    #         signal_plot = signal_local[:,:,t]
    #         x_plot = X[:,t]
    #         y_plot = Y[:,t]
    #         signal_plot = signal[:,t]
    #         g = sns.scatterplot(x=x_plot,
    #                             y=y_plot,
    #                             hue=signal_plot,
    #                             vmin=signal_min,vmax=signal_max,
    #                             palette=sns.color_palette("rocket_r", as_cmap=True))
    #         plt.savefig(os.path.join(outdir,f'LocalSignalScatter-{t}.svg'))
    #         plt.close()

samples = combine_dataloader(samples)
adjs = np.array(adjs)
samples.setAdjMtx(adjs,adj_settings)
samples.write(os.path.join(outdir,'samples.npz'))

## Save sample adjs
## TODO
