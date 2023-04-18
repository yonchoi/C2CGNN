import os,gc

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import argparse

import scanpy as sc
from scipy.stats import rankdata

parser = argparse.ArgumentParser()

parser.add_argument('--anndatadir', type=str,
                    help='Directory to the anndata file')

parser.add_argument('--outdir', type=str, default='NA',
                    help='Directory where results will be saved')

parser.add_argument('--treatment', type=str, default='all',
                    help='Directory to the anndata file')

args = parser.parse_args()

anndatadir = args.anndatadir
treatment  = args.treatment
outdir     = args.outdir

# Overwrite input arguments
anndatadir = "data/Ram/ERK_ETGs_Replicate1/combined_data_raw.h5ad"

if outdir == 'NA':
    outdir = os.path.join(os.path.dirname(anndatadir),'plot')

## Setup directory
os.makedirs(outdir,exist_ok=True)

## Load anndata
adata = sc.read_h5ad(anndatadir)

## Filter out cells with NA values
idx_keep = np.invert(np.isnan(adata.X).any(1))
adata = adata[idx_keep]

## Filter by treatment
if treatment is 'all':
    adata_t = adata
else:
    adata_t = adata[adata.obs.Treatment == treatment]

## Set the variables
erk = adata_t.X
exp = adata_t.obsm['Y']
from sklearn.preprocessing import scale
exp = scale(exp,axis=0)

cellmeta = adata_t.obs
exp_columns = adata_t.uns['Y']

wells = adata.obs[['Treatment','Well of Origin']]
wells.index = wells["Well of Origin"]
wells_name = wells.drop_duplicates()
wells_name.index = wells_name["Well of Origin"]

from scipy.spatial.distance import cdist
from scipy.stats import rankdata

for name, (t,w) in wells_name.iterrows():
    # Load coord
    adata_w = adata[wells.index == name]
    X = adata_w.obsm['x']
    Y = adata_w.obsm['y']
    coord = np.array([X,Y]) # n_coord x n_cell x n_time
    coord = np.swapaxes(coord,0,2)
    # Load coord
    dist = np.array([cdist(c,c) for c in coord])
    dist_rank = rankdata(dist,axis=-1)
    adj = dist_rank < 10
    adj = adj.astype('float32')
    deltaAdj = (np.abs(adj[:-1] - adj[1:]).sum(axis=1).sum(axis=1)) / 2 / adj.sum(1).sum(1).mean(0)
    sns.lineplot(y=deltaAdj,
                 x=np.arange(len(deltaAdj)))
    plt.savefig(os.path.join(outdir,'deltaAdj_per_step.svg'))
    plt.close()
    deltaAdj = np.abs(adj[0] - adj).sum(axis=1).sum(axis=1) / 2 / adj.sum(1).sum(1).mean(0)
    sns.lineplot(y=deltaAdj,
                 x=np.arange(len(deltaAdj)))
    plt.savefig(os.path.join(outdir,'deltaAdj_per_ref.svg'))
    plt.close()
    #
    idx = np.round(np.linspace(0, len(coord) - 1, 10)).astype(int)
    # Choose one node to plot
    idx_node = 100
    idx_neighborhood = np.where(adj[0][idx_node])[0]
    for i in idx:
        coord_t = coord[i] # n_node x n_coord
        adj_t = adj[i] # n_node x n_node
        df_plot = pd.DataFrame(coord_t,columns=['x','y'])
        idx_neighborhood_curr = np.where(adj_t[idx_node])[0]
        df_plot['Type'] = 'None'
        df_plot['Type'][idx_neighborhood] = 'Neighbor_old'
        df_plot['Type'][idx_neighborhood_curr] = 'Neighbor'
        df_plot['Type'][idx_node] = 'Target'
        palette = {'None'     : 'gray',
                   'Neighbor' : 'orange',
                   'Neighbor_old' : 'green',
                   'Target'   : 'red'}
        sns.scatterplot(data=df_plot,
                        x = 'x',
                        y = 'y',
                        hue = 'Type',
                        palette = palette)
        plt.savefig(os.path.join(outdir,f'NeighborhoodPlot-{i}.svg'))
        plt.close()
