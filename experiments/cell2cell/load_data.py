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

####
import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GConvGRU

class RecurrentGCN(torch.nn.Module):
    #
    def __init__(self, node_features, num_classes):
        super(RecurrentGCN, self).__init__()
        self.recurrent_1 = GConvGRU(node_features, 32, 5)
        self.recurrent_2 = GConvGRU(32, 16, 5)
        self.linear = torch.nn.Linear(16, num_classes)
    #
    def forward(self, x, edge_index, edge_weight):
        x = self.recurrent_1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.recurrent_2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.linear(x)
        return F.log_softmax(x, dim=1)
