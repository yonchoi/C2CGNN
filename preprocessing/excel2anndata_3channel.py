import os,gc

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import scanpy as sc

import argparse

from create_dataset import sample_time_window, TemporalDataLoader, combine_dataloader, generate_adj

parser = argparse.ArgumentParser()

parser.add_argument('--filedir', type=str, default='',
                    help='Directory to file where excel files are stored')

args = parser.parse_args()

filedir = args.filedir

# overwrite input arguments
filedir = "../data/Devan/3channel"

## Read excel files
path_x     = os.path.join(filedir,'df_x.xlsx') # x coordinate
path_y     = os.path.join(filedir,'df_y.xlsx') # y coordinate
path_size  = os.path.join(filedir,'df_narea.xlsx') # Cell size
path_w     = os.path.join(filedir,'Cell_Info.xlsx') # Well of Origin
# path_erk   = os.path.join(filedir,'df_erk.xlsx') # ERK activity time-series
path_etgs  = os.path.join(filedir,'df_ETGs.xlsx') # Protein expressions

reporters = ['ERK','AMPK','NFkB']
path_reporters = [os.path.join(filedir,f'{s}data.xlsx') for s in reporters]
path_params = [os.path.join(filedir,f'{s}_params.xlsx') for s in reporters]

dfs_list_reporters = [pd.read_excel(p, sheet_name='Sheet1', header=None) for p in path_reporters]
dfs_list_params    = [pd.read_excel(p, sheet_name='Sheet1', header=[0]) for p in path_params]
dfs_x = pd.read_excel(path_x, sheet_name='Sheet1', header=None)
dfs_y = pd.read_excel(path_y, sheet_name='Sheet1', header=None)
dfs_size = pd.read_excel(path_size, sheet_name='Sheet1', header=None)
dfs_w = pd.read_excel(path_w, sheet_name='Sheet1', header=[0])
dfs_etg = pd.read_excel(path_etgs, sheet_name='Sheet1', header=[0])

## Plot
fig,ax = plt.subplots()
for df,color,label in zip(dfs_list_reporters,['red','blue','green'],reporters):
    ax.plot(df.values[0],color=color,label=label)
plt.savefig(os.path.join(filedir,'signal.svg'))
plt.close()

## Save the excel files as annata in h5ad format
h5ad_dir = os.path.join(filedir,"combined_data_raw.h5ad")

if os.path.isfile(h5ad_dir):
    adata = sc.read_h5ad(h5ad_dir)
else:
    obs = dfs_w
    obs.columns = ['CellID','Well of Origin','Treatment']
    # obs['Size'] = dfs_size.values
    act_array = np.concatenate([df.values for df in dfs_list_reporters],axis=1)
    # Set var
    n_timepoint = dfs_list_reporters[0].shape[-1]
    reporter_type = np.repeat(reporters,n_timepoint)
    timepoints = np.tile(np.arange(n_timepoint),len(reporters))
    var = pd.DataFrame({"Reporter" : reporter_type,
                        "Timepoint": timepoints})
    #
    adata = sc.AnnData(act_array,obs=obs,var=var)
    #
    adata.obs['Well of Origin'] = adata.obs['Well of Origin'].astype('category')
    #
    adata.obsm['x'] = dfs_x.values
    adata.obsm['y'] = dfs_y.values
    #
    adata.obsm['Y'] = dfs_etg.values
    adata.uns['Y'] = dfs_etg.columns.values
    #
    param_array = np.concatenate([df.values for df in dfs_list_params],axis=1)
    adata.obsm['params'] = param_array
    adata.uns['params'] = dfs_list_params[0].columns.values
    #
    adata.write(h5ad_dir)
