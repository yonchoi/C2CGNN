import os,gc

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import scanpy as sc

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--filedir', type=str, default='',
                    help='Directory to file where excel files are stored')
parser.add_argument('--datatype', type=str, default='',
                    help='Directory to file where excel files are stored')

args = parser.parse_args()

filedir  = args.filedir
datatype = args.datatype

# overwrite input arguments
filedir = "data/Ram/ERK_ETGs_Replicate1"
filedir = "data/Ram/All_Replicates_dataset"
datatype = 'Ram'

filedir = "data/Nick/16HBE14nEKARSpreadTest"
datatype = 'Nick'
# filedir = "data/Ram/All_Replicates_dataset"

## Read excel files
path_x     = os.path.join(filedir,'df_x.xlsx') # x coordinate
path_y     = os.path.join(filedir,'df_y.xlsx') # y coordinate
path_size  = os.path.join(filedir,'df_narea.xlsx') # Cell size
path_w     = os.path.join(filedir,'df_tx.xlsx') # Well of Origin
path_erk   = os.path.join(filedir,'df_erk.xlsx') # ERK activity time-series
path_etgs  = os.path.join(filedir,'df_ETGs.xlsx') # Protein expressions

engine = 'openpyxl'
dfs_size = pd.read_excel(path_size, sheet_name='Sheet1', header=None, engine=engine)
dfs_erk  = pd.read_excel(path_erk, sheet_name='Sheet1', header=None, engine=engine)
dfs_w    = pd.read_excel(path_w, sheet_name='Sheet1', header=[0], engine=engine)

etg_exist = os.path.isfile(path_etgs)

if len(dfs_size.shape) > 1:
    dfs_size = dfs_size.values[:,1:-1].mean(-1)

if etg_exist:
    dfs_etg  = pd.read_excel(path_etgs, sheet_name='Sheet1', header=[0], engine=engine)
    dfs_etg.columns = dfs_etg.columns.map(lambda s: s.replace('_nuc',''))

if isinstance(dfs_erk.values[0][0],str):
    dfs_erk = dfs_erk.iloc[1:]

if isinstance(dfs_size.values[0][0],str):
    dfs_size = dfs_size.iloc[1:]

if os.path.isfile(path_x):
    dfs_x = pd.read_excel(path_x, sheet_name='Sheet1', header=None)
else:
    dfs_x = None

if os.path.isfile(path_y):
    dfs_y = pd.read_excel(path_y, sheet_name='Sheet1', header=None)
else:
    dfs_y = None

if datatype == 'Ram':
    dfs_w['Treatment_shorthand'] = dfs_w.Treatment.map(lambda x: x.split(" ")[0])
    dfs_w['Treatment_shorthand2'] = dfs_w.Treatment_shorthand.map(lambda x: x.split("ng")[0])
    cat_mapping = {'EGF31.64': 'EGF-high',
                   'PD100nM' : 'PD100nM',
                   'Imagi'   : 'Imaging',
                   'EGF10'   : 'EGF-high',
                   'EGF3.164': 'EGF-high',
                   'EGF'     : 'EGF',
                   'EGF1'    : 'EGF-high',
                   'EGF0.1'  : 'EGF-low',
                   'EGF0.01' : 'EGF-low',
                   'EGF0.3164'  : 'EGF-low',
                   'EGF0.03164' : 'EGF-low',
    }
    dfs_w['Treatment_shorthand2'] = dfs_w['Treatment_shorthand2'].map(cat_mapping)
elif datatype == 'Devan':
    dfs_w['Treatment_shorthand2'] = dfs_w.Treatment.map(lambda x: x.split(" ")[0])
elif datatype == 'Nick':
    dfs_w['Treatment_shorthand'] = dfs_w.Treatment.map(lambda x: x.split(" ")[0])
    dfs_w['Treatment_shorthand2'] = dfs_w['Treatment_shorthand']

path_erk_param = os.path.join(filedir,'df_erk_paramterized.xlsx') # x coordinate
param_exist = os.path.isfile(path_erk_param)
if param_exist:
    dfs_erk_param = pd.read_excel(path_erk_param, sheet_name='Sheet1', header=[0])

## Save the excel files as annata in h5ad format
h5ad_dir = os.path.join(filedir,"combined_data_raw.h5ad")

if os.path.isfile(h5ad_dir):
    adata = sc.read_h5ad(h5ad_dir)
else:
    obs = dfs_w
    obs['Size'] = dfs_size.astype('float32')
    # var = pd.DataFrame({'Reporter'  : 'ERK',
    #                     'Timepoint' : np.arange(dfs_erk.shape[-1])})
    #
    adata = sc.AnnData(dfs_erk,obs=obs)
    adata.var['Reporter']  = 'ERK'
    adata.var['Timepoint'] = np.arange(dfs_erk.shape[-1])
    #
    adata.obs['Well of Origin'] = adata.obs['Well of Origin'].astype('category')
    #
    if dfs_x is not None:
        adata.obsm['x'] = dfs_x.values
    if dfs_y is not None:
        adata.obsm['y'] = dfs_y.values
    #
    if etg_exist:
        adata.obsm['Y'] = dfs_etg.values
        adata.uns['Y'] = dfs_etg.columns.values
    #
    if param_exist:
        adata.obsm['param'] = dfs_erk_param.values
        adata.uns['param'] = dfs_erk_param.columns.values
    #
    adata.write(h5ad_dir)


# ==============================================================================
# Subset
# ==============================================================================
path_subset  = os.path.join(filedir,'Subset_mask.xlsx') # subset_mask
h5ad_dir_subset = os.path.join(filedir,"combined_data_raw_subset.h5ad")
if os.path.isfile(path_subset):
#
    mask = pd.read_excel(path_subset, sheet_name='Sheet1', header=[0])
    mask = mask.values.flatten().astype('bool')
#
    adata_subset = adata[mask]
    adata_subset.write(h5ad_dir_subset)
