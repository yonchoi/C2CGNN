import os,gc,glob

import seaborn as sns
import matplotlib.pyplot as plt

import scipy.io as spio
import numpy as np
import pandas as pd

from tqdm import tqdm

import scanpy as sc

from util.data_handler.multichannel_adata import DataHandler

DATA_DIR = 'data/Alex/S1T4/20170502 EXPT8B'
DATA_DIR = 'data/Alex/S1T4/20171030 S1 T4 Plate 2'

PLOT_DIR = os.path.join(DATA_DIR,'plot')
os.makedirs(PLOT_DIR, exist_ok=True)

filenames = glob.glob(os.path.join(DATA_DIR,'*.mat'))
filenames = [f for f in filenames if 'Global.mat' not in f]
filenames = [f for f in filenames if 'proc.mat' not in f]
filenames.sort()

Xs = []
Ws = []

min_nuc_value = 15

for filename in filenames:
    print(filename)
#
    mat_contents = spio.loadmat(filename, simplify_cells=True)
#
    if 'valcube' in mat_contents.keys():
        valcube = mat_contents['valcube']
        n_cell, n_timepoints , n_category = valcube.shape
        cat = mat_contents['vcorder'][:n_category]
        #
        reporters = [c.strip('_Nuc') for c in cat if c.endswith('Nuc')]
        XYcoord = ['XCoord', 'YCoord']
        ### Calculate the ratio between cyto and nuc
        idx_nuc = np.isin(cat, pd.Series(reporters) + '_Nuc')
        idx_cyt = np.any([np.isin(cat, pd.Series(reporters) + '_Cyto'),
                          np.isin(cat, pd.Series(reporters) + '_Cyt')], axis=0)
        idx_XY = np.isin(cat, XYcoord)
        val_cyt = valcube[:,:,idx_cyt]
        val_nuc = valcube[:,:,idx_nuc]
        X = val_cyt / val_nuc
        ## Set ratio where nucleus fluor. is below threshold to be 0
        X[val_nuc < min_nuc_value] = 0 # set
        X = np.concatenate([X,
                            valcube[:,:,idx_XY],
                            val_cyt,
                            val_nuc],axis=-1)
        catnames = reporters + XYcoord + list(cat[idx_cyt]) + list(cat[idx_nuc])
        Xs.append(X)
        W = np.repeat(filename,len(X))
        Ws.append(W)
        print(np.nanmin(valcube[:,:,idx_nuc]))
        print(np.nanmin(X[:,:,1:3]))
        print(np.nanmax(X[:,:,1:3]))

X_final = np.concatenate(Xs,axis=0)
W_final = np.concatenate(Ws,axis=0)
W_final = [w.strip('.mat').split('xy')[-1] for w in W_final]
W_final = np.array(W_final).astype('int')


# ==============================================================================
# Try reloading file
# ==============================================================================

npzname = os.path.join(DATA_DIR,'rawfiles.npz')

np.savez(npzname,
         X=X_final,
         W=W_final,
         C=catnames)

#### Try loading file
npz = np.load(npzname)
X_final = npz['X']
W_final = npz['W']
catnames = npz['C']

# ==============================================================================
# Assign S1,T4
# ==============================================================================

#### Plot S1reporter vs T4reporter
X_new = X_final[:,:,1:3]

for stat in ['mean','max']:
    if stat == 'mean':
        statfun = np.nanmean
    elif stat == 'max':
        statfun = np.nanmax
    df_plot = pd.DataFrame(statfun(X_new,1),columns=catnames[1:3])
    df_plot['Well'] = W_final
    df_plot.columns = ['S1','T4','Well']
    df_plot['S1T4'] = 'S1'
    df_plot['S1T4'][df_plot['T4'] > df_plot['S1']] = 'T4'
    palette = dict(zip(['S1','T4'],sns.color_palette()))
    g = sns.FacetGrid(data=df_plot, col='Well', col_wrap=10,sharey=False,sharex=False)
    g.map_dataframe(sns.scatterplot, x='S1', y='T4', hue = 'S1T4', palette=palette)
    plt.savefig(os.path.join(PLOT_DIR,f'VvsT_{stat}.svg'))
    plt.close()

#### Plot stats on NaNs
naninfo = np.any(np.isnan(X_final),axis=-1)

num_nan_tp = pd.Series(np.sum(naninfo,axis=-1)).value_counts()
num_nan_tp = num_nan_tp.sort_index()
g = sns.barplot(x=num_nan_tp.index, y=num_nan_tp.values)
plt.savefig(os.path.join(PLOT_DIR,f'NumNanTp.svg'))
plt.close()

# ==============================================================================
# Interpolate and deal with NaN values
# ==============================================================================

def max_na(s):
    s = pd.Series(s)
    isna = s.isna()
    blocks = (~isna).cumsum()
    return isna.groupby(blocks).sum().max()

def interpolate(X, type='linear', longest_nan=5):
    """
    Interpolate missing datas
    Input: dataframe with columns abstime, X, Y
    Output: dtaframe where X and Y has been interpolated
    ## To speed up the process, find ways to not have to loop through non-zero values
    """
    #
    invalid_nan_length = max_na(X) > longest_nan
    # Filter for samples that are valid for interpolation
    def nan_helper(y):
        """Helper to handle indices and logical indices of NaNs.
        Input:
            - y, 1d numpy array with possible NaNs
        Output:
            - nans, logical indices of NaNs
            - index, a function, with signature indices= index(logical_indices),
              to convert logical indices of NaNs to 'equivalent' indices
        Example:
            >>> # linear interpolation of NaNs
            >>> nans, x= nan_helper(y)
            >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
        """
        return np.isnan(y), lambda z: z.nonzero()[0]
    nans, x = nan_helper(X)
    if invalid_nan_length or np.sum(~nans) == 0:
        X_in = X
    else:
        X_in = np.array(X)
        X_in[nans] = np.interp(x(nans),x(~nans), X_in[~nans])
    return X_in

def apply_on_axis(X,fun,axis=1,**kwargs):
    """ """
    original_shape = list(X.shape)
    modified_shape = list(original_shape)
    _ = modified_shape.pop(axis)
    X_reshape = np.moveaxis(X,axis,-1)
    X_reshape = X_reshape.reshape(-1,original_shape[axis])
    y_reshape = [fun(x,**kwargs) for x in tqdm(X_reshape)]
    y_reshape = np.array(y_reshape)
    y_reshape = y_reshape.reshape(*modified_shape,-1)
    y_reshape = np.moveaxis(y_reshape,-1,axis)
    return y_reshape

# Set maximum continuous NaN length
max_na_num = apply_on_axis(X_final,
                           max_na,
                           axis=1)
nums = np.array([np.all(np.all(max_na_num < i, -1),-1) for i in np.arange(20)])
pd.Series(nums.sum(1))

longest_nan = 7
W_cell_dist = pd.Series(np.all(np.all((max_na_num < longest_nan),-1),-1)).groupby(W_final).sum()
pd.cut(W_cell_dist,bins=10).value_counts().sort_index()
pd.cut(pd.Series(W_final).value_counts(),bins=10).value_counts().sort_index()


df_well_org = pd.read_csv(os.path.join(DATA_DIR,'WellInfoReformated.csv'),
                          index_col=0)

(W_cell_dist > 100).groupby(df_well_org['CellType']).sum()
df_well_org['CellsValid'] = W_cell_dist

# Interpolate the NaNs
X_int = apply_on_axis(X_final,
                      interpolate,
                      axis=1,
                      longest_nan=longest_nan)

#### Save the updated to npz
npzname = os.path.join(DATA_DIR,'activity_interpolated.npz')

np.savez(npzname,
         X=X_int,
         W=W_final,
         C=catnames)

# Reload
npz = np.load(npzname)
X_int = npz['X']
W_final = npz['W']
catnames = npz['C']

# ==============================================================================
#                          Count number of viable cells
# ==============================================================================
from ASTGNN.create_dataset import sample_time_window, TemporalDataLoader, combine_dataloader, generate_adj

a = []
for well in np.unique(W_final):
    X_well = X_int[W_final == well]
    naninfo_well = np.any(np.any(np.isnan(X_well),1),1)
    XCoord = np.nanmean(X_well[:,:,3],1)
    YCoord = np.nanmean(X_well[:,:,4],1)
    #
    adj = generate_adj(XCoord.reshape(-1,1),
                       YCoord.reshape(-1,1),
                       num_neighbor=11)[0]
    num_neighbors = adj.sum(1)
    adj[:,naninfo_well] = 0
    num_neighbors_valid = adj.sum(1)
    valid_neighbors_ratio = num_neighbors_valid / num_neighbors
    a.append(valid_neighbors_ratio[~naninfo_well])
    print((valid_neighbors_ratio > 0.9).sum(),(~naninfo_well).sum(),)
    #

pd.cut(pd.Series(np.concatenate(a)),10).value_counts()
# ==============================================================================
# Assign cell type
# ==============================================================================

X_analysis = np.swapaxes(X_int,1,2) # n_cell x n_channel x n_timepoint

#### Assign CellType
reporters = catnames[1:3]
X_ct = X_analysis[:,np.isin(catnames,reporters)]

X_stat = np.nanmean(X_ct,-1)

eps = 0.1

CT = np.repeat('T4',len(X_ct))
diff = X_stat[:,0] - X_stat[:,1]
CT[diff > 0] = 'S1'
CT[np.abs(diff) < eps] = 'unassigned'


# ==============================================================================
# Save as anndata
# ==============================================================================

# Save as anndata
ANNDATA_DIR = os.path.join(DATA_DIR,'activity_interpolated.h5ad')

# Obs,var
df_well_org.index = df_well_org['XY']
obs_well = df_well_org.copy()
obs_well['Treatment'] = obs_well['pre-treatment']
obs = obs_well.reindex(W_final)
obs["Well of Origin"] = obs.XY
obs = obs.reset_index(drop=True)
obs['CellTypeWell'] = obs['CellType']
obs['CellType'] = CT
obsm = {'x': X_int[:,:,np.where(np.isin(catnames,'XCoord'))[0][0]],
        'y': X_int[:,:,np.where(np.isin(catnames,'YCoord'))[0][0]]}

# Datahandler for anndata
channel_names = ['ERKTR-mTurq', 'ERKTR-mVenus']
X = X_analysis[:,np.isin(catnames,channel_names)]
dh = DataHandler(X=X,
                 obs=obs,
                 obsm=obsm,
                 channel_name=channel_names)

dh.save(ANNDATA_DIR)


# ==============================================================================
# Plot samples
# ==============================================================================

# Filter out NaN cells
keep_idx = ~np.any(np.any(np.isnan(X_int),-1),-1)
X_int = X_int[keep_idx]
W_final = W_final[keep_idx]
obs = obs.iloc[keep_idx]

# sample_index = np.random.choice(np.arange(len(X_int)),100, replace=False)
sample_index = np.arange(len(X_int))

dfs = []

for f in [1,6,9,2,7,10]:
    X_f = X_int[:,:,f]
    df = pd.DataFrame(X_f[sample_index])
    df = pd.concat([df,obs.reset_index(drop=True)],axis=1)
    df['CellID'] = df.index
    df = df.melt(id_vars=obs.columns.to_list()+['CellID'],
                 var_name='Timepoint',
                 value_name= catnames[f])
    dfs.append(df)

df_plot = pd.concat(dfs,axis=1)
df_plot = df_plot.loc[:,~df_plot.columns.duplicated()]
df_plot = df_plot.melt(id_vars=obs.columns.to_list()+['CellID','Timepoint'],
                       var_name='Channel',value_name='Activity')

df_plot['Geno+Treatment'] = df_plot.Genotype + df_plot.Treatment

# for col in ['Genotype']:
for col in ['Geno+Treatment']:
    for ct in ['all','S1','T4']:
        if ct == 'all':
            df_plot_ = df_plot
        else:
            df_plot_ = df_plot[df_plot.CellType == ct]
        g = sns.FacetGrid(data=df_plot_,col=col,row='Channel',sharey=False)
        g.map_dataframe(sns.lineplot, x='Timepoint', y='Activity',units='CellID', estimator=None, lw=0.2)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR,f'sampled_lines_{col}_{ct}.pdf'))
        plt.close()

# ==============================================================================
# Visualize cells in well with X,Y coord
# ==============================================================================

from ASTGNN.create_dataset import sample_time_window, TemporalDataLoader, combine_dataloader, generate_adj

timerange = 'first50'

#### Plot cell type as X,Y
X_in = X_int[:,:50]
X_in = np.swapaxes(X_in,1,2) # n_cell x n_channel x n_timepoint
# X_t = X_in[:,:,0]
X_t = X_in.mean(-1)
X_coord = X_t[:,[1,2,3,4]]


df_plot = pd.concat([pd.DataFrame(X_coord,columns=['S1_ERK','T4_ERK','XCoord','YCoord']),
                     obs.reset_index(drop=True)],axis=1)
gt_dict = {'ERKTR-mTurq'  : 'S1g',
           'ERKTR-mVenus' : 'T4g',
           'ERKTR-mTurq; ERKTR-mVenus' : 'S1T4g'}

## Filte by well size
min_well_size=1
well_size = df_plot['Well of Origin'].value_counts()
wells_keep = well_size[well_size > min_well_size].index
df_plot = df_plot[np.isin(df_plot['Well of Origin'],wells_keep)]


## Set conditions/wells
wells = df_plot['Well of Origin'].unique()
conditions = df_plot.Treatment.unique()
setting = [['S1','ERKTR-mTurq','S1_ERK',None],
           ['T4','ERKTR-mVenus','T4_ERK',None],
           ['S1','ERKTR-mTurq; ERKTR-mVenus','S1_ERK',None],
           ['T4','ERKTR-mTurq; ERKTR-mVenus','T4_ERK',None],
           ['S1','ERKTR-mTurq; ERKTR-mVenus','S1_ERK','S1'],
           ['S1','ERKTR-mTurq; ERKTR-mVenus','S1_ERK','T4'],
           ['T4','ERKTR-mTurq; ERKTR-mVenus','T4_ERK','S1'],
           ['T4','ERKTR-mTurq; ERKTR-mVenus','T4_ERK','T4'],
          ]



def get_high_neighbor_cell(adj, CT, ct_queue, ct_neighbor,
                           percentile=None, top_n=None):
    num_neighbor = adj[:,CT==ct_neighbor].sum(1)
    num_neighbor[CT!=ct_queue] = 0
    if top_n is not None:
        idx_top_neighbor = np.argsort(-num_neighbor)[:top_n]
    elif percentile is not None:
        pass
    return np.isin(np.arange(len(num_neighbor)),idx_top_neighbor)

figdim   = np.array([len(conditions),len(setting)]) # (n_row,n_col)
fig,axes = plt.subplots(*figdim,figsize=figdim[::-1]*4)

for j,(axs,cond) in enumerate(zip(axes,conditions)):
    df_cond = df_plot[df_plot.Treatment == cond]
    #
    cond = df_cond.Treatment.unique()[0]
    #
    for i,(ct,gt,rep,neigh) in enumerate(setting):
        ax = axs[i]
        ## Subset
        df_subset = df_cond[df_cond.Genotype == gt]
        if len(df_subset) > 0:
            # Select well with larger number of cells
            largest_well = df_subset['Well of Origin'].value_counts().sort_values(ascending=False).index[0]
            df_subset = df_subset[df_subset['Well of Origin'] == largest_well]
            # Select cells with high number of neighbors
            adj = generate_adj(df_subset.XCoord.values.reshape(-1,1),
                               df_subset.YCoord.values.reshape(-1,1),
                               min_dist=100)[0]
            if neigh is not None:
                idx = get_high_neighbor_cell(adj,df_subset.CellType, ct, neigh, top_n=20)
            else:
                idx = np.repeat(True,len(df_subset))
            df_subset = df_subset[(df_subset.CellType == ct) & idx]
        ##  Plot
        if i == 0:
            _ = ax.set_ylabel(cond)
        if j == 0:
            _ = ax.set_title(f"{gt_dict[gt]} | {ct} | {rep} | {neigh}")
        if len(df_subset) > 0:
            ax = sns.scatterplot(data=df_subset, ax=ax,
                                 x='XCoord', y='YCoord', hue=rep)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR,f'WellXY_{timerange}.svg'))
plt.close()

# ==============================================================================
# Plot distribution of reporter in replicated wells
# ==============================================================================

from ASTGNN.create_dataset import sample_time_window, TemporalDataLoader, combine_dataloader, generate_adj

timerange = 'all'

#### Plot cell type as X,Y
X_in = X_int
X_in = np.swapaxes(X_in,1,2) # n_cell x n_channel x n_timepoint
# X_t = X_in[:,:,0]
X_t = X_in.mean(-1)
X_coord = X_t[:,[1,2,3,4]]


df_plot = pd.concat([pd.DataFrame(X_coord,columns=['S1_ERK','T4_ERK','XCoord','YCoord']),
                     obs.reset_index(drop=True)],axis=1)
gt_dict = {'ERKTR-mTurq'  : 'S1g',
           'ERKTR-mVenus' : 'T4g',
           'ERKTR-mTurq; ERKTR-mVenus' : 'S1T4g'}

## Filte by well size
min_well_size=1
well_size = df_plot['Well of Origin'].value_counts()
wells_keep = well_size[well_size > min_well_size].index
df_plot = df_plot[np.isin(df_plot['Well of Origin'],wells_keep)]

df_plot = df_plot[df_plot.CellType != 'un']
df_plot['CT|GT'] = df_plot.CellType + "|" + df_plot.Genotype.map(gt_dict)
df_plot['ERK'] = 0
df_plot['ERK'][df_plot.CellType == 'S1'] = df_plot.S1_ERK
df_plot['ERK'][df_plot.CellType == 'T4'] = df_plot.T4_ERK
df_plot['Well of Origin'] = df_plot['Well of Origin'].astype('str')
df_plot['Well of Origin Str'] = 'Well-' + df_plot['Well of Origin'].astype('str')

df_plot = df_plot[(df_plot['CT|GT'] != 'S1|T4g') & (df_plot['CT|GT'] != 'T4|S1g')]

## Set conditions/wells
wells = df_plot['Well of Origin'].unique()
conditions = df_plot.Treatment.unique()

g = sns.FacetGrid(data=df_plot, row='Treatment', col='CT|GT', sharey=False, sharex=False)
g.map_dataframe(sns.histplot,x='ERK', hue="Well of Origin", alpha=0.5)
plt.savefig(os.path.join(PLOT_DIR,f'HistPlot_Replicates_{timerange}.svg'))
plt.close()

g = sns.FacetGrid(data=df_plot, row='Treatment', col='CT|GT', sharey=False, sharex=False)
g.map_dataframe(sns.histplot,x='Well of Origin', y='ERK')
plt.savefig(os.path.join(PLOT_DIR,f'ViolinPlot_Replicates_{timerange}.svg'))
plt.close()

g = sns.catplot(x="Treatment", y="ERK",
                col='CT|GT', hue = "Well of Origin Str",
                data=df_plot, kind="violin",
                sharey=False,
                sharex=False,
                dodge=False);
g.set_xticklabels(rotation=90)
plt.savefig(os.path.join(PLOT_DIR,f'ViolinPlot_Replicates_{timerange}.svg'))
plt.close()

g = sns.catplot(row="Treatment", y="ERK", hue='CellType',
                col='CT|GT', x = "Well of Origin Str",
                data=df_plot, kind="violin",
                margin_titles=False,
                sharey=True,
                sharex=False,
                dodge=False);
g.set_xticklabels(rotation=90)

for ax,m in zip(g.axes[0,:],g.__dict__['col_names']):
    ax.set_title(m, fontweight='bold')
for ax,l in zip(g.axes[:,0],g.__dict__['row_names']):
    ax.set_ylabel(l, fontweight='bold', rotation=90, ha='right', va='center')

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR,f'ViolinPlot_Replicates_{timerange}.svg'))
plt.close()

## Per cell type
for ct in df_plot.CellType.unique():
    df_plot_cell = df_plot[df_plot.CellType == ct]
    g = sns.catplot(col="Treatment", y="ERK", hue='CT|GT',
                    x = "Well of Origin Str",
                    data=df_plot_cell, kind="violin",
                    margin_titles=False,
                    sharey=True,
                    sharex=False,
                    dodge=False);
    g.set_xticklabels(rotation=90)
#
    for ax,m in zip(g.axes[0,:],g.__dict__['col_names']):
        _ = ax.set_title(m, fontweight='bold')
    for ax,l in zip(g.axes[:,0],g.__dict__['row_names']):
        _ = ax.set_ylabel(l, fontweight='bold', rotation=90, ha='right', va='center')
#
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR,f'ViolinPlot_Replicates_{timerange}_{ct}.svg'))
    plt.close()



# ==============================================================================
# Compare expression in S1/T4 vs S1T4
# ==============================================================================
# use the same df_plot as above

## Set conditions/wells
wells = df_plot['Well of Origin'].unique()
conditions = df_plot.Treatment.unique()
setting = [['S1','ERKTR-mTurq','S1_ERK',None],
           ['T4','ERKTR-mVenus','T4_ERK',None],
           ['S1','ERKTR-mTurq; ERKTR-mVenus','S1_ERK',None],
           ['T4','ERKTR-mTurq; ERKTR-mVenus','T4_ERK',None],
           ['S1','ERKTR-mTurq; ERKTR-mVenus','S1_ERK','S1'],
           ['S1','ERKTR-mTurq; ERKTR-mVenus','S1_ERK','T4'],
           ['T4','ERKTR-mTurq; ERKTR-mVenus','T4_ERK','S1'],
           ['T4','ERKTR-mTurq; ERKTR-mVenus','T4_ERK','T4'],
          ]

figdim   = np.array([len(conditions),len(setting)]) # (n_row,n_col)
fig,axes = plt.subplots(*figdim,figsize=figdim[::-1]*4)

for j,(axs,cond) in enumerate(zip(axes,conditions)):
    df_cond = df_plot[df_plot.Treatment == cond]
    #
    cond = df_cond.Treatment.unique()[0]
    #
    for i,(ct,gt,rep,neigh) in enumerate(setting):
        ax = axs[i]
        ## Subset
        df_subset = df_cond[df_cond.Genotype == gt]
        if len(df_subset) > 0:
            # Select well with larger number of cells
            largest_well = df_subset['Well of Origin'].value_counts().sort_values(ascending=False).index[0]
            df_subset = df_subset[df_subset['Well of Origin'] == largest_well]
            # Select cells with high number of neighbors
            adj = generate_adj(df_subset.XCoord.values.reshape(-1,1),
                               df_subset.YCoord.values.reshape(-1,1),
                               min_dist=100)[0]
            if neigh is not None:
                idx = get_high_neighbor_cell(adj,df_subset.CellType, ct, neigh, top_n=20)
            else:
                idx = np.repeat(True,len(df_subset))
            df_subset = df_subset[(df_subset.CellType == ct) & idx]
        ##  Plot
        if i == 0:
            _ = ax.set_ylabel(cond)
        if j == 0:
            _ = ax.set_title(f"{gt_dict[gt]} | {ct} | {rep} | {neigh}")
        if len(df_subset) > 0:
            ax = sns.histplot(data=df_subset,x =rep, ax=ax)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR,f'HistPlot_{timerange}.svg'))
plt.close()

# ==============================================================================
# Old .mat files, confirm new matches with old for ratio calculation
# ==============================================================================
filenames = glob.glob(os.path.join(DATA_DIR,'*proc.mat'))

filename = filenames[0]
mat_contents = spio.loadmat(filename, simplify_cells=True)
mat_contents['d'][0]

X_old = np.array([mat_contents['d'][0]['data']['TKTR'],
                  mat_contents['d'][0]['data']['VKTR']])
X_old = np.moveaxis(X_old,0,-1)



# figdim = np.array([6,10])
# fig,axes = plt.subplots(*figdim,figsize=figdim[::-1]*2)
# axes = axes.flatten()
#
# for ax,w in zip(axes,np.unique(W_final)):
# #
#     X_new = X_final[[W_final == w]]
#     X_new = X_new[:,:,1:3]
# #
#     df_plot = pd.DataFrame(np.nanmean(X_new,1),columns=catnames[1:3])
#     g = sns.scatterplot(data=df_plot,
#                         x=catnames[1:3][0],
#                         y=catnames[1:3][1],
#                         ax=ax)
#     g.set_title(f'Well-{w}')
#
# plt.tight_layout()
# plt.savefig(os.path.join(PLOT_DIR,'VvsT.svg'))
# plt.close()
