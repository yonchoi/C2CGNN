import os,gc

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


import scanpy as sc
from scipy.stats import rankdata

from util.invnorm import rank_INT

import argparse

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
# anndatadir = "data/Ram/ERK_ETGs_Replicate1/combined_data_raw.h5ad"
# anndatadir = "data/Devan/3channel/combined_data_raw.h5ad"
# anndatadir = "combined_data_raw.h5ad"
anndatadir = "data/Ram/All_Replicates_dataset/combined_data_raw.h5ad"
outdir =  "data/Ram/All_Replicates_dataset/plot"

anndatadir = "data/Ram/All_Replicates_dataset/combined_data_raw_subset.h5ad"
outdir =  "data/Ram/All_Replicates_dataset/plot_subset"

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

try:
    channels = adata.var.Reporter.unique()
    n_channel = len(channels)
except:
    channels = ['ERK']
    n_channel = 1

## Set the variables
erk = adata_t.X
exp = adata_t.obsm['Y']

from sklearn.preprocessing import scale
# exp = scale(exp,axis=0)

cellmeta = adata_t.obs
exp_columns = adata_t.uns['Y']

#### Correlation
def corr2_coeff(A, B, method = 'pearson'):
    """
    Input:
        A,B are matrix of size (n,f1), (n,f2)
    Return:
        Corelation matrix of size (f1,f2)
    """
    if method == 'spearman':
        from scipy.stats import rankdata
        A=rankdata(A,axis=1)
        B=rankdata(B,axis=1)
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]
    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)
    # Finally get corr coeff
    corr = np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None]))
    corr = np.array(corr)
    corr[np.isnan(corr)] = 0
    return corr


Y = adata_t.obsm['Y']
corr = corr2_coeff(erk.transpose(),Y.transpose())

# ==============================================================================
# Plot sampled ERK activity
# ==============================================================================

from src.util.util import scale_data
from sklearn.preprocessing import scale

## Randomly sample and plot
idx_sample = np.random.choice(np.arange(len(erk)),5,replace=False)

fig,axes = plt.subplots(3)
axes = axes.flatten()
for scale_method,ax in zip([None,'sample','feature'],
                           axes):
    X = scale_data(erk, method = scale_method)
    _ = ax.set_title(str(scale_method))
    for x in X[idx_sample]:
        ax.plot(x)

plt.tight_layout()
plt.savefig(os.path.join(outdir,'SampledERKActivity.svg'))
plt.close()

## Distribution of gene expressions per gene
df_exp = pd.DataFrame(exp,columns=exp_columns)
df_plot = df_exp.melt(var_name='Gene',value_name='Expression')

g = sns.FacetGrid(df_plot,
                  col="Gene",
                  col_wrap=4,
                  sharex=False,
                  sharey=False)
g.map(sns.distplot, "Expression")
plt.savefig(os.path.join(outdir,'DistributionETGs.svg'))
plt.close()

# ==============================================================================
# Plot gene to gene correlations
# ==============================================================================

corr_pearson  = np.corrcoef(exp.transpose())
corr_spearman = np.corrcoef(rankdata(exp.transpose(),axis=1))

for corr_type, corr in zip(('pearson','spearman'),
                [corr_pearson,corr_spearman]):
    #
    df_plot = pd.DataFrame(corr,
                           index=exp_columns,
                           columns=exp_columns)
    #
    g = sns.clustermap(df_plot,cmap='RdBu',center=0,
                       annot=True,annot_kws={"size": 15},fmt='.2f')
    _ = g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), size=18)
    _ = g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), size=18)
    # g.ax_heatmap.set_title(corr_type)
    plt.savefig(os.path.join(outdir,f'G2G_Corr_ERKActivity_{corr_type}.svg'))
    plt.close()


g1 = 'pcFos'
g2 = 'cFos'
x = exp[:,exp_columns == g1].flatten()
y = exp[:,exp_columns == g2].flatten()

sns.scatterplot(np.log(x+1),np.log(y+1), s=1, hue=adata.obs.Treatment_shorthand2)
plt.savefig(os.path.join(outdir,f'Scatter_{g1}_vs_{g2}.svg'))
plt.close()

# ==============================================================================
# Plot cell2cell correlations based on ERK activity
# ==============================================================================

# from scipy.stats import rankdata
#
# # get cell2cell correlation matrix based on ERK and gene expressions
# # ERK activity
# corr_pearson  = np.corrcoef(erk)
# corr_spearman = np.corrcoef(rankdata(erk,axis=1))
#
# # gene expression
# corr_pearson2  = np.corrcoef(exp)
# corr_spearman2 = np.corrcoef(rankdata(exp,axis=1))
#
# corr_dict = {'ERK_pearson' : corr_pearson,
#              'ERK_spearman': corr_spearman,
#              'ETG_pearson' : corr_pearson2,
#              'ETG_pearson' : corr_spearman2}
#
# from src.util.util import series2colors
# # Set row_colors as binned gene expressions for the 8 Immuno genes
# row_colors_multi = []
# for Y in exp.transpose():
#     row_colors = series2colors(Y,palette=['red','orange','yellow','green','blue','purple'])
#     row_colors_multi.append(row_colors)
#
# row_colors_ETG = row_colors_multi
#
# # Set row_colors as cell meta data
# row_colors_multi = []
# for Y,row_colors in cellmeta.iteritems():
#     if Y == 'Size':
#         row_colors = rankdata(row_colors)
#     row_colors = series2colors(row_colors,
#                                palette=sns.color_palette())
#     row_colors_multi.append(row_colors)
#
# row_colors_meta = row_colors_multi
#
# colors_dict = dict(zip(['ETG','meta'],
#                        [row_colors_ETG, row_colors_meta]))
#
# # Plot
# for color_type, row_colors in colors_dict.items():
#     for corr_type, corr in corr_dict.items():
#         g = sns.clustermap(corr,
#                            row_colors=row_colors,
#                            col_colors=row_colors,
#                            cmap='RdBu',
#                            center=0)
#         plt.savefig(os.path.join(outdir,'Cell2cell_Corr_{}-{}.svg'.format(corr_type,
#                                                                           color_type)))
#         plt.close()


## Distribution of the metadata
sns.countplot(cellmeta['Well of Origin'].astype('category'))
plt.savefig(os.path.join(outdir,'MetaDistribution_Well.svg'))
plt.close()

# Correlation between cell size and gene expression
Size = cellmeta['Size'].values.reshape(1,-1)
corrcoef(Size,np.array(exp).transpose(),method='spearman')

# ==============================================================================
# Visualize the PCA/tSNE/UMAP
# ==============================================================================

# palette = sns.color_palette("rocket_r", as_cmap=True)
palette = "Oranges"

## Calculate logged ETG expression
targets = adata.uns['Y']
logged = np.log(adata.obsm['Y']+1)
logged = np.nan_to_num(logged,logged[~np.isnan(logged)].min())
# logged[logged<3] = 3
# logged[logged>6] = 6
df_Y = pd.DataFrame(logged,columns=targets).reset_index(drop=True)

params = adata.uns['param']
df_param = pd.DataFrame(adata.obsm['param'],columns=params).reset_index(drop=True)

# for obsm in ['Y','params','X','param']:
for obsm in ['Y','param','X']:
# for obsm in ['Y']:
#
    if obsm in adata.obsm.keys() or obsm == 'X':
#
        if obsm == 'X':
            adata_param = adata
        else:
            adata_param = sc.AnnData(X=adata.obsm[obsm],
                                     var = pd.DataFrame(adata.uns[obsm],columns=['Feature']),
                                     obs = adata.obs)
        adata_param = adata_param.copy()
#
        # adata_param.obs['Treatment_shorthand'] = adata_param.obs.Treatment.map(lambda x: x.split(" ")[0])
        # adata_param.obs['Treatment_shorthand2'] = adata_param.obs.Treatment_shorthand.map(lambda x: x.split("ng")[0])
        # adata_param.obs['Treatment_shorthand2'] = adata_param.obs.Treatment.map(lambda x: x.split(" ")[0])
#
        adata_param.obs = pd.concat([adata_param.obs.reset_index(drop=True),df_Y,df_param],axis=1)
#
        # sc.pp.scale(adata_param)
#
        ## Visualize the cells with DR
        sc.tl.tsne(adata_param)
#
        cat2color = {'Treatment' : ['Treatment_shorthand2'],
                     'Targets'   : targets,
                     'Param'     : params}
#
        for category,colors in cat2color.items():
            print(category)
            # sc.tl.pca(adata_param, svd_solver='arpack')
            # sc.pl.pca(adata_param, color=colors, color_map=palette)
            # plt.savefig(os.path.join(outdir,f'PCA_Cell_{obsm}_{category}.svg'))
            # plt.close()
        #
            # sc.pp.neighbors(adata_param)
            # sc.tl.umap(adata_param)
            # sc.pl.umap(adata_param, color=colors, palette=palette)
            # plt.savefig(os.path.join(outdir,f'UMAP_Cell_{obsm}_{category}.svg'))
            # plt.close()
        #
            sc.pl.tsne(adata_param, color=colors, color_map=palette)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir,f'tSNE_Cell_{obsm}_{category}.svg'))
            plt.close()

# ==============================================================================
# Plot distribution of ETG after either log transform or IVN
# ==============================================================================

transform_types = ['log','raw','scale','invnorm']

for t_type in transform_types:
    if t_type == 'log':
        logged = np.log(adata.obsm['Y']-min(0,adata.obsm['Y'].min())+1)
        y = logged
    elif t_type == 'invnorm':
        y_invnorm = np.array([rank_INT(pd.Series(y)) for y in adata.obsm['Y'].transpose()]).transpose()
        y = y_invnorm
    elif t_type == 'scale':
        y = scale(adata.obsm['Y'],axis=0)
    elif t_type == 'raw':
        y = adata.obsm['Y']
    else:
        raise ValueError('t_type must be a valid transform type')
    # Format into df
    df_plot = pd.DataFrame(y,columns=adata.uns['Y'])
    df_plot['Treatment_shorthand2'] = adata.obs['Treatment_shorthand2'].values
    # Plot per treatment
    g = sns.FacetGrid(df_plot,col='Treatment_shorthand2',col_wrap=5,sharex=False)
    g.map(sns.histplot, df_plot.columns[0])
    plt.savefig(os.path.join(outdir,f'distETG_{t_type}.svg'))
    plt.close()
    # Plot for all cells
    g = sns.histplot(data=df_plot,x=df_plot.columns[0])
    plt.savefig(os.path.join(outdir,f'distETG_all_{t_type}.svg'))
    plt.close()

# ==============================================================================
# Plot correlation between sensor,ETG,params
# ==============================================================================

from src.util.util import corrcoef
X = adata.X
Y = adata.obsm['Y']

try:
    P = adata.obsm['erk_param']
except:
    P = adata.obsm['param']

values_dict = {'ERK': X, 'ETG': Y, 'ERK_P': P}
# labels_dict = {'ERK': np.arange(X.shape[1]),
#                'ERK_P': pd.Series(adata.uns["erk_param"]).map(lambda x:x.split(" (")[0])}
labels_dict = {'ERK': np.arange(X.shape[1]/n_channel),
               'ERK_P': pd.Series(adata.uns["param"]).map(lambda x:x.split(" (")[0])}

ETG_order = ['pERK','FRA1','EGR1','pRB','cMyc','cFos','pcFos','cJun']
ETG_order = exp_columns

for method in ('pearson','spearman'):
    dfs = []
    for m1,m2 in [['ERK','ETG'],['ERK_P','ETG']]:
        x1 = values_dict[m1]
        x2 = values_dict[m2]
        #
        corr = corr2_coeff(x1.transpose(),x2.transpose(),method=method)
        df_corr = pd.DataFrame(corr,
                               columns = exp_columns,
                               index   = np.tile(labels_dict[m1],n_channel))
        df_corr = df_corr[ETG_order]
        #
        df_corr.to_csv(os.path.join(outdir,f'Corr-{method}-{m1}-{m2}.csv'))
        #
        annot = len(df_corr) < 20
        luts = dict(zip(channels,sns.color_palette()))
        colors = np.repeat(channels,len(labels_dict[m1]))
        colors = pd.Series(colors).astype('str').map(luts).values
        g = sns.clustermap(df_corr.transpose(),
                       row_cluster=False,
                       col_cluster=False,
                       col_colors = colors,
                       cmap='RdBu',
                       center=0,
                       annot=annot,
                       fmt=".2f")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir,f'Corr-{method}-{m1}-{m2}.svg'))
        plt.close()
        dfs.append(df_corr)
        if n_channel > 1:
            for target in df_corr.columns:
                df_plot = df_corr[target] # (n_feature*n_channel) x 1
                df_plot = df_plot.values.reshape(n_channel,len(labels_dict[m1])) # n_channel x n_feature
                df_plot = pd.DataFrame(df_plot,
                                       columns=labels_dict[m1],
                                       index=channels)
                colors = pd.Series(channels).astype('str').map(luts).values
                #
                g = sns.clustermap(df_plot,
                               row_cluster=False,
                               col_cluster=False,
                               row_colors = colors,
                               cmap='RdBu',
                               center=0,
                               annot=annot,
                               fmt=".2f")
                plt.tight_layout()
                plt.savefig(os.path.join(outdir,f'Corr-{method}-{m1}-{m2}-{target}.svg'))
                plt.close()
    df_max = pd.concat([df_corr.abs().max() for df_corr in dfs],axis=1)
    df_max.columns = ['Activity','Params']
    g = sns.heatmap(df_max,
                   cmap='RdBu',
                   center=0,
                   annot=True,
                   fmt=".2f")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir,f'CorrMax-{method}-{m1}_{m2}.svg'))
    plt.close()
    ## Per target
    # if n_channel > 1:
    #     for target in df_max.columns:
    #         df_plot = df_corr[target] # (n_feature*n_channel) x 1
    #         df_plot = df_plot.values.reshape(n_channel,len(labels_dict[m1])) # n_channel x n_feature
    #         df_plot = pd.DataFrame(df_plot,
    #                                columns=labels_dict[m1],
    #                                index=channels)
    #         colors = pd.Series(channels).astype('str').map(luts).values
    #         #
    #         g = sns.clustermap(df_plot,
    #                        row_cluster=False,
    #                        col_cluster=False,
    #                        row_colors = colors,
    #                        cmap='RdBu',
    #                        center=0,
    #                        annot=annot,
    #                        fmt=".2f")
    #         plt.tight_layout()
    #         plt.savefig(os.path.join(outdir,f'Corr-{method}-{m1}-{m2}-{target}.svg'))
    #         plt.close()
