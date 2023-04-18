import numpy as np
import pandas as pd

import scanpy as sc
from sklearn.preprocessing import scale
from scipy.stats import rankdata

# Need to define input, output

def transform(Y,t_type):
    if t_type == 'log':
        Y = np.log(Y-min(0,Y.min())+1)
        Y = scale(Y,axis=0)
    elif t_type == 'invnorm':
        Y = np.array([rank_INT(pd.Series(y)) for y in Y.transpose()]).transpose()
        Y = scale(Y,axis=0)
    elif t_type == 'scale':
        Y = scale(Y,axis=0)
    elif t_type == 'raw':
        pass
    else:
        raise ValueError('t_type must be a valid transform type')
    return Y


def get_n_channel(adata):

    try:
        channels = adata.var.Reporter.unique()
        n_channel = len(channels)
    except:
        n_channel = 1

    return n_channel


def get_data(adata,name,t_type='raw',subset='all'):

    ## Set X, input
    inputs = name.split("-")

    try:
        channels = adata.var.Reporter.unique()
        n_channel = len(channels)
    except:
        n_channel = 1

    Xs = []
    for inp in inputs:

        if inp == 'ERK':
            if n_channel == 1:
                X = adata.X
            else:
                X = adata.X
                # X = adata.X.reshape(adata.X.shape[0],len(channels),-1)
                # X = np.swapaxes(X,1,2)
        elif inp == 'param':
            try:
                X = scale(adata.obsm['params'],axis=0)
            except:
                X = scale(adata.obsm['param'],axis=0)
        elif inp == 'ETG' or inp == 'Y':
            X = adata.obsm["Y"]
        else:
            raise Exception('Input argument must be start with either ERK or ERK_P')

        if subset != 'all':

            if subset.startswith('pca'):

                from sklearn.decomposition import PCA
                pca = PCA(5)
                X = pca.fit_transform(X)

                if subset != 'pca':
                    comp = int(subset.lstrip('pca'))
                    X = X[:,[comp]]

            else:

                if inp == 'ERK':

                    sign,lim = subset[0],int(subset[1:])

                    if sign == '+':
                        X = X[:,lim:]
                    elif sign == '-':
                        X = X[:,:lim]
                    else:
                        raise Exception('subset name must start with +/- for input type ERK')

                else:

                    targets = adata.uns[inp]
                    X = X[:,np.isin(targets,subset)]

        X = transform(X,t_type)

        Xs.append(X)

    X = np.concatenate(Xs,axis=1)

    return X


def remove_outlier(adata,Y,keep_percentile):
    """"""
    if keep_percentile > 0:
        Y_index = np.arange(len(Y))
        cells_keep_idx = []
        for t in adata.obs.Treatment.unique():
            cells_treatment = adata.obs.Treatment == t
            Y_t = Y[cells_treatment]
            Y_index_t = Y_index[cells_treatment]
            #
            cutoff_min = len(Y_t) * keep_percentile
            cutoff_max = len(Y_t) * (1 - keep_percentile)
            #
            cells_ranked = rankdata(Y_t,axis=0)
            # Get cells with value between cutoff_min and cutoff_max
            cells_keep = np.all([(cells_ranked >= cutoff_min),(cells_ranked <= cutoff_max)],axis=0)
            # Get cells whose every features for Y satisfy the above requirement
            cells_keep = np.all(cells_keep, axis=1)
            #
            cells_keep_idx.append(Y_index_t[cells_keep])
            len(Y_index_t[cells_keep])/len(Y_index_t)
        # Subset
        cells_keep = np.concatenate(cells_keep_idx)
        adata = adata[cells_keep]
        Y = Y[cells_keep]
    return adata,Y

def filter_cells(adata, treatment):

    # Filter out nan
    adata = adata[np.invert(np.isnan(adata.X).any(1))]

    if treatment != 'all':
        adata = adata[adata.obs.Treatment.str.replace("/",".") == treatment]

    assert len(adata) != 0, 'No cells left after initial filtering'

    return adata


def get_meta(adata):
    ## Set M, meta
    from sklearn.preprocessing import OneHotEncoder

    cellmeta = adata.obs
    cellmeta = cellmeta[['Well of Origin','Treatment']]

    cellmeta['plot_treatment'] = pd.Categorical(cellmeta.Treatment.map(lambda s: s.split(" ",1)[0]))

    enc = OneHotEncoder(handle_unknown='ignore')
    M_well = enc.fit_transform(cellmeta[['Well of Origin']].values).toarray()

    M = np.concatenate([M_well],axis=1)
    M = scale(M,axis=0)

    split_meta = cellmeta.astype('str').sum(1)

    return M, split_meta


def get_cell(adata):

    cellmeta = adata.obs
    C = cellmeta.Treatment

    return C

def get_param(adata):
    try:
        P = adata.obsm["param"]
    except:
        P = adata.obsm['params']
    return P
