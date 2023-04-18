import numpy as np

import scanpy as sc

def flatten_data(X):
    return X.reshape(len(X),-1), X.shape[1], X.shape[2]

class DataHandler():
    """
    """
    def __init__(self,
                 adata=None,
                 X=None,
                 obs=None,
                 var=None,
                 channel_name=None,
                 **kwargs):
        if adata is None:
    #
            if len(X.shape) > 2:
                X, n_channel, n_timepoint = flatten_data(X)
            else:
                n_channel = 1
    #
            n_timepoint = int(X.shape[1] / n_channel)
    #
            self.adata = sc.AnnData(X, obs, var, **kwargs)
    #
            self.n_channel   = n_channel
            self.n_timepoint = n_timepoint
            self.adata.uns['n_channel'] = n_channel
            self.adata.uns['n_timepoint'] = n_timepoint
    #
            if channel_name is None:
                channel_name = np.arange(n_channel)
    #
            self.adata.var['Channel']   = np.repeat(channel_name,n_timepoint)
            self.adata.var['Timepoint'] = np.tile(np.arange(n_timepoint), n_channel)
    #
        else:
    #
            self.adata = adata
            self.n_channel   = adata.uns['n_channel']
            self.n_timepoint = adata.uns['n_timepoint']
    #
    def save(self,filename):
        self.adata.write_h5ad(filename)
    #
    def get_3D_data(self):
        return self.adata.X.reshape(len(self.adata),self.n_channel,self.n_timepoint)
