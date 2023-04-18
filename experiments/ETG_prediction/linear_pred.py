from sklearn.linear_model import LinearRegression, Ridge, Lasso

import numpy as np
import pandas as pd

import scanpy as sc

import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    """ Custom dataset """
    #
    def __init__(self, X, Y, M=None, P=None):
        self.X = X
        self.Y = Y
        self.M = M
        self.P = P
        #
    def __len__(self):
        return len(self.X)
        #
    def __getitem__(self, idx):
        sample = {"X" : self.X[idx],
                  "Y" : self.Y[idx]}
                  #
        if self.M is not None:
            sample['M'] = self.M[idx]
            #
        if self.P is not None:
            sample['P'] = self.P[idx]
            #
        return sample

# ==============================================================================
# Load data
# ==============================================================================

anndatadir = "data/Ram/ERK_ETGs_Replicate1/combined_data_raw.h5ad"

adata = sc.read_h5ad(anndatadir)

adata = adata[np.invert(np.isnan(adata.X).any(1))]

treatment = 'PD100nM'
adata = adata[adata.obs.Treatment == treatment]

X = adata.obsm["erk_param"]
Y = adata.obsm["Y"]
cellmeta = adata.obs

from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(handle_unknown='ignore')
M_well = enc.fit_transform(cellmeta[['Well of Origin']].values).toarray()
# M_size = cellmeta[['Size']].values

# M = np.concatenate([M_well,M_size],axis=1)
M = np.concatenate([M_well],axis=1)

P = adata.obsm["erk_param"]

batch_size = 128
X = torch.tensor(X).float()
Y = torch.tensor(Y).float()
M = torch.tensor(M).float()
P = torch.tensor(P).float()

from sklearn.model_selection import train_test_split

X, X_final, Y, Y_final, M, M_final, P, P_final = train_test_split(X,Y,M,P, train_size=0.8)
# X, X_final, Y, Y_final = train_test_split(X,Y, train_size=0.8)

print(X.shape)
print(Y.shape)
print(M.shape)

# Split train/test
X_train, X_test, Y_train, Y_test, M_train, M_test, P_train, P_test = train_test_split(X,Y,M,P, train_size=0.8)

# Create datasets
trainset = CustomDataset(X_train,Y_train,M_train,P_train)
testset  = CustomDataset(X_test,Y_test,M_test,P_test)
finalset  = CustomDataset(X_final,Y_final,M_final,P_final)

# Create dataloaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

finalloader = torch.utils.data.DataLoader(finalset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

# ==============================================================================

from src.util.util import corrcoef

dataset_dict = trainloader.dataset[:]
inputs = dataset_dict["P"]
targets = dataset_dict["Y"]

# lm = LinearRegression()
lm = Ridge()
lm.fit(inputs.numpy(),targets.numpy())

## Correlations for train,test sets

for train_type,dataloader in zip(['train','test','final'],
                        [trainloader,testloader,finalloader]):
                        #
  dataset_dict = dataloader.dataset[:]
  inputs = dataset_dict["P"].numpy()
  targets = dataset_dict["Y"].numpy()
  #
  outputs = lm.predict(inputs)
  #
  x = targets
  y = outputs
  #
  df_corr = pd.DataFrame({'spearman': corrcoef(x.transpose(),y.transpose(),method='spearman'),
                          'pearson': corrcoef(x.transpose(),y.transpose(),method='pearson')})
                          #
  df_corr['Target']    = adata.uns['Y']
  df_corr['Data']      = train_type
  df_corr['Model']     = 'Ridge'

  print(df_corr)
  print(df_corr.mean(0))

df_corr['Treatment'] = 'all'
df_corr['Input']     = 'ERK_P'

# ==============================================================================

import statsmodels.api as sm

dataset_dict = trainloader.dataset[:]
inputs = dataset_dict["P"].numpy()
targets = dataset_dict["Y"].numpy()

inputs = sm.add_constant(pd.DataFrame(inputs))

gamma_model = sm.GLM(targets, inputs, family=sm.families.Gaussian())
gamma_results = gamma_model.fit()

gamma_results.summary()

gamma_results.predict(inputs).shape

# ## Correlations for train,test sets

for train_type,dataloader in zip(['train','test','final'],
                        [trainloader,testloader,finalloader]):

  dataset_dict = dataloader.dataset[:]
  inputs = dataset_dict["P"].numpy()
  inputs = sm.add_constant(pd.DataFrame(inputs))

  targets = dataset_dict["Y"].numpy()

  outputs = gamma_results.predict(inputs)

  x = targets
  y = outputs
  y = y.values.reshape(len(x),-1)

  df_corr = pd.DataFrame({'spearman': corrcoef(x.transpose(),y.transpose(),method='spearman'),
                          'pearson': corrcoef(x.transpose(),y.transpose(),method='pearson')})

  print(df_corr)
  print(df_corr.mean(0))
