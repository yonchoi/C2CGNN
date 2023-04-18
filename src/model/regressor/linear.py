from src.util.util import corrcoef
from sklearn.linear_model import LinearRegression, Ridge, Lasso

import numpy as np
import torch

class LinearRegressor():
#
    def __init__(self, method='linear', **kwargs):
#
        self.method = method
        if method == 'linear':
            self.model = LinearRegression(**kwargs)
        elif method == 'ridge':
            self.model = Ridge(**kwargs)
        elif method == 'lasso':
            self.model = Lasso(**kwargs)
#
    def __call__(self,x,**kwargs):
        x = x.cpu().detach().numpy()
        return self.model.predict(x)
#
    def fit_transform(self,*args,**kwargs):
        return self.model.fit_transform(*args,**kwargs)
#
    def fit(self,*args,**kwargs):
        return self.model.fit(*args,**kwargs)
#
    def predict(self,*args,**kwargs):
        return self.model.predict(*args,**kwargs)
#
    def fit_dataloader(self, trainloader, testloader):
#
        dataset_train = trainloader.dataset[:]
        dataset_test  = testloader.dataset[:]
        #
        inputs = np.concatenate([dataset_train["X"].cpu().detach().numpy(),
                                 dataset_test["X"].cpu().detach().numpy()])
        targets = np.concatenate([dataset_train["Y"].cpu().detach().numpy(),
                                  dataset_test["Y"].cpu().detach().numpy()])
#
        self.model.fit(X=inputs,y=targets)
#
        return
