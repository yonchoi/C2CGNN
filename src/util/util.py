import numpy as np
import pandas as pd

import seaborn as sns
from sklearn.metrics import r2_score

def corrcoef(x,y,method='pearson'):

    if method == 'spearman':
        from scipy.stats import rankdata
        x=rankdata(x,axis=1)
        y=rankdata(y,axis=1)

    x_ = (x-x.mean(1).reshape(-1,1))/x.std(1).reshape(-1,1)
    y_ = (y-y.mean(1).reshape(-1,1))/y.std(1).reshape(-1,1)

    corr = np.mean(x_*y_,1)
    corr[np.isnan(corr)] = 0

    return corr

def r2coef(x,y):
    return np.array([r2_score(x1,y1) for x1,y1 in zip(x,y)])

from sklearn.preprocessing import scale

def scale_data(X,method='feature'):
    """ Scale input data X by either feature or sample """

    if method == 'feature':
        X = scale(X,axis=0)
    elif method == 'sample':
        X = scale(X,axis=1)
    elif method is None:
        X = X
    else:
        raise Exception("Input valid method")

    return X


def series2colors(x, palette='default',
                  lut=None, bin=True, n_cuts=5,
                  return_lut=False):
    """ Maps 1D list/array/series to colors """
#
    x = pd.Series(x)
    dtype = str(x.dtype)
#
    # Bin for floats/ints
    if 'int' in dtype or 'float' in dtype:
        if bin:
            x = pd.cut(pd.Series(x),n_cuts)
        if isinstance(palette,str):
            if palette=='default':
                palette = 'rocket_r'
    else:
        x = x.astype('category')
        if isinstance(palette,str):
            if palette=='default':
                palette = 'tab10'
#
    # Create mapping
    if lut is None:
        if isinstance(palette,str):
            palette = sns.color_palette(palette,n_colors=len(x.cat.categories))
        lut = dict(zip(x.cat.categories.values.astype('str'),
                       palette))
#
    # Map to color
    colors = x.astype('str').map(lut)
#
    if return_lut:
        return colors, lut
    else:
        return colors
