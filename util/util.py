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

def scale_data(X,method='feature'):

    if method == 'feature':
        X = scale(erk,axis=0)
    elif method == 'sample':
        X = scale(erk,axis=1)
    elif method is None:
        X = X
    else:
        raise Exception("Input valid method")

    return X

def series2colors(x, palette=sns.color_palette(), bin=True, n_cuts=5):
    """ Maps 1D list/array/series to colors """
    x = pd.Series(x)
    dtype = str(x.dtype)

    # Bin for floats/ints
    if 'int' in dtype or 'float' in dtype:
        if bin:
            x = pd.cut(pd.Series(x),n_cuts)
    else:
        x = x.astype('category')

    # Create mapping
    lut = dict(zip(x.cat.categories.values.astype('str'),
                   palette))

    # Map to color
    colors = x.astype('str').map(lut)

    return colors
