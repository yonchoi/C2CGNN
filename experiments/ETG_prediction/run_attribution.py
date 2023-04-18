import os,gc

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

from src.model.regressor.FFN import CustomDataset
from src.util.util import series2colors

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import torch
from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
)

# import argparse
#
# parser = argparse.ArgumentParser()
#
# parser.add_argument('--modeldir', type=str,
#                     help='Directory to the pt file')
#
# args = parser.parse_args()
#
# modeldir  = args.modeldir

## Overwrite
# anndatadir = "data/Ram/ERK_ETGs_Replicate1/combined_data_raw.h5ad"
use_cuda = True
input  = 'ERK'
target = 'pERK'
treatment = 'all'

outdir = f'out/exp3_scaled-128/plot/{treatment}/{input}/'
os.makedirs(outdir,exist_ok=True)

targets = ['pERK','FRA1','EGR1','pRB','cMyc','cFos','pcFos','cJun']

attr_models = []

adata_orig = sc.read_h5ad(anndatadir)

attr_method = 'IG'

for model_type in ['conv-mixed','conv','flatten']:

    outdir_model = os.path.join(outdir,model_type)
    os.makedirs(outdir_model,exist_ok=True)

    attr_targets = []
    attr_targets2 = []

    for target in targets:

        modeldir = f'out/exp3_scaled-128/all/{input}/{target}/{model_type}/kfold-0/best_model.pt'

        # ==============================================================================
        # Load data
        # ==============================================================================
        from sklearn.preprocessing import scale

        # Proecss
        adata = adata_orig[np.invert(np.isnan(adata_orig.X).any(1))]

        if treatment != 'all':
            adata = adata[adata.obs.Treatment == treatment]

        assert len(adata) != 0, 'No cells left after initial filtering'

        ## Set X, input
        inputs = input.split("-")

        Xs = []
        for inp in inputs:
            if inp == 'ERK':
                X = adata.X
            elif inp == 'ERK_P':
                X = scale(adata.obsm['param'],axis=0)
            else:
                raise Exception('Input argument must be start with either ERK or ERK_P')
            Xs.append(X)

        X = np.concatenate(Xs,axis=1)

        ## Set Y, target genes
        if target == 'all':
            Y = adata.obsm["Y"]
        else:
            target_names = adata.uns["Y"]
            Y = adata.obsm["Y"][:,target==target_names]

        # Transform the predicted ETG
        from sklearn.preprocessing import scale
        if t_type == 'log':
            Y = np.log(Y-min(0,Y.min())+1)
        elif t_type == 'invnorm':
            Y = np.array([rank_INT(pd.Series(y)) for y in Y.transpose()]).transpose()
        elif t_type == 'scale':
            y = scale(Y,axis=0)
        elif t_type == 'raw':
            pass
        else:
            raise ValueError('t_type must be a valid transform type')

        ## Set M, meta
        from sklearn.preprocessing import OneHotEncoder

        cellmeta = adata.obs
        split_meta = cellmeta.astype('str').sum(1)

        enc = OneHotEncoder(handle_unknown='ignore')
        M_well = enc.fit_transform(cellmeta[['Well of Origin']].values).toarray()
        # M_size = cellmeta[['Size']].values

        # M = np.concatenate([M_well,M_size],axis=1)
        M = np.concatenate([M_well],axis=1)
        M = scale(M,axis=0)

        ## Set P if required
        P = adata.obsm["param"]
        P = scale(P,axis=0)

        X = torch.tensor(X).float()
        Y = torch.tensor(Y).float()
        M = torch.tensor(M).float()
        P = torch.tensor(P).float()

        device = torch.device("cuda" if use_cuda else "cpu")

        if use_cuda:
            X,Y,M,P = X.to(device), Y.to(device), M.to(device), P.to(device)

        X,_,Y,_,M,_,P,_,cellmeta,_ = train_test_split(X,Y,M,P,cellmeta,
                                           train_size=0.01,
                                           stratify=split_meta)

        # ==============================================================================
        # Attribution
        # ==============================================================================

        # ------------------------------------------------------------------------------
        # Data loader
        dataset = CustomDataset(X,Y,M,P)

        model = torch.load(modeldir)
        # Create dataloaders
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=10000,
                                                 shuffle=True, num_workers=0)

        torch.manual_seed(123)
        np.random.seed(123)

        attr_mean = 0
        for i, data in enumerate(dataloader):

            X_in = data['X']
            M_in = data['M']
            Y_in = data['Y']

            if model_type == 'conv-mixed':

                input_attr = (X_in,M_in)

                baseline = (torch.zeros(input_attr[0].shape).to(device),
                            torch.zeros(input_attr[1].shape).to(device))

                row_colors = ['blue'] * 191 + ['red'] * 83
            else:

                input_attr = (X_in,)

                baseline   = (torch.zeros(input_attr[0].shape).to(device))

                row_colors = ['blue'] * 191

            ig = IntegratedGradients(model)
            attributions, delta = ig.attribute(input_attr,
                                               baseline,
                                               target=0,
                                               return_convergence_delta=True)

            attr_new = np.concatenate([attr.cpu().detach().numpy() for attr in attributions],axis=1)

            if i == 0:
                # Plot attribution as cell x feature heatmap
                scaler = MinMaxScaler()
                attr = attr_new
                # attr = scaler.fit_transform(attr)
                df_attr = pd.DataFrame(attr).transpose() # n_feature x n_cell

                treatment_shorthand = cellmeta.Treatment.map(lambda x: x.split(" ")[0])
                treatment_shorthand = treatment_shorthand.map(lambda x: x.split(".")[0])
                treatment_shorthand = treatment_shorthand.map(lambda x: x.split("ng")[0])
                col_colors = series2colors(treatment_shorthand).values

                sns.clustermap(data=df_attr,
                               row_cluster=False,
                               col_cluster=True,
                               row_colors=row_colors,
                               col_colors=col_colors,
                               cmap='RdBu',center=0
                               # cmap=sns.color_palette("rocket_r", as_cmap=True)
                               )

                plt.savefig(os.path.join(outdir_model,f'attr-{attr_method}-{target}-notscaled.svg'))
                plt.close()

            ## Save
            attr_new = np.abs(attr_new).mean(0)
            # attr_new = np.concatenate([calculate_mean(attr) for attr in attributions])
            attr_mean += attr_new * len(X_in) / len(X) # Scale by # cells

        attr_targets.append(attr_mean)

    df_attr = pd.DataFrame(attr_targets,index=targets).transpose() # n_features x n_genes
    df_attr.to_csv(os.path.join(outdir_model,f'attr-{attr_method}.csv'))

    scaler = MinMaxScaler()
    df_attr = pd.DataFrame(scaler.fit_transform(df_attr),columns=targets)

    sns.clustermap(data=df_attr,
                   row_cluster=False,
                   col_cluster=False,
                   row_colors=row_colors,
                   cmap=sns.color_palette("rocket_r", as_cmap=True)
                   )
    plt.savefig(os.path.join(outdir_model,f'attr-{attr_method}.svg'))
    plt.close()
