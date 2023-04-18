from src.util.util import corrcoef, r2coef
from src.util.util import series2colors

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import os
import numpy as np
import pandas as pd

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

import matplotlib.pyplot as plt
import seaborn as sns

# def evaluate_model(model,
#                    dataloaders,
#                    input_keys=['x'],
#                    model_type='torch',
#                    per_sample_error=False,
#                    outdir='',
#                    do_plot=False):
#     """
#     Input
#         dataloader: dictionary mapping data_type to torch dataloader
#     """
#     #
#     dfs = []
#     #
#     for train_type, dataloader in dataloaders.items():
#     # if True:
#         # train_type = 'final'
#         dataloader = dataloaders[train_type]
#         #
#         dataset_dict = dataloader.dataset[:]
#         #
#         targets = dataset_dict.pop("Y").cpu().detach().numpy()
#         #
#         inputs = {key: dataset_dict[key.upper()] for key in input_keys}
#
#         with torch.no_grad():
#             outputs = model(**inputs)
#             #
#         if torch.is_tensor(outputs):
#             outputs = outputs.cpu().detach().numpy()
#             #
#         y = outputs.reshape(len(targets),-1)
#         y = y.transpose()
#         x = targets.transpose()
#         #
#         df_corr = pd.DataFrame({'spearman': corrcoef(x,y,method='spearman'),
#                                 'pearson' : corrcoef(x,y,method='pearson'),
#                                 'R2'      : r2coef(x,y),
#                                 'Data': train_type
#                                 })
#         dfs.append(df_corr)
#         ### Plot
#         if do_plot:
#             os.makedirs(outdir,exist_ok=True)
#             x,y = x.transpose()[:100].reshape(-1),y.transpose()[:100].reshape(-1)
#             print(x.reshape(-1)[:10])
#             x0,x1 = np.min(x),np.max(x)
#             y0,y1 = np.min(y),np.max(y)
#             g = sns.scatterplot(x=x,
#                                 y=y)
#             lims = [x0-1, x1+1]
#             g.set_xlim(lims)
#             g.set_ylim(lims)
#             _ = g.plot(lims, lims, '-k')
#             plt.savefig(os.path.join(outdir,f'scatter-{train_type}.svg'))
#             plt.close()
#             print(os.path.join(outdir,f'scatter-{train_type}.svg'))
#
#     return pd.concat(dfs)

def evaluate_model(model,
                   dataloaders,
                   input_keys=['x'],
                   model_type='torch',
                   per_sample_error=False,
                   outdir='',
                   do_plot=False,
                   do_plot_time=False,
                   do_attribution=False,
                   do_plot_range=False,
                   n_channel=1,
                   device=None):
    """
    Input
        dataloader: dictionary mapping data_type to torch dataloader
    """
    #
    dfs = []
    #
    for train_type, dataloader in dataloaders.items():
    # if True:
        # train_type = 'final'
        dataloader = dataloaders[train_type]
        dataset_dict = dataloader.dataset[:]

        X = dataset_dict['X']
        Y = dataset_dict['Y']
        M = dataset_dict['M']
        #
        targets = dataset_dict.pop("Y").cpu().detach().numpy()
        #
        inputs = {key: dataset_dict[key.upper()] for key in input_keys}

        with torch.no_grad():
            outputs = model(**inputs)
            #
        if torch.is_tensor(outputs):
            outputs = outputs.cpu().detach().numpy()
            #
        y = outputs.reshape(len(targets),-1)
        y = y.transpose()
        x = targets.transpose()
        #
        df_corr = pd.DataFrame({'spearman': corrcoef(x,y,method='spearman'),
                                'pearson' : corrcoef(x,y,method='pearson'),
                                'R2'      : r2coef(x,y),
                                'MSE'     : np.square(x-y).mean(1),
                                'Data': train_type
                                })
        dfs.append(df_corr)

        #### Plot scatterplots of predicted and target (target must be 1D)
        if do_plot:
            os.makedirs(outdir,exist_ok=True)
            x_,y_ = x.transpose()[:100].reshape(-1),y.transpose()[:100].reshape(-1)
            x0,x1 = np.min(x_),np.max(x_)
            y0,y1 = np.min(y_),np.max(y_)
            g = sns.scatterplot(x=x_,
                                y=y_)
            lims = [x0-1, x1+1]
            g.set_xlim(lims)
            g.set_ylim(lims)
            _ = g.plot(lims, lims, '-k')
            plt.savefig(os.path.join(outdir,f'scatter-{train_type}.svg'))
            plt.close()
            print(os.path.join(outdir,f'scatter-{train_type}.svg'))

        #### Plot timepoints where each features
        if do_plot_time:
            targets,preds = x.transpose()[:100],y.transpose()[:100]
            channels = inputs['x'].cpu().detach().numpy()
            channels = channels.reshape(len(channels),-1,targets.shape[1]) # n_cell, n_channel, n_time
            plot_time_pred(targets,
                           preds,
                           channels,
                           filename=os.path.join(outdir,f'sampled_timeplot-{train_type}.svg'))
            # print(os.path.join(outdir,f'sampled_timeplot-{train_type}.svg'))

        #### Plot the predicted ETG per treatment
        if do_plot_range:

            targets,preds = x.transpose().reshape(-1),y.transpose().reshape(-1)

            if 'C' in dataset_dict.keys():
                # treatment = dataset_dict['C'][:,1]
                treatment = dataset_dict['C']
            else:
                treatment = 'None'

            df_targets = pd.DataFrame({'ETG' : targets,
                                       'Type': 'Observed',
                                       'Treatment': treatment})
            df_preds   = pd.DataFrame({'ETG' : preds,
                                       'Type': 'predicted',
                                       'Treatment': treatment})
            df_plot = pd.concat([df_targets,df_preds],axis=0,ignore_index=True)
            fig,ax = plt.subplots(figsize=[1 * len(df_plot.Treatment.unique()),10])
            g = sns.violinplot(data=df_plot, x='Treatment',y='ETG',hue='Type', split=True, ax=ax)
            g.set_xticklabels(g.get_xticklabels(),rotation=90)
            plt.tight_layout()
            # g = sns.violinplot(data=df_plot, x='Treatment',y='ETG',hue='Type')
            # g = sns.violinplot(data=df_plot, x='Treatment',y='Test')
            filename = os.path.join(outdir,f'ViolinPlot-PredictedETGRange-{train_type}.svg')
            plt.savefig(filename)
            plt.close()

        #### Run attribution
        if do_attribution:
            run_attribution(model,
                            model_type,
                            X=X,Y=Y,M=M,
                            plot_per_cell_heatmap=True,
                            plot_per_target_heatmap=True,
                            attr_method='IG',
                            outdir=outdir,
                            device=device,
                            n_channel=n_channel,
                            dataset_dict=dataset_dict
                            )

    return pd.concat(dfs)


def plot_time_pred(targets,preds,channels,filename):
    n_node_plot = min(len(targets),20)
    node_idx = np.linspace(0,len(targets)-1,n_node_plot).astype('int')
    fig,axes=plt.subplots(len(node_idx),figsize=(20,4*n_node_plot),
                          dpi=80, squeeze=False)
    axes = axes.flatten()
    for n,ax in zip(node_idx,axes):
        input  = channels[n] # n_time x n_feature
        label  = targets[n] # n_time x 1
        output = preds[n] # n_time x 1
        ax.plot(label,color='green',label='True')
        ax.plot(output,color='red',label='Pred')
        for color,input_f in zip(['springgreen','blue','skyblue'],
                                 input):
            ax.plot(input_f,color=color,label='Sensors',alpha=0.1)
        ax.legend()
#
    plt.savefig(filename)
    plt.close()


def run_attribution(model,model_type,X,Y,M,dataset_dict,
                    plot_per_cell_heatmap=False,
                    plot_per_target_heatmap=True,
                    attr_method='IG',
                    outdir="",
                    device=None,
                    n_channel=1,
                    max_sample=1000,
                    ):
    """
    Input
        model: torch model
        model_type: model type used
        X: Input matrix (n_cell, n_channel * n_time)
        Y: Target matrix (n_cell, n_time or n_ETG)
        M: Cell meta (n_cell, n_metafeatures)
        attr_method: which attr_method to use
        outdir: dir to save output figures
    Return

    """
    torch.manual_seed(123)
    np.random.seed(123)
    X,Y,M = X[:max_sample], Y[:max_sample], M[:max_sample]

    if model_type:

        input_attr = (X,M)

        baseline = (torch.zeros(input_attr[0].shape).to(device),
                    torch.zeros(input_attr[1].shape).to(device))

        col_colors = ['blue'] * X.shape[-1] + ['red'] * M.shape[-1]
    else:

        input_attr = (X,)

        baseline   = (torch.zeros(input_attr[0].shape).to(device))

        col_colors = ['blue'] * X.shape[-1]

    if 'C' in dataset_dict.keys():
        # row_categories = dataset_dict['C'][:,0]
        row_categories = dataset_dict['C']
        row_colors = series2colors(row_categories).values
    else:
        row_colors = None

    attr_targets = []
    for target in np.arange(Y.shape[-1]):
        ig = IntegratedGradients(model)
        with torch.no_grad():
            attributions, delta = ig.attribute(input_attr,
                                               baseline,
                                               target=int(target),
                                               return_convergence_delta=True)
        attr_new = np.concatenate([attr.cpu().detach().numpy() for attr in attributions],axis=1) # n_cell x n_feature
        attr_mean = np.abs(attr_new).mean(0) # n_feature
        attr_targets.append(attr_mean)
    #
        if plot_per_cell_heatmap and target==0:
            scaler = MinMaxScaler()
            attr = attr_new
            # attr = scaler.fit_transform(attr)
            df_attr = pd.DataFrame(attr) # n_cell x n_feature
    #
            # treatment_shorthand = cellmeta.Treatment.map(lambda x: x.split(" ")[0])
            # treatment_shorthand = treatment_shorthand.map(lambda x: x.split(".")[0])
            # treatment_shorthand = treatment_shorthand.map(lambda x: x.split("ng")[0])
            # row_colors = series2colors(treatment_shorthand).values
    #
            sns.clustermap(data=df_attr,
                           row_cluster=True,
                           col_cluster=False,
                           col_colors=col_colors,
                           row_colors=row_colors,
                           cmap='RdBu',center=0
                           # cmap=sns.color_palette("rocket_r", as_cmap=True)
                           )
    #
            plt.savefig(os.path.join(outdir,f'attr_cell-{attr_method}_target-{target}.svg'))
            plt.close()

    if plot_per_target_heatmap:

        df_attr = pd.DataFrame(attr_targets) # n_genes x n_features
        df_attr.to_csv(os.path.join(outdir,f'attr_target-{attr_method}.csv'))

        # scaler = MinMaxScaler()
        # df_attr = scaler.fit_transform(df_attr)
        df_attr = pd.DataFrame(df_attr)

        sns.clustermap(data=df_attr,
                       row_cluster=False,
                       col_cluster=False,
                       col_colors=col_colors,
                       cmap=sns.color_palette("rocket_r", as_cmap=True)
                       )

        plt.savefig(os.path.join(outdir,f'attr_target-{attr_method}.svg'))
        plt.close()

        if (n_channel > 1) and (len(df_attr) == 1) and (not model_type):

            df_attr = pd.DataFrame(np.array(attr_targets).reshape(n_channel,-1))
            # scaler = MinMaxScaler()
            # df_attr = scaler.fit_transform(df_attr)

            sns.clustermap(data=df_attr,
                           row_cluster=False,
                           col_cluster=False,
                           cmap=sns.color_palette("rocket_r", as_cmap=True)
                           )

            plt.savefig(os.path.join(outdir,f'attr_target-{attr_method}-melted.svg'))
            plt.close()
