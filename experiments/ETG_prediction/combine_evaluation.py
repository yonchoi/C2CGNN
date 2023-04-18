import os,gc

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import argparse

import scanpy as sc

import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()

parser.add_argument('--outdir', type=str,
                    help='Directory where results will be saved')
parser.add_argument('--do_attribution', type=str2bool, default=False,
                    help='Top and bottom percentile to remove')

args = parser.parse_args()

outdir      = args.outdir
do_attribution = args.do_attribution

# outdir="out/exp4/3channel/scaled-128"
# outdir="out/exp4/3channel/scaled-128"
# outdir="out/3channel/exp5/invnorm"
# outdir="out/ram/replicate_subset/exp"
outdir='out/ram/replicate/exp5/all/param'
# outdir='out/ram/replicate/exp5/all/ERK'
# outdir='out/ram/replicate/exp4/all'

plotdir = os.path.join(outdir,'plot')
os.makedirs(plotdir,exist_ok=True)

settings_list = []

treatments = next(os.walk(outdir))[1]
treatments = [t for t in treatments if t != 'plot']

for treatment in treatments:
    inputs = os.listdir(os.path.join(outdir,treatment))
    for input in inputs:
        targets = os.listdir(os.path.join(outdir,treatment,input))
        for target in targets:
            models = os.listdir(os.path.join(outdir,treatment,input,target))
            for model in models:
                kfolds = os.listdir(os.path.join(outdir,treatment,input,target,model))
                for k in kfolds:
                    settings_list.append([treatment,input,target,model,k])

df_settings = pd.DataFrame(settings_list,
                           columns=['Treatment','Input','Target','Model','Kfold'])


dfs_result = []
for _,settings in df_settings.iterrows():
    result_dir  = os.path.join(outdir,*settings.values)
    # Read result csv file then concantenate settings
    df_result   = pd.read_csv(os.path.join(result_dir,'Eval.csv'),index_col=0).reset_index(drop=True)
    df_setting = pd.concat([settings]*len(df_result),axis=1).transpose().reset_index(drop=True)
    df_result = pd.concat([df_result,df_setting],axis=1)
    dfs_result.append(df_result)

dfs_result = pd.concat(dfs_result)
dfs_result['Name'] = dfs_result.Input+'-'+dfs_result.Model
## Save combined
dfs_result.to_csv(os.path.join(outdir,'combined_result.csv'))

#### Target subset
subset_target = None
subset_target = ['pca0','pca1','pca2']
if subset_target:
    # dfs_result = dfs_result[np.isin(dfs_result.Target,subset_target)]
    dfs_result = dfs_result[~np.isin(dfs_result.Target,subset_target)]

#### Model subset
subset_settings=None
# subset_settings = ['ERK-conv-mixed','ERK-conv','ERK-ridge','ERK-linear','ERK_P-ridge','ERK_P-linear']
# subset_settings = np.array(['ERK-conv-mixed','ERK-conv','ERK-flatten','ERK_P-flatten','ERK-ridge','ERK-linear','ERK_P-ridge','ERK_P-linear'])
# subset_settings = np.array(['ERK-conv-mixed','ERK-conv','ERK-flatten','ERK-ridge','ERK-linear','ERK_P-ridge','ERK_P-linear'])
# subset_settings = np.array(['ERK-conv', 'ERK-flatten', 'ERK-ridge', 'ERK_P-ridge'])
# subset_settings = np.array(['ERK-conv-mixed','ERK-conv','ERK_P-ridge','ERK_P-linear'])
# subset_settings = np.array(['ERK_-70-conv','ERK_+70-conv','ERK_all-conv'])
# subset_settings = np.array(['ERK-conv-mixed','ERK_P-ridge','ERK_P-linear'])

subset_settings = np.array([f'{input}-lasso' for input in dfs_result.Input.unique()])
subset_settings_temp = subset_settings[0]
subset_settings[0] = subset_settings[1]
subset_settings[1] = subset_settings_temp

# subset_settings = np.array(['Y_all-linear','Y_all-lasso','Y_all-ridge'])

if subset_settings is not None:
    # print(subsets-se
    subset_settings = subset_settings[np.isin(subset_settings,dfs_result.Name.unique())]
    dfs_result = dfs_result[np.isin(dfs_result.Name,subset_settings)]
    dfs_result['Name'] = pd.Categorical(dfs_result.Name,categories=subset_settings)

# sort_method = dfs_result.Name.sort_values().iloc[0] # method to sort the order of targets
sort_method = dfs_result.Name.sort_values().iloc[0] # method to sort the order of targets

for treatment in treatments:
    plotdir_treatment = os.path.join(plotdir,treatment)
    os.makedirs(plotdir_treatment,exist_ok=True)
    for data_type in ['train','test','final']:
        df_plot = dfs_result[dfs_result.Data==data_type]
        # Calculate mean of each target gene for sorting
        means = df_plot[df_plot.Name==sort_method].groupby("Target").mean()
        # For all correlation types
        for corr_type in ['pearson','spearman','R2']:
            df_plot['Target'] = pd.Categorical(df_plot['Target'],
                                   categories=means.sort_values(corr_type,ascending=False).index)
            # Plot as barplot
            g =sns.barplot(data = df_plot,
                           x='Target',
                           y=corr_type,
                           hue='Name')
            g.tick_params(axis='x', rotation=90)
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            plt.tight_layout()
            plt.savefig(os.path.join(plotdir_treatment,
                                     f'Barplot-{corr_type}-{data_type}.svg'))
            plt.close()
            # Plot as heatmap
            df_rect = df_plot.groupby(['Target','Name']).mean()[corr_type]
            df_rect = df_rect.reset_index().pivot(index='Target',columns='Name')[corr_type]
            sns.clustermap(data=df_rect,
                           row_cluster=False,
                           cmap='rocket_r',
                           annot=True,
                           fmt='.2f')
            plt.savefig(os.path.join(plotdir_treatment,
                                     f'Heatmap-{corr_type}-{data_type}.svg'))
            plt.close()

# ==============================================================================

if do_attribution:
#
    for input in df_settings.Input.unique():
#
        df_settings_ = df_settings[df_settings.Input == input]
        dfs_result = []
        for _,settings in df_settings_.iterrows():
            result_dir  = os.path.join(outdir,*settings.values)
            fa_dir = os.path.join(result_dir,'FFN','attr_target-IG.csv')
            if os.path.isfile(fa_dir):
                df_result = pd.read_csv(fa_dir,index_col=0).reset_index(drop=True)
                n_timepoints = df_result.shape[1]
                # Read result csv file then concantenate settings
                df_setting = pd.concat([settings]*len(df_result),axis=1).transpose().reset_index(drop=True)
                df_result = pd.concat([df_result,df_setting],axis=1)
                dfs_result.append(df_result)
#
        dfs_result = pd.concat(dfs_result)
        dfs_result['Name'] = dfs_result.Input+'-'+dfs_result.Model
        dfs_result.index = dfs_result.Target
        dfs_result.to_csv(os.path.join(outdir,f'FA_{input}_combined.csv'))
#
        for name in dfs_result.Name.unique():
            def norm(X,axis=0):
                MIN = X.min(axis)
                MAX = X.max(axis)
                return X.sub(MIN, 1-axis).div(MAX-MIN, 1-axis)
            X = dfs_result[dfs_result.Name == name].iloc[:,:n_timepoints]
            X = norm(X,1)
            g = sns.heatmap(X,cmap='rocket_r')
            plt.savefig(os.path.join(outdir,f'FA_{input}_heatmap_{name}.svg'))
            plt.close()


pd.concat([X.iloc[:,:70].sum(1),X.iloc[:,70:].sum(1)],axis=1)
