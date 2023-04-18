import os,gc

import numpy as np
import pandas as pd

filedir = ""

## Read excel files
path_x     = os.path.join(filedir,'df_x.xlsx') # x coordinate
path_y     = os.path.join(filedir,'df_y.xlsx') # y coordinate
path_size  = os.path.join(filedir,'df_narea.xlsx') # Cell size
path_w     = os.path.join(filedir,'df_tx.xlsx') # Well of Origin
path_erk   = os.path.join(filedir,'df_erk.xlsx') # ERK activity time-series
path_etgs  = os.path.join(filedir,'df_ETGs.xlsx') # Protein expressions

dfs_size = pd.read_excel(path_size, sheet_name='Sheet1', header=None)
dfs_erk = pd.read_excel(path_erk, sheet_name='Sheet1', header=None)
dfs_w = pd.read_excel(path_w, sheet_name='Sheet1', header=[0])
dfs_etg = pd.read_excel(path_etgs, sheet_name='Sheet1', header=[0])
