import pandas as pd

DATA_DIR = 'data/Alex/S1T4/20171030 S1 T4 Plate 1'

df_well = pd.read_csv(os.path.join(DATA_DIR,'WellInfo.csv'),index_col=0,header=None)

types = df_well.index.unique()

n_row = df_well.index.value_counts()[0]

df_well_org = pd.DataFrame({t: df_well[df_well.index == t].values.flatten() for t in types})
df_well_org['XY'] = df_well_org['XY'].astype('int')
df_well_org = df_well_org.sort_values(['XY'])
df_well_org = df_well_org.reset_index(drop=True)

df_well_org.to_csv(os.path.join(DATA_DIR,'WellInfoReformated.csv'))
