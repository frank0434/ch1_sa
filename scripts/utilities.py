import pandas as pd
import os
import pickle
import config

def process_AUC_file(file):
    df = pd.read_csv(file)
    df['label'] = os.path.basename(file)
    df['country'] = "NLD" if "NL" in file else "IND"
    df_pawn = df.loc[:, ['variable','label', 'country', 'PAWN']].sort_values(by='PAWN', ascending=False)
    df_st = df.loc[:, ['variable', 'label', 'country', 'ST']].sort_values(by='ST', ascending=False)

    return df_pawn[(df_pawn['PAWN']>0) & (df_pawn['PAWN'].notna())], df_st[(df_st['ST']>0) & (df_st['ST'].notna())]


# figure 3 
def load_data(day, output_var='TWSO', method='Saltelli'):
    # Load data
    with open(f'{config.p_out_daySi}/{method}_{day}_{output_var}.pkl', 'rb') as f:
        Si = pickle.load(f)
    return Si
# %%
def to_df(self):
    """Convert dict structure into Pandas DataFrame."""
    keys_to_include = ['S1', 'S1_conf', 'ST', 'ST_conf']
    return pd.DataFrame(
        {k: v for k, v in self.items() if k in keys_to_include}, index = config.params_of_interests
    )