import pandas as pd
import os

def process_AUC_file(file):
    df = pd.read_csv(file)
    df['label'] = os.path.basename(file)
    df['country'] = "NLD" if "NL" in file else "IND"
    df_pawn = df.loc[:, ['variable','label', 'country', 'PAWN']].sort_values(by='PAWN', ascending=False)
    df_st = df.loc[:, ['variable', 'label', 'country', 'ST']].sort_values(by='ST', ascending=False)

    return df_pawn[(df_pawn['PAWN']>0) & (df_pawn['PAWN'].notna())], df_st[(df_st['ST']>0) & (df_st['ST'].notna())]