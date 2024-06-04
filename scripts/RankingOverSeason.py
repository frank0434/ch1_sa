# %%

import pandas as pd
import os
import config

def process_AUC_file(file):
    df = pd.read_csv(file)
    df['label'] = os.path.basename(file)
    df['country'] = "NLD" if "NL" in file else "IND"
    df_pawn = df.loc[:, ['variable','label', 'country', 'PAWN']].sort_values(by='PAWN', ascending=False)
    df_st = df.loc[:, ['variable', 'label', 'country', 'ST']].sort_values(by='ST', ascending=False)

    return df_pawn[(df_pawn['PAWN']>0) & (df_pawn['PAWN'].notna())], df_st[(df_st['ST']>0) & (df_st['ST'].notna())]

def standardize_rank(df, column):
    group_sum = df.groupby(['label', 'country'])[column].transform('sum')
    df['group_sum'] = group_sum
    df['standarded_rank'] = df[column] / df['group_sum']
    return df[df[column]>0]

# Main execution
base_path = "C:/Users/liu283/GitRepos/ch1_SA/"
col_variable = "TWSO" 
file = os.path.join(base_path, f"output_NL_AUC_{col_variable}.csv") if config.run_NL_conditions else os.path.join(base_path, f"output_AUC_{col_variable}.csv")

# Test the function below
# df_pawn, df_st = process_AUC_file(file)
# df_pawn = standardize_rank(df_pawn, 'PAWN')
# df_st = standardize_rank(df_st, 'ST')

# df_st, df_pawn
# # %% 
# col_variable = "TWSO" 
# file = os.path.join(base_path, f"output_NL_AUC_{col_variable}.csv") 

# df_pawn, df_st = process_AUC_file(file)
# df_pawn = standardize_rank(df_pawn, 'PAWN')
# df_st = standardize_rank(df_st, 'ST')
# df_pawn, df_st
# #%%
# col_variable = "LAI" 
# file = os.path.join(base_path, f"output_NL_AUC_{col_variable}.csv") if config.run_NL_conditions else os.path.join(base_path, f"output_AUC_{col_variable}.csv")

# df_pawn, df_st = process_AUC_file(file)
# df_pawn = standardize_rank(df_pawn, 'PAWN')
# df_st = standardize_rank(df_st, 'ST')
# df_pawn, df_st

# col_variable = "LAI" 
# file = os.path.join(base_path, f"output_NL_AUC_{col_variable}.csv") 

# df_pawn, df_st = process_AUC_file(file)
# df_pawn = standardize_rank(df_pawn, 'PAWN')
# df_st = standardize_rank(df_st, 'ST')
# df_pawn, df_st