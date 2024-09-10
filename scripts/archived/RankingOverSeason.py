# The script is used to rank the variables based on the AUC values
# The script will read the AUC values from the output files and 
# rank the variables based on the AUC values
# only for exploratory purpose
# %%
import pandas as pd
import os
import config


def standardize_rank(df, column):
    group_sum = df.groupby(['label', 'country'])[column].transform('sum')
    df['group_sum'] = group_sum
    df['standarded_rank'] = df[column] / df['group_sum']
    return df[df[column]>0]

# Main execution
base_path = "C:/Users/liu283/GitRepos/ch1_SA/"
col_variable = "TWSO" 
file = os.path.join(base_path, f"output_NL_AUC_{col_variable}.csv") if config.run_NL_conditions else os.path.join(base_path, f"output_AUC_{col_variable}.csv")
