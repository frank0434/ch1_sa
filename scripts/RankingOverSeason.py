# %%
import pandas as pd
import os

# List of file paths
files = [
    "C:/Users/liu283/GitRepos/ch1_SA/output_AUC_DVS.csv",
    "C:/Users/liu283/GitRepos/ch1_SA/output_AUC_LAI.csv",
    "C:/Users/liu283/GitRepos/ch1_SA/output_AUC_TWSO.csv",
    "C:/Users/liu283/GitRepos/ch1_SA/output_NL_AUC_DVS.csv",
    "C:/Users/liu283/GitRepos/ch1_SA/output_NL_AUC_LAI.csv",
    "C:/Users/liu283/GitRepos/ch1_SA/output_NL_AUC_TWSO.csv"
]
# %%
df = pd.read_csv(files[5])
df['label'] = os.path.basename(files[5])
df['country'] = "NLD" if "NL" in files[5] else "IND"

df = df.iloc[0:5, :]
df_pawn = df.loc[:, ['variable','label', 'country', 'PAWN']].sort_values(by='PAWN', ascending=False)
df_st = df.loc[:, ['variable', 'label', 'country', 'ST']].sort_values(by='ST', ascending=False)
df_st
# %%
# Read each file, select top 5, label and merge
dfs_pawn = []
dfs_st = []
for file in files:

    df = pd.read_csv(file)
    df['label'] = os.path.basename(file)
    df['country'] = "NLD" if "NL" in file else "IND"
    # df = df.iloc[0:5, :]
    df_pawn = df.loc[:, ['variable','label', 'country', 'PAWN']].sort_values(by='PAWN', ascending=False)
    # df_pawn['rankid'] = range(1, 6)
    df_st = df.loc[:, ['variable', 'label', 'country', 'ST']].sort_values(by='ST', ascending=False)
    # df_st['rankid'] = range(1, 6)
    dfs_pawn.append(df_pawn)
    dfs_st.append(df_st)
# %%
df_pawn_long = pd.concat(dfs_pawn, axis=0)
# Group 'pawn_long' by 'label' and 'country' and compute sum
group_sum = df_pawn_long.groupby(['label', 'country'])['PAWN'].transform('sum')

# Add 'group_sum' as a new column to 'pawn_long'
df_pawn_long['group_sum'] = group_sum

df_st_long = pd.concat(dfs_st, axis=0)
df_st_long['group_sum'] = df_st_long.groupby(['label', 'country'])['ST'].transform('sum')
df_pawn_long['standarded_rank'] = df_pawn_long['PAWN'] / df_pawn_long['group_sum']
df_st_long['standarded_rank'] = df_st_long['ST'] / df_st_long['group_sum']
df_pawn_long[df_pawn_long['label'] == "output_AUC_TWSO.csv"]
# df_pawn_long[df_pawn_long['label'] == "output_NL_AUC_TWSO.csv"]
# %%
df_st_long[df_st_long['label'] == "output_AUC_TWSO.csv"]
# df_st_long[df_st_long['label'] == "output_NL_AUC_TWSO.csv"]

#%%
# Merge all dataframes
merged_df = pd.concat(dfs_pawn + dfs_st, axis=0)

long_df = merged_df.melt(id_vars=['variable', 'label', 'country','rankid'], var_name='method', value_name='value')
# long_df.drop(columns=['value'], inplace=True)
# Set multiple index for 'long_df'
# long_df = long_df.set_index(['label', 'method', 'country'])
# Set 'rankid' and 'variable' as index
# %%
long_df[long_df['country']=="IND"]
# .pivot(index='rankid', columns=[ 'label','method', 'country'], values='value')
#%%
# Merge all dataframes
#
# Create multi-level column index
top_level = ['LAI', 'TWSO'] * 2
second_level = ['NLD'] * 3 + ['IND'] * 3
merged_df.columns = pd.MultiIndex.from_tuples(zip(top_level, second_level))

print(merged_df)