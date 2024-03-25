#%%
import json
import config
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import math
# %%
para = config.params_of_interests
config.p_out_LSAsims
with open(f'{config.p_out_LSAsims}/hash_dict_final.json', 'r') as f:
    para_vals = json.load(f)
len(para_vals)
para_vals.values()
para.extend([] * config.LSA_sample_size)
keys = list(itertools.chain.from_iterable(itertools.repeat(x, config.LSA_sample_size) for x in para))

# %%

dfs = []  # List to store DataFrames

for i, key, value in zip(para_vals.keys(), keys, para_vals.values()):
    # print(i, value)
    with open(f'{config.p_out_LSAsims}/{i}_{key}.json', 'r') as f:
        results = json.load(f)
    df = pd.DataFrame(results)
    df['key'] = key
    df['value'] = value
    dfs.append(df)

# Concatenate all DataFrames
large_df = pd.concat(dfs)
# %%

colors = config.name_color_map
colors


# %%
DAPs = np.tile(np.arange(107), config.LSA_sample_size * len(para))
large_df['DAP'] = DAPs[:len(large_df)]
def plot_diff_for_key(df, key, ax, output_var = 'TWSO', color = 'blue'):
    filtered_df = df[df['key'] == key].sort_values('day').groupby('value')
    diff_df = filtered_df[output_var].apply(lambda x: x.diff()).reset_index()
    diff_df.plot(x='level_1', y=output_var, kind='line', ax=ax, label=key, color=color)
def plot_final_for_key(df, key, ax, output_var = 'TWSO', color = 'blue'):
    filtered_df = df[df['key'] == key]
    filtered_df.loc[:, 'value'] = filtered_df['value'].astype(float)
    filtered_df.plot(x='value', y=output_var, kind='scatter', ax=ax, label=key, color=color)
def plot_timeseries_for_key(df, key, ax, output_var = 'TWSO', c='value', cmap='viridis'):
    filtered_df = df[df['key'] == key]
    filtered_df.loc[:, 'value'] = filtered_df['value'].astype(float)
    filtered_df.plot(x='DAP', y=output_var, kind='scatter', ax=ax, label=key, c=c, cmap='viridis')
def plot_2nd_diff_for_key(df, key, ax, output_var = 'TWSO', color = 'blue'):
    filtered_df = df[df['key'] == key].sort_values('day').groupby('value')
    diff_df = filtered_df[output_var].apply(lambda x: x.diff()).reset_index()

# %%
plot_timeseries_for_key(large_df, 'TBASEM', plt.gca(), 'LAI')

# %% 
final_output = large_df[large_df['day'] == '2023-02-24'].drop(columns=['DAP','day','WWLOW','RD','SM'])
final_output.loc[:, 'value'] = final_output['value'].astype(float)

final_long = final_output.melt(id_vars = ['key', 'value'],  value_name='vals')
final_long

# Define a function that will sort the values in each group
def sort_group(group):
    return group.sort_values(by='value', ascending=True)

# Group the DataFrame by 'group', apply the function, and reset the index
sorted_df = final_long.groupby(['key', 'variable']).apply(sort_group).reset_index(drop=True)

# sorted_df.plot(x='value', y='vals', kind='scatter', c=sorted_df['key'].map(colors), cmap='viridis')

sorted_df.groupby(['key', 'variable'])
# Define a function that will calculate the first derivative of 'vals' with respect to 'value' in each group
def calculate_second_derivative(group):
    group['first_derivative'] = np.gradient(group['vals'], group['value'])
    group['second_derivative'] = np.gradient(group['first_derivative'], group['value'])
    return group

# Group the DataFrame by 'group', apply the function, and reset the index
df = sorted_df.groupby(['key', 'variable']).apply(calculate_second_derivative).reset_index(drop=True)

print(df)
# %%
df[~(df['second_derivative'] == 0)]['key'].unique()

# print(sorted_df)
# %%
import seaborn as sns

# Assuming your long-format DataFrame is named 'df'
# Define the variables for x, y, and facets
g = sns.FacetGrid(df, col='variable', row='key', sharex=False, sharey=False)
g.map(sns.scatterplot, 'value', 'first_derivative', color='black')
# g.map(sns.scatterplot, 'value', 'vals', color='black')
# %% 

print(df[df['second_derivative'] > 0.001])
df[df['second_derivative'] > 0.001]['key'].unique()
# %%
# make a matrix 
pre_m = df.loc[:,['key', 'variable', 'second_derivative']].drop_duplicates()
pre_m = pre_m.groupby(['key', 'variable']).apply(lambda second_derivative: np.mean(np.abs(second_derivative))).reset_index()
pre_m['effects'] = np.where(abs(pre_m[0]) > 0.0001, 1, 0)
pre_m = pre_m.pivot(index='key', columns='variable', values='effects')
pre_m = pre_m.fillna(0).sort_values(by=['DVS', 'LAI', 'TWSO'], ascending=False)


# Create a function to generate the annotations
def create_annotations(df):
    annotations = df.copy()
    annotations[df > 0] = '+'
    annotations[df == 0] = '-'
    return annotations

# Create heatmap
sns.heatmap(pre_m, annot=create_annotations(pre_m), fmt='', cmap='viridis', alpha=0, cbar=False)
plt.ylabel('Input Parameters', fontsize = 16)
plt.xlabel('')

plt.tick_params(axis='x', bottom=False, top=True, labelbottom=False, labeltop=True)
plt.savefig(f'{config.p_out_LSAsims}/LSAheatmap_plusM.png', dpi = 300, bbox_inches='tight', pad_inches=0.1)

plt.show()
# %%
plt.figure(figsize=(10, 8))
ax = sns.heatmap(pre_m, annot=True,  fmt='', cbar=False, cmap='RdYlBu')
for text, color in zip(ax.texts, (color.get_array() for color in ax.collections)):
    text.set_text('+' if '1' in text.get_text() else '-')
    text.set_color('black' if '1' in text.get_text() else 'white')

plt.ylabel('Input Parameters', fontsize = 16)
plt.xlabel('Output Variables', fontsize = 16)

plt.tick_params(axis='x', bottom=False, top=True, labelbottom=False, labeltop=True)

plt.savefig(f'{config.p_out_LSAsims}/LSAheatmap.png', dpi = 300, bbox_inches='tight', pad_inches=0.1)
plt.show()
# %%

# List of all your parameters
#%%

def create_subplots(df, para, variable, colors):
    n = len(para)
    rows = math.ceil(n / 3)
    fig_height = rows * 5  # Adjust this value to change the height of each subplot

    fig, axes = plt.subplots(rows, 3, figsize=(15, fig_height))

    # Use the function for each parameter
    for ax, param in zip(axes.flatten(), para):
        plot_diff_for_key(df, param, ax, variable, colors[param])
        ax.set_xlabel('')

    # If the number of subplots is odd, remove the last one
    if n % 3 != 0:
        fig.delaxes(axes.flatten()[-1])
    plt.legend()
    fig.text(0.5, 0, 'DAP', ha='center', va='center', fontsize=16)
    fig.text(0, 0.5, f'{variable}', va='center', rotation='vertical', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{config.p_out_LSAsims}/{variable}_diff.png', dpi = 300, bbox_inches='tight', pad_inches=0.1)

    plt.show()

def create_subplots_final(df, para, variable, colors, suffix = 'allparams'):
    n = len(para)
    rows = math.ceil(n / 2)
    fig_height = rows * 3  # Adjust this value to change the height of each subplot

    fig, axes = plt.subplots(rows, 2, figsize=(9, fig_height), sharey=True)

    # Use the function for each parameter
    for ax, param in zip(axes.flatten(), para):
        plot_final_for_key(df, param, ax, variable, colors[param])
        ax.set_xlabel('')
        ax.set_ylabel('')
    if n % 3 != 0:
        fig.delaxes(axes.flatten()[-1])
    plt.legend()
    fig.text(0.5, 0, 'value', ha='center', va='center', fontsize=16)
    fig.text(0, 0.5, f'{variable}', va='center', rotation='vertical', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{config.p_out_LSA}/{variable}_final_{suffix}.svg', bbox_inches='tight', pad_inches=0.1)

    plt.savefig(f'{config.p_out_LSAsims}/{variable}_final_{suffix}.png', dpi = 300, bbox_inches='tight', pad_inches=0.1)

    plt.show()
def create_subplots_time(df, para, variable, colors):
    n = len(para)
    rows = math.ceil(n / 3)

    fig, ax = plt.subplots(rows, 3, figsize=(15, 15))

    # Use the function for each parameter
    for ax, param in zip(ax.flatten(), para):
        plot_timeseries_for_key(df, param, ax, variable, colors[param])
        ax.set_xlabel('')
        ax.set_ylabel('')

    plt.legend()
    fig.text(0.5, 0, 'DAP', ha='center', va='center', fontsize=16)
    fig.text(0, 0.5, f'{variable}', va='center', rotation='vertical', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{config.p_out_LSAsims}/{variable}_timeseries.png', dpi = 300, bbox_inches='tight', pad_inches=0.1)

    plt.show()
# %% 
# second round of visualization 
para_LAI = ['TDWI']
final_lai_df  = large_df[(large_df['key'].isin(para_LAI))&(large_df['day'] == '2023-02-24')]

create_subplots_final(final_lai_df, para_LAI, 'LAI', colors, suffix='selected')
# %%
# DVS
para_DVS = ['TSUM1', 'SPAN', 'TBASEM', 'TSUMEM', 'TEFFMX', 'TDWI','te', 't1_pheno']
final_DVS_df  = large_df[(large_df['key'].isin(para_DVS))&(large_df['day'] == '2023-02-24')]

create_subplots_final(final_DVS_df, para_DVS, 'DVS', colors, suffix='selected')
# %%
para_TWSO = ['TSUM1', 'SPAN', 'TBASEM', 'TSUMEM', 'TEFFMX', 'TDWI','te', 't1_pheno']
final_TWSO_df  = large_df[(large_df['key'].isin(para_TWSO))&(large_df['day'] == '2023-02-24')]

create_subplots_final(final_TWSO_df, para_TWSO, 'TWSO', colors, suffix='selected')

# %%
#  ['2022-12-19', '2023-01-16']
para_LAI = ['SPAN']
final_lai_df  = large_df[(large_df['day'] == '2022-12-19')]

create_subplots_final(final_lai_df, config.params_of_interests, 'LAI', colors)

final_lai_df  = large_df[(large_df['day'] == '2023-01-16')]

create_subplots_final(final_lai_df, config.params_of_interests, 'LAI', colors)

# %%

# first round visualization 
# Call the function for each variable
for i in config.cols_of_interests:
    create_subplots(large_df, para, i, colors)

# create_subplots(large_df, para, 'TWSO', colors)
# create_subplots(large_df, para, 'LAI', colors)
# create_subplots(large_df, para, 'DVS', colors)
# %% 
# only focus on the final output 
final_output = large_df[large_df['day'] == '2023-02-24']    
# final_output[final_output['key'] == 'TSUM1'].plot(x='value', y='TWSO', kind='scatter')
# final_output[final_output['key'] == 'SPAN'].plot(x='value', y='TWSO', kind='scatter')
# final_output[final_output['key'] == 'te'].plot(x='value', y='TWSO', kind='scatter')
# final_output[final_output['key'] == 'SPAN'].plot(x='value', y='TWSO', kind='scatter')
for i in config.cols_of_interests:
    create_subplots_final(final_output, para, i, colors)

# %%
for i in config.cols_of_interests:
    create_subplots_time(large_df, para, i, colors)
        
# %%
TSUM1 = large_df[large_df['key'] == 'TSUM1'].sort_values('day').groupby('value')
TWSO_diff = TSUM1['TWSO'].apply(lambda x: x.diff())
TWSO_diff = TWSO_diff.reset_index()
TWSO_diff[TWSO_diff['value'] == "150.0"].plot(x='level_1', y='TWSO', kind='line')
TWSO_diff.plot(x='level_1', y='TWSO', kind='line')


SPAN = large_df[large_df['key'] == 'SPAN'].sort_values('day').groupby('value')
TWSO_diff = SPAN['TWSO'].apply(lambda x: x.diff())
TWSO_diff = TWSO_diff.reset_index()
# TWSO_diff[TWSO_diff['value'] == "150.0"].plot(x='level_1', y='TWSO', kind='line')
TWSO_diff.plot(x='level_1', y='TWSO', kind='line')
te = large_df[large_df['key'] == 'te'].sort_values('day').groupby('value')
TWSO_diff = te['TWSO'].apply(lambda x: x.diff())
TWSO_diff = TWSO_diff.reset_index()
# TWSO_diff[TWSO_diff['value'] == "150.0"].plot(x='level_1', y='TWSO', kind='line')
TWSO_diff.plot(x='level_1', y='TWSO', kind='line')