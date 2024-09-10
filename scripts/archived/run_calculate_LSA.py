# The script was used to calculate the ratio of the maximum 
# and minimum values of LAI, TWSO, and DVS for each parameter value.
# This was not used in the final analysis, but was used to explore
# %%
import json
import config
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

# Load parameter values
with open(f'{config.p_out_LSAsims}/hash_dict_final.json', 'r') as f:
    para_vals = json.load(f)

# Prepare keys
keys = list(itertools.chain.from_iterable(itertools.repeat(x, config.LSA_sample_size) for x in config.params_of_interests))

# Load data and create a large DataFrame
dfs = []
for i, key in zip(para_vals.keys(), keys):
    with open(f'{config.p_out_LSAsims}/{i}_{key}.json', 'r') as f:
        results = json.load(f)
    df = pd.DataFrame(results)
    df['key'] = key
    df['value'] = para_vals[i]
    dfs.append(df)
large_df = pd.concat(dfs)

# Additional DataFrame setup
large_df['value'] = large_df['value'].astype(float)
large_df['DAP'] = np.tile(np.arange(len(large_df.day.unique())), config.LSA_sample_size * len(config.params_of_interests))[:len(large_df)]
large_df.set_index('DAP', inplace=True)
# %%
# Function to calculate ratio
def calculate_ratio(df, output_var, param_name):
    grouped_df = df.groupby('DAP')
    output_max = grouped_df[output_var].max()
    output_min = grouped_df[output_var].min()
    df = pd.DataFrame({f'{output_var}_MAX': output_max, f'{output_var}_MIN': output_min})
    df[f'{param_name}_ratio'] = (df[f'{output_var}_MAX'] - df[f'{output_var}_MIN']) / df[f'{output_var}_MAX']
    return df

# Function to plot ratio
def plot_ratio(df, output_var, param_names):
    fig, axs = plt.subplots(5, 3, figsize=(9, 12), sharex=True, sharey=True)    
    ax = axs.flatten()

    for i, param_name in enumerate(param_names):
        df_filtered = df[df['key'] == param_name]
        ratio_df = calculate_ratio(df_filtered, output_var, param_name)
        ratio_df.plot(y=f'{param_name}_ratio', ax=ax[i], color='black', linewidth=2,
                      legend=False, title=param_name, xlabel='DAP', ylabel='Ratio')
        # ax[i].set_title(param_name)

        ax[i].set_xlabel('DAP')

    plt.savefig(f'{config.p_out_LSA}/{param_name}_{output_var}_ratio.png', dpi = 300, bbox_inches='tight', pad_inches=0.1)
    plt.show()

# Run the plot_ratio function for each variable
for var in ['LAI', 'TWSO', 'DVS']:
    plot_ratio(large_df, var, config.params_of_interests)