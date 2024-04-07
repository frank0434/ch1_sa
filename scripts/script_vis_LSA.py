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
large_df['value'] = large_df['value'].astype(float)
colors = config.name_color_map
no_ofdays = len(large_df.day.unique())
# %%
DAPs = np.tile(np.arange(no_ofdays), config.LSA_sample_size * len(para))
large_df['DAP'] = DAPs[:len(large_df)]

# %%
def create_subplot(ax, output_df, output_var, param_name):
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(output_df['value'].min(), output_df['value'].max())
    sc = ax.scatter(output_df.index, output_df[output_var], c=output_df['value'], cmap=cmap, norm=norm)
    return sc

def create_figure(large_df, output_var, param_names):
    fig, axs = plt.subplots(5, 3, figsize=(9, 12), sharex=True, sharey=True)    
    fig.subplots_adjust(wspace=-.5, hspace= -0.5)
    axs = axs.flatten()

    for i, param_name in enumerate(param_names):
        output_df = large_df[large_df['key'] == param_name].sort_values('day')
        output_df.set_index('DAP', inplace=True)
        sc = create_subplot(axs[i], output_df, output_var, param_name)
        fig.colorbar(sc, ax=axs[i])

        if i >= 12:
            axs[i].set_xlabel('DAP')
        else:
            axs[i].set_xticklabels([])
        if i % 3 == 0:
            axs[i].set_ylabel(output_var)
        axs[i].set_title(f'{param_name}')

    plt.tight_layout()
    plt.savefig(f'{config.p_out_LSA}/{output_var}_timeseries.svg', bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f'{config.p_out_LSAsims}/{output_var}__timeseries.png', dpi = 300, bbox_inches='tight', pad_inches=0.1)
    plt.show()
# %%

for var in ['LAI', 'TWSO', 'DVS']:
    create_figure(large_df, var, config.params_of_interests)
