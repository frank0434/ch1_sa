# graph for GEM symposium poster
# session 
# %%
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from datetime import datetime
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
import config
colors = ['#0072B2', '#E69F00', '#009E73', '#D3D3D3', '#696969']
config.set_variables(config.GSA_sample_size)
# %%  Figure 1 DVS, LAI, TWSO in both NL and IND - manual run with modifing config.py
import utilities as ros
from run_vis_GSA import process_dvs_files, process_files, create_dataframe_from_dict, load_PAWN, normalize_sensitivity, config
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Load dummy sensitivity indices
with open('DummySi_results.pkl', 'rb') as f:
    Dummy_si = pickle.load(f)

# Process DVS files
emergence_date, tuber_initiation = process_dvs_files(base_path="C:/Users/liu283/GitRepos/ch1_SA/output/daysims_32768/")
emergence_date_NL, tuber_initiation_NL = process_dvs_files(base_path="C:/Users/liu283/GitRepos/ch1_SA/output_NL/daysims_32768/", 
                                                           planting_date="2021-04-22")
# Define base path and file
base_path = "C:/Users/liu283/GitRepos/ch1_SA/"
col = "TWSO"
file = os.path.join(base_path, f"output_NL_AUC_{col}.csv") if config.run_NL_conditions else os.path.join(base_path, f"output_AUC_{col}.csv")
file_nl = os.path.join(base_path, f"output_NL_AUC_{col}.csv")
# Process AUC file
df_pawn_ros, df_st_ros = ros.process_AUC_file(file)
df_ros = pd.merge(df_st_ros, df_pawn_ros, how='inner', on=['variable', 'label', 'country'])
df_pawn_ros_NL, df_st_ros_NL = ros.process_AUC_file(file_nl)
df_ros_NL = pd.merge(df_st_ros_NL, df_pawn_ros_NL, how='inner', on=['variable', 'label', 'country'])
# Process sensitivity files
df_sensitivity_S1, df_sensitivity_ST = process_files(col)
df_pawn_long = create_dataframe_from_dict(load_PAWN(col))
df_pawn_long = df_pawn_long[df_pawn_long['median'] > Dummy_si[1][1]]
df_pawn_median = df_pawn_long.pivot_table(index='DAP', columns='names', values='median').reset_index()

df_sensitivity_S1_NL, df_sensitivity_ST_NL = process_files(col, period=161, path="C:/Users/liu283/GitRepos/ch1_SA/output_NL/daySi_32768/")
df_pawn_long_NL = create_dataframe_from_dict(load_PAWN(col, period=161,path="C:/Users/liu283/GitRepos/ch1_SA/output_NL/daySi_32768/"))
df_pawn_long_NL = df_pawn_long_NL[df_pawn_long_NL['median'] > Dummy_si[1][1]]
df_pawn_median_NL = df_pawn_long_NL.pivot_table(index='DAP', columns='names', values='median').reset_index()

# Align dataframes
df_pawn_median.set_index('DAP', inplace=True)
df_pawn_median_NL.set_index('DAP', inplace=True)
df_sensitivity_ST, df_pawn_median = df_sensitivity_ST.align(df_pawn_median, axis=0, join='left')
df_sensitivity_ST_NL, df_pawn_median_NL = df_sensitivity_ST_NL.align(df_pawn_median_NL, axis=0, join='left')
# Print sensitivity indices
print(f"Print 1st and total order Sobol indices for {col}.")

# Adjust data based on column type
if col in ['LAI', 'TWSO']:
    start_date = emergence_date[0] if col == 'LAI' else tuber_initiation[0]
    df_sensitivity_ST = df_sensitivity_ST.iloc[start_date:]
    df_pawn = df_pawn_median.iloc[start_date:]
    start_date_NL = emergence_date_NL[0] if col == 'LAI' else tuber_initiation_NL[0]
    df_sensitivity_ST_NL = df_sensitivity_ST_NL.iloc[start_date_NL:]
    df_pawn_NL = df_pawn_median_NL.iloc[start_date_NL:]

else:
    df_pawn = df_pawn_median
    df_pawn_NL = df_pawn_median_NL
# Normalize sensitivity data
df2 = normalize_sensitivity(df_sensitivity_ST)
df3 = normalize_sensitivity(df_pawn)
df2_NL = normalize_sensitivity(df_sensitivity_ST_NL)
df3_NL = normalize_sensitivity(df_pawn_NL)

# Define colors
colors2 = [config.name_color_map.get(col, 'black') for col in df2.columns]
colors3 = [config.name_color_map.get(col, 'black') for col in df3.columns]
colors2_NL = [config.name_color_map.get(col, 'black') for col in df2_NL.columns]
colors3_NL = [config.name_color_map.get(col, 'black') for col in df3_NL.columns]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7, 5), sharey=True)

# Plot data for NL
df2_NL.plot.area(ax=axes[0, 0], stacked=True, color=colors2_NL, legend=False)
df3_NL.plot.area(ax=axes[1, 0], stacked=True, color=colors3_NL, legend=False)

# Plot data for IND
df2.plot.area(ax=axes[0, 1], stacked=True, color=colors2, legend=False)
df3.plot.area(ax=axes[1, 1], stacked=True, color=colors3, legend=False)

plt.ylim(0, 1.05)
fig.text(0.02, 0.5, 'Proportion of Sensitivity indices', va='center', rotation='vertical', fontsize=config.subplot_fs)
fig.text(0.4, 0.02, 'Day After Planting', va = 'center', fontsize =  config.subplot_fs)
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
# plt.gca().invert_yaxis()
# Add subplot labels and fill between dates
labels = ['a)', 'c)', 'b)', 'd)']

for i, ax in enumerate(axes.flatten()):
    label = labels[i]
    # ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    # ax.set_yticklabels(['100%', '75%', '50%', '25%', '0%'])
    ax.text(0.01, config.subplotlab_y, label, transform=ax.transAxes, size=config.subplot_fs, weight='bold')
    ax.set_xlabel('')
    if i < 2:
        ax.set_xticks([])
        ax.set_title('')  
    if i % 2 == 0:
        ax.set_xlim([0, 161])  # Adjust the limits as needed
    if i % 2 == 1:
        ax.set_xlim([0, 106])
plt.subplots_adjust(hspace=0.05, wspace=0.05)  # Adjust hspace and wspace as needed

# Save and show plot
scenario = 'NL_' if config.run_NL_conditions else ''
# plt.tight_layout()
plt.savefig(f'{config.p_out}/{scenario}Sobol_Salteli_PAWN_{col}_poster.svg', bbox_inches='tight')
plt.savefig(f'{config.p_out}/{scenario}Sobol_Salteli_PAWN_{col}_poster.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()


# %% bar plot
# Assuming `ros` is an object with a method `process_AUC_file`
# Create a 2x2 grid of subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

# Plot data for IND
df_ros.loc[:,'ST'].plot.barh(ax=axes[0, 0], x='label', y='AUC_x', legend=False, color='blue', title='IND - ST')
df_ros.loc[:,'PAWN'] .plot.barh(ax=axes[1, 0], x='label', y='AUC_y', legend=False, color='green', title='IND - PAWN')

# Plot data for NL
df_ros_NL.loc
# Add subplot labels
labels = ['a)', 'b)', 'c)', 'd)']
for i, ax in enumerate(axes.flatten()):
    label = labels[i]
    ax.text(-0.1, 1.05, label, transform=ax.transAxes, size=14, weight='bold')

# Adjust the spacing between subplots
plt.subplots_adjust(hspace=0.4, wspace=0.4)


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
large_df.set_index('DAP', inplace=True)
# %%  make plot for the LSA results twso 

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
with open(f'{config.p}/output/LSA/sims_100/hash_dict_final.json', 'r') as f:
    para_vals = json.load(f)
para.extend([] * config.LSA_sample_size)

keys = list(itertools.chain.from_iterable(itertools.repeat(x, config.LSA_sample_size) for x in para))
with open(f'{config.p}/output/LSA_NL/sims_NL_100/hash_dict_final.json', 'r') as f:
    para_vals_nl = json.load(f)

# %%

dfs = []  # List to store DataFrames

for i, key, value in zip(para_vals.keys(), keys, para_vals.values()):
    # print(i, value)
    with open(f'{config.p}/output/LSA/sims_100/{i}_{key}.json', 'r') as f:
        results = json.load(f)
    df = pd.DataFrame(results)
    df['key'] = key
    df['value'] = value
    df['value'] = df['value'].astype(float)
    dfs.append(df)
df_NL = []
for i, key, value in zip(para_vals_nl.keys(), keys, para_vals_nl.values()):
    # print(i, value)
    with open(f'{config.p}/output/LSA_NL/sims_NL_100/{i}_{key}.json', 'r') as f:
        results = json.load(f)
    df = pd.DataFrame(results)
    df['key'] = key
    df['value'] = value
    df['value'] = df['value'].astype(float)
    df_NL.append(df)
# Concatenate all DataFrames
large_df = pd.concat(dfs)
colors = config.name_color_map
no_ofdays = len(large_df.day.unique())
large_df_NL = pd.concat(df_NL)
no_ofdays_NL = len(large_df_NL.day.unique())
# %%
import string
key_fig6 = ['t1_pheno', 'SPAN', 'te']
df_fig6_IND = large_df[large_df['key'].isin(key_fig6)].loc[:,['day','TWSO','DVS','key','value']]
df_fig6_IND['DAP'] = np.tile(np.arange(no_ofdays), config.LSA_sample_size * len(key_fig6))
df_fig6_NL = large_df_NL[large_df_NL['key'].isin(key_fig6)].loc[:,['day','TWSO','DVS','key','value']]
df_fig6_NL['DAP'] = np.tile(np.arange(no_ofdays_NL), config.LSA_sample_size * len(key_fig6))
df_fig6_IND['country'] = 'IND'
df_fig6_NL['country'] = 'NL'
df_fig6 = pd.concat([df_fig6_IND, df_fig6_NL])
df_fig6.set_index('DAP', inplace=True)
countries = ['NL', 'IND']
df_fig6['key'] = pd.Categorical(df_fig6['key'], key_fig6)
df_fig6['country'] = pd.Categorical(df_fig6['country'], categories=countries, ordered=True)
df_fig6['TWSO'] = df_fig6['TWSO'] / 1000  # Convert to t ha-1
# %%
# Figure 6
# Import necessary libraries
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
import string  # Ensure string is imported for subplot labels

# Assuming `key_fig6`, `countries`, `df_fig6`, and `config` are defined in the user's context.
plt.rcParams['font.size'] = config.subplot_fs
# Create a figure with subplots on the left and colorbars on the right
fig = plt.figure(figsize=(5,7))
gs = gridspec.GridSpec(3, 3, width_ratios=[4, 4, 0.5], wspace=0.2, hspace=0.2)  # Adjust the width ratio for colorbars
labels = string.ascii_lowercase[:len(key_fig6) * len(countries)]
pointsize = 1
subplotlab_x = config.subplotlab_x
subplotlab_y = config.subplotlab_y

# Flatten the array of axes for easy iteration
axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(2)]
colorbar_axes = [fig.add_subplot(gs[i, 2]) for i in range(3)]  # For colorbars

# Iterate over parameters and countries to create subplots
for i, param in enumerate(key_fig6):
    for j, country in enumerate(countries):
        ax = axes[i * len(countries) + j]
        data = df_fig6[(df_fig6['key'] == param) & (df_fig6['country'] == country)]
        cmap = plt.get_cmap('viridis')
        norm = plt.Normalize(data['value'].min(), data['value'].max())
        sc = ax.scatter(x=data.index, y=data['TWSO'], c=data['value'], cmap=cmap, norm=norm, s=pointsize)
        
        emergence_date = df_fig6[(df_fig6['DVS'] == 0) & (df_fig6['country'] == country)]['DVS'].drop_duplicates()
        tuberinitiation_date = df_fig6[(df_fig6['DVS'] == 1) & (df_fig6['country'] == country)]['DVS'].drop_duplicates()
        
        # Only set x-axis label for the last row
        if i == len(key_fig6) - 1:
            ax.set_xlabel('')
        else:
            ax.set_xticklabels([])
        if i == 0:
            ax.axvline(emergence_date.index, color='green', linestyle='-')
            ax.axvline(tuberinitiation_date.index, color='green', linestyle='-')
        if j == 1:
            ax.set_yticklabels([])    
        
        # Add subplot label
        subplot_label = labels[j * len(key_fig6) + i]
        ax.text(subplotlab_x, subplotlab_y - 0.1, subplot_label + ")", transform=ax.transAxes, size=config.subplot_fs - 2, weight='bold')
        ax.set_ylim(0, 20)

# Add shared y-axis label
fig.text(-0.02, 0.5, 'Tuber Dry Weight (t ha$^{-1}$)', va='center', rotation='vertical', fontsize=config.subplot_fs)
fig.text(0.5, 0.02, 'Days After Planting', ha='center', fontsize=config.subplot_fs) 
# Create colorbars on the right for each parameter
for i, param in enumerate(key_fig6):
    ax_cbar = colorbar_axes[i]
    data = df_fig6[df_fig6['key'] == param]
    cmap = plt.get_cmap('viridis')
    min_val = data['value'].min()
    max_val = data['value'].max()
    norm = Normalize(min_val, max_val)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Only needed for older versions of Matplotlib
 
    cbar = fig.colorbar(sm, cax=ax_cbar, label=config.label_map.get(param, param))
    cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))  # Ensure integer ticks
    cbar.ax.yaxis.set_major_formatter('{x:.0f}')  # Format ticks without decimals

# Save and show the plot
output_path = f'{config.p_out}/PosterLSA'
plt.savefig(f'{output_path}.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()
plt.rcParams['font.size'] = original_font_size
# %%
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

# %% Produce a fig that shows the last day of TWSO for both conditions with bar chart 
# load NLD manually
with open(f'../output_NL/daySi_32768/Saltelli_160_TWSO.pkl', 'rb') as f:
    Si_NLD = pickle.load(f)
Si_df_NLD = to_df(Si_NLD['si_day_160_TWSO'])
df_sorted_NLD = Si_df_NLD.sort_values(by='ST', ascending=True)
order = df_sorted_NLD.index
conf_cols = df_sorted_NLD.columns.str.contains('_conf')
confs = df_sorted_NLD.loc[:, conf_cols]
confs.columns = [c.replace('_conf', "") for c in confs.columns]

display = ['t2', 'te', 'TDWI','Q10', 'TSUM1', 't1_pheno', 'SPAN']
Sis_NLD = df_sorted_NLD.loc[display, ['ST']]

# %%
# load indian 
Si_IND = load_data(105, 'TWSO', 'Saltelli')
# %%
Si_df_IND = to_df(Si_IND['si_day_105_TWSO'])

df_sorted_new = Si_df_IND.reindex(order)
conf_cols = df_sorted_new.columns.str.contains('_conf')
confs = df_sorted_new.loc[:, conf_cols]
confs.columns = [c.replace('_conf', "") for c in confs.columns]
Sis_IND = df_sorted_new.loc[display, ['ST']]
# Save the original default font size
original_font_size = plt.rcParams['font.size']
# Set the new default font size
plt.rcParams['font.size'] = 18
color = ['red', 'blue']
fig, axs = plt.subplots(1, 2, figsize=(7, 6), sharex=True)

plt.subplots_adjust(wspace=0.15, hspace=0.05)
# NLD
barplot = Sis_NLD.plot(kind='barh' , width = 0.9, ax=axs[0],
                   legend=False, color = color)
# indian
barplot = Sis_IND.plot(kind='barh' , width = 0.9, ax=axs[1],
                   legend=False, color = color)
                   
# Define the label mapping
label_map = {
    't2': '$T_{opt}$ for $A_{max}$',
    'te': '$T_{max}$ for $A_{max}$',
    'TDWI': 'Seed DW',
    'Q10': 'Q10',
    'TSUM1': 'TSUM1',
    't1_pheno': '$T_b$',
    'SPAN': 'Leaf Lifespan'
}

yticklabels =  [item.get_text() for item in axs[0].get_yticklabels()]
new_yticklabels = [label_map.get(label, label) for label in yticklabels]
axs[0].set_yticklabels(new_yticklabels)
axs[1].set_yticklabels([])
fig.text(0.5, 0.02, 'Sensitivity Indices', ha='center') 
# barplot
plt.xlim(0, 1)
plt.show()
plt.rcParams['font.size'] = original_font_size
