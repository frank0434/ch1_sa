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
import utilities as ros
from run_vis_GSA import process_dvs_files, process_files, create_dataframe_from_dict, load_PAWN, normalize_sensitivity
import os
colors = ['#0072B2', '#E69F00', '#009E73', '#D3D3D3', '#696969']
config.set_variables(config.GSA_sample_size)
original_font_size = plt.rcParams['font.size']
plt.rcParams['font.size'] = config.subplot_fs
# %%  Figure 1 DVS, LAI, TWSO in both NL and IND - manual run with modifing config.py


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
file = os.path.join(base_path, f"output_AUC_{col}.csv") 
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
# %%
# Align dataframes
df_pawn_median.set_index('DAP', inplace=True)
df_pawn_median_NL.set_index('DAP', inplace=True)
df_sensitivity_ST, df_pawn_median = df_sensitivity_ST.align(df_pawn_median, axis=0, join='left')
df_sensitivity_ST_NL, df_pawn_median_NL = df_sensitivity_ST_NL.align(df_pawn_median_NL, axis=0, join='left')
# Print sensitivity indices
print(f"Print 1st and total order Sobol indices for {col}.")
# %%
# # Adjust data based on column type
# if col in ['LAI', 'TWSO']:
#     start_date = emergence_date[0] if col == 'LAI' else tuber_initiation[0]
#     df_sensitivity_ST = df_sensitivity_ST.iloc[start_date:]
#     df_pawn = df_pawn_median.iloc[start_date:]
#     start_date_NL = emergence_date_NL[0] if col == 'LAI' else tuber_initiation_NL[0]
#     df_sensitivity_ST_NL = df_sensitivity_ST_NL.iloc[start_date_NL:]
#     df_pawn_NL = df_pawn_median_NL.iloc[start_date_NL:]

# else:
#     df_pawn = df_pawn_median
#     df_pawn_NL = df_pawn_median_NL
# %%
# Normalize sensitivity data
df2 = normalize_sensitivity(df_sensitivity_ST)
df3 = normalize_sensitivity(df_pawn_median)
df2_NL = normalize_sensitivity(df_sensitivity_ST_NL)
df3_NL = normalize_sensitivity(df_pawn_median_NL)

# Define colors
colors2 = [config.name_color_map.get(col, 'black') for col in df2.columns]
colors3 = [config.name_color_map.get(col, 'black') for col in df3.columns]
colors2_NL = [config.name_color_map.get(col, 'black') for col in df2_NL.columns]
colors3_NL = [config.name_color_map.get(col, 'black') for col in df3_NL.columns]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7, 5), sharey=True)

# Plot data for NL
df2_NL.plot.area(ax=axes[0, 0], stacked=True, color=colors2_NL, legend=False)
axes[0, 0].set_title('NL', fontsize=config.subplot_fs)  # Set title with custom font size

df3_NL.plot.area(ax=axes[1, 0], stacked=True, color=colors3_NL, legend=False)

# Plot data for IND
df2.plot.area(ax=axes[0, 1], stacked=True, color=colors2, legend=False)
axes[0, 1].set_title('India', fontsize=config.subplot_fs)  # Set title with custom font size
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
    if i % 2 == 0:
        ax.set_xlim([0, 161])  # Adjust the limits as needed
    if i % 2 == 1:
        ax.set_xlim([0, 106])
    ax.tick_params(labelsize=config.subplot_fs - 6)
fig.text(0.9, 0.3, 'PAWN', va='center', rotation=270, fontsize=config.subplot_fs)
fig.text(0.9, 0.7, 'Sobol', va='center', rotation=270, fontsize=config.subplot_fs)

plt.subplots_adjust(hspace=0.05, wspace=0.05)  # Adjust hspace and wspace as needed

# Save and show plot
scenario = 'NL_' if config.run_NL_conditions else ''
# plt.tight_layout()
plt.savefig(f'{config.p_out}/{scenario}Sobol_Salteli_PAWN_{col}_poster.svg', bbox_inches='tight')
plt.savefig(f'{config.p_out}/{scenario}Sobol_Salteli_PAWN_{col}_poster.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()


# %% bar plot
import seaborn as sns
df_ros_sorted = df_ros.sort_values(by='ST', ascending=True).set_index('variable')
df_ros_sorted_NL = df_ros_NL.sort_values(by='ST', ascending=True).set_index('variable')
df_ros_sorted_NL.loc['ST'] = 0
df_ros_sorted['country'] = 'India'
df_ros_sorted_NL['country'] = 'NL'

order = df_ros_sorted_NL.index
# df_ros_sorted_NL = df_ros_sorted_NL.reindex(order)
# df_ros_sorted = df_ros_sorted.reindex(order)
display = ['te','TDWI',  'TSUM1', 't1_pheno', 'SPAN']
Sis_ST_NLD = df_ros_sorted_NL.loc[display, ['ST', 'country']]
Sis_PAWN_NLD = df_ros_sorted_NL.loc[display, ['PAWN', 'country']]
Sis_ST_IND = df_ros_sorted.loc[display, ['ST', 'country']]
Sis_PAWN_IND = df_ros_sorted.loc[display, ['PAWN', 'country']]

# %%
# Create a 2x2 grid of subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 5), sharex=True)

# Plot data for IND
Sis_ST_NLD.plot.barh(ax=axes[0, 0], legend=False, color='red')
axes[0, 0].set_title('NL', fontsize=config.subplot_fs)  # Set title with custom font size
Sis_PAWN_NLD.plot.barh(ax=axes[1, 0],  legend=False, color='black')

# Plot data for NL
Sis_ST_IND.plot.barh(ax=axes[0, 1], legend=False, color='red')
axes[0, 1].set_title('India', fontsize=config.subplot_fs)  # Set title with custom font size
Sis_PAWN_IND.plot.barh(ax=axes[1, 1], legend=False, color='black')

# Add subplot labels
# labels = ['a)', 'c)', 'b)', 'd)']
for i, ax in enumerate(axes.flatten()):
    # label = labels[i]
    # ax.text(0.9, 0.9, label, transform=ax.transAxes, size=config.subplot_fs,weight='bold')
    ax.set_ylabel('')
    if i % 2 == 0:        
        yticklabels =  [item.get_text() for item in ax.get_yticklabels()]
        new_yticklabels = [config.label_map.get(label, label) for label in yticklabels]
        ax.set_yticklabels(new_yticklabels)
        
    else:
        ax.set_yticklabels([])
        

fig.text(0.5, 0.02, 'Integrated Sensitivity Indices', ha='center', fontsize=config.subplot_fs) 
fig.text(0.91, 0.3, 'PAWN', va='center', rotation=270, fontsize=config.subplot_fs)
fig.text(0.91, 0.7, 'Sobol', va='center', rotation=270, fontsize=config.subplot_fs)

# Adjust the spacing between subplots
plt.subplots_adjust(hspace=0.1, wspace=0.05)

# Save and show plot

plt.savefig(f'{config.p_out}/AUC_barh_plots.svg', bbox_inches='tight')
plt.savefig(f'{config.p_out}/AUC_barh_plots.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

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
df_fig6_IND['country'] = 'India'
df_fig6_NL['country'] = 'NL'
df_fig6 = pd.concat([df_fig6_IND, df_fig6_NL])
df_fig6.set_index('DAP', inplace=True)
countries = ['NL', 'India']
df_fig6['key'] = pd.Categorical(df_fig6['key'], key_fig6)
df_fig6['country'] = pd.Categorical(df_fig6['country'], categories=countries, ordered=True)
df_fig6['TWSO'] = df_fig6['TWSO'] / 1000  # Convert to t ha-1
# %%
# Figure 2
# Import necessary libraries
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
# Assuming `key_fig6`, `countries`, `df_fig6`, and `config` are defined in the user's context.
plt.rcParams['font.size'] = config.subplot_fs
# Create a figure with subplots on the left and colorbars on the right
fig = plt.figure(figsize=(5,7))
gs = gridspec.GridSpec(3, 3, width_ratios=[4, 4, 0.5], wspace=0.1, hspace=0.1)  # Adjust the width ratio for colorbars
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
            ax.set_title(country)
            # ax.axvline(emergence_date.index, color='green', linestyle='-')
            # ax.axvline(tuberinitiation_date.index, color='green', linestyle='-')
        if j == 1:
            ax.set_yticklabels([])    
        
        # Add subplot label
        subplot_label = labels[j * len(key_fig6) + i]
        ax.text(subplotlab_x, subplotlab_y - 0.1, subplot_label + ")", transform=ax.transAxes, size=config.subplot_fs , weight='bold')
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
plt.savefig(f'{output_path}.svg', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()
plt.rcParams['font.size'] = original_font_size

# %% Figure 1 
# Produce a fig that shows the last day of TWSO for both conditions with bar chart 
# load NLD manually
with open(f'../output_NL/daySi_32768/Saltelli_160_TWSO.pkl', 'rb') as f:
    Si_NLD = pickle.load(f)
Si_df_NLD = ros.to_df(Si_NLD['si_day_160_TWSO'])
df_sorted_NLD = Si_df_NLD.sort_values(by='ST', ascending=True)
order = df_sorted_NLD.index
conf_cols = df_sorted_NLD.columns.str.contains('_conf')
confs = df_sorted_NLD.loc[:, conf_cols]
confs.columns = [c.replace('_conf', "") for c in confs.columns]

display = ['t2', 'te', 'TDWI','Q10', 'TSUM1', 't1_pheno', 'SPAN']
Sis_NLD = df_sorted_NLD.loc[display, ['ST']]

# load indian 
Si_IND = ros.load_data(105, 'TWSO', 'Saltelli')
# %%
Si_df_IND = ros.to_df(Si_IND['si_day_105_TWSO'])

df_sorted_new = Si_df_IND.reindex(order)
conf_cols = df_sorted_new.columns.str.contains('_conf')
confs = df_sorted_new.loc[:, conf_cols]
confs.columns = [c.replace('_conf', "") for c in confs.columns]
Sis_IND = df_sorted_new.loc[display, ['ST']]
# Set the new default font size

color = ['red', 'blue']
fig, axs = plt.subplots(1, 2, figsize=(7, 6), sharex=True)

plt.subplots_adjust(wspace=0.15, hspace=0.05)
# NLD
barplot = Sis_NLD.plot(kind='barh' , width = 0.9, ax=axs[0],
                   legend=False, color = color, title='NL')
# indian
barplot = Sis_IND.plot(kind='barh' , width = 0.9, ax=axs[1],
                   legend=False, color = color, title='India')
yticklabels =  [item.get_text() for item in axs[0].get_yticklabels()]
new_yticklabels = [config.label_map.get(label, label) for label in yticklabels]
axs[0].set_yticklabels(new_yticklabels)
axs[1].set_yticklabels([])
# axs[0].tick_params(labelsize=config.subplot_fs - 6)

fig.text(0.5, 0.02, 'Sensitivity Indices', ha='center') 
# barplot
plt.xlim(0, 1)
plt.savefig(f'{config.p_out}/PosterSobol_TWSO.svg', dpi=300, bbox_inches='tight')
plt.show()

# %% 
def create_legend_figure(params, colors, labels, output_path, ncol):
    fig, ax = plt.subplots(figsize=(5, 5), frameon=False)
    ax.axis('off')  # Hide the axes
    lines = [plt.Line2D([0], [0], color=c, linewidth=15, linestyle='-') for c in colors]
    fig.legend(lines, labels, loc='center', ncol=ncol, handlelength=1, handleheight=2, borderpad=1,
               markerscale=3, handletextpad=1, columnspacing=1.5, fontsize=config.subplot_fs + 4, frameon=False)
    plt.savefig(output_path, format='png', transparent=True)
    plt.savefig(output_path.replace('.png', '.svg'), format='svg', transparent=True)
    plt.show()
    plt.close()

# TWSO legend
TWSO_params = config.params_of_interests
colors_TWSO = [config.name_color_map.get(col, 'black') for col in TWSO_params]
labels_TWSO = [config.label_map.get(label, label) for label in TWSO_params]
create_legend_figure(TWSO_params, colors_TWSO, labels_TWSO, '../output/posterlegend_TWSO.png', ncol=2)

#%% 
Sis_IND_pct = Sis_IND * 100
Sis_NLD_pct = Sis_NLD * 100
# %% FIGURE EXTRA FOR THE REVISTED PAPER 
# Create a 2x2 grid of subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 5), sharex=True)

# Plot data for IND
Sis_ST_NLD.plot.barh(ax=axes[0, 0], legend=False, color='red')
axes[0, 0].set_title('NL', fontsize=config.subplot_fs)  # Set title with custom font size
Sis_NLD_pct.plot.barh(ax=axes[1,0], legend=False, color = color)
# Plot data for NL
Sis_ST_IND.plot.barh(ax=axes[0, 1], legend=False, color='red')
axes[0, 1].set_title('India', fontsize=config.subplot_fs)  # Set title with custom font size
Sis_IND_pct.plot.barh( ax=axes[1,1], legend=False, color = color)
# Add subplot labels
# labels = ['a)', 'c)', 'b)', 'd)']
for i, ax in enumerate(axes.flatten()):
    # label = labels[i]
    # ax.text(0.9, 0.9, label, transform=ax.transAxes, size=config.subplot_fs,weight='bold')
    ax.set_ylabel('')
    if i % 2 == 0:        
        yticklabels =  [item.get_text() for item in ax.get_yticklabels()]
        new_yticklabels = [config.label_map.get(label, label) for label in yticklabels]
        ax.set_yticklabels(new_yticklabels)
        
    else:
        ax.set_yticklabels([])
        

fig.text(0.5, 0.02, 'Integrated Sensitivity Indices', ha='center', fontsize=config.subplot_fs) 
fig.text(0.91, 0.3, 'Final Output\nRanking', va='center', rotation=270, fontsize=config.subplot_fs)
fig.text(0.91, 0.7, 'Seasonal\nRanking', va='center', rotation=270, fontsize=config.subplot_fs)

# Adjust the spacing between subplots
plt.subplots_adjust(hspace=0.1, wspace=0.05)

# Save and show plot

plt.savefig(f'{config.p_out}/AUC_barh_ranking_compare_plots.svg', bbox_inches='tight')
plt.savefig(f'{config.p_out}/AUC_barh_ranking_compare_plots.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# %%
