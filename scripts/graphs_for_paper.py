# %%
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os
from run_vis_GSA import process_dvs_files, process_files, create_dataframe_from_dict, load_PAWN, normalize_sensitivity, config
import pickle
import pandas as pd
# Ensure these variables are defined or imported
with open('DummySi_results.pkl', 'rb') as f:
    Dummy_si = pickle.load(f)
GSA_sample_size = 32768  # Example definition, replace with actual value

original_font_size = plt.rcParams['font.size']
# plt.rcParams['font.size'] = 22

emergence_date, tuber_initiation = process_dvs_files()
# %% 
import RankingOverSeason as ros

base_path = "C:/Users/liu283/GitRepos/ch1_SA/"
col_variable = "DVS" 
file = os.path.join(base_path, f"output_NL_AUC_{col_variable}.csv") if config.run_NL_conditions else os.path.join(base_path, f"output_AUC_{col_variable}.csv")
df_pawn_ros, df_st_ros = ros.process_AUC_file(file)
df_ros = pd.merge(df_st_ros, df_pawn_ros, how='inner', on=['variable','label','country'])

col = 'DVS'
df_sensitivity_S1, df_sensitivity_ST = process_files(col)
df_pawn_long = create_dataframe_from_dict(load_PAWN(col))
df_pawn_long = df_pawn_long[df_pawn_long['median'] > Dummy_si[1][1]]
df_pawn_median = df_pawn_long.loc[:, ["DAP","median", "names"]].pivot_table(index='DAP', columns='names', values='median').reset_index()

df_pawn_median.set_index('DAP', inplace=True)
df_pawn_median.index.name = 'index'
df_sensitivity_ST, df_pawn_median = df_sensitivity_ST.align(df_pawn_median, axis=0, join='left')

print(f"Print 1st and total order Sobol indices for {col}.")
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(7, 9), sharex=True, sharey=True)
if col in ['LAI', 'TWSO']:
    start_date = emergence_date[0] if col == 'LAI' else tuber_initiation[0]
    df_sensitivity_ST = df_sensitivity_ST.iloc[start_date:]
    df_pawn = df_pawn_median.iloc[start_date:]
else:
    df_pawn = df_pawn_median

df2 = normalize_sensitivity(df_sensitivity_ST)
df3 = normalize_sensitivity(df_pawn)
# colors_final = [config.name_color_map.get(col, 'black') for col in df2.columns]
colors2 = [config.name_color_map.get(col, 'black') for col in df2.columns]
colors3 = [config.name_color_map.get(col, 'black') for col in df3.columns]
df2.plot.area(ax=axes[0],stacked=True, color=colors2, legend=False)
df3.plot.area(ax=axes[1],stacked=True, color=colors3, legend=False)
lines, labels = fig.axes[0].get_legend_handles_labels()
plt.ylim(0, 1.05)
plt.xlim(0, config.sim_period)
plt.xlabel('Day After Planting', fontsize = config.subplot_fs)
fig.text(0, 0.5, 'Proportion of Sensitivity indices', va='center', rotation='vertical', fontsize = config.subplot_fs)
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
plt.gca().invert_yaxis()
labels_final = [config.label_map.get(label, label) for label in labels]
# fig.legend(lines, labels_final, loc='center left', bbox_to_anchor=(1.0, 0.5), handlelength=1, borderpad=1, fontsize = 8)
labels_AUC = df_ros.variable.unique()

colors_AUC = [config.name_color_map.get(col, 'black') for col in labels_AUC]
labels_AUC = [config.label_map.get(label, label) for label in labels_AUC]
# labels_AUC = [f"{i+1}. {label}" for i, label in enumerate(labels_AUC)]
lines_AUC = [plt.Line2D([0], [0], color=c, linewidth=8, linestyle='-') for c in colors_AUC]
# fig.legend(lines_AUC, labels_AUC, loc='center left',  bbox_to_anchor=(1.2, 0.5), handlelength=0.3)
for i, ax in enumerate(axes.flatten(), start=1):
    i = i if config.run_NL_conditions else i+2
    ax.text(0.01, config.subplotlab_y, chr(96+i) + ")", transform=ax.transAxes, 
            size=config.subplot_fs , weight='bold')
    ax.fill_betweenx([1, 1.05], emergence_date[0], emergence_date[1], color='dimgray')
    ax.fill_betweenx([1, 1.05], tuber_initiation[0], tuber_initiation[1], color='dimgray')
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(['100%', '75%', '50%', '25%', '0%'])
scenario = 'NL_' if config.run_NL_conditions else ''
plt.tight_layout()
plt.savefig(f'{config.p_out}/{scenario}Sobol_Salteli_PAWN_{col}_samplesize{GSA_sample_size}.svg', bbox_inches='tight')
plt.show()
plt.close()

# %%

# Assuming lines and labels_final are already defined as per your code
fig, ax = plt.subplots(figsize=(12.5, 1), frameon=False)
ax.axis('off')  # Hide the axes

colors = [config.name_color_map.get(col, 'black') for col in config.params_of_interests]
labels = [config.label_map.get(label, label) for label in config.params_of_interests]
lines = [plt.Line2D([0], [0], color=c, linewidth=15, linestyle='-') for c in colors]
fig.legend(lines, labels, loc='center', ncol=8, handlelength = 1, handleheight = 2, borderpad=1,
           markerscale = 3, handletextpad=1, columnspacing = 1.5, fontsize=config.subplot_fs, frameon=False)

# plt.tight_layout(pad=0)  # Attempt to minimize padding, might not affect legends outside axes

# Save the figure as an SVG file with minimal padding around the legend
plt.savefig('../output/legend_graph.svg', format='svg')

# Show the plot
plt.show()

# %% Legend for DVS
fig, ax = plt.subplots(figsize=(12.5, 1), frameon=False)
ax.axis('off')  # Hide the axes
DVS_params = ['t1_pheno', 'TSUM1', 'TSUM2', 'TSUMEM', 'TBASEM', 'TEFFMX'] 
colors_DVS = [config.name_color_map.get(col, 'black') for col in DVS_params]
labels_DVS = [config.label_map.get(label, label) for label in DVS_params]
lines_DVS = [plt.Line2D([0], [0], color=c, linewidth=15, linestyle='-') for c in colors_DVS]
fig.legend(lines_DVS, labels_DVS, loc='center', ncol=6, handlelength = 1, handleheight = 2, borderpad=1,
              markerscale = 3, handletextpad=1, columnspacing = 1.5, fontsize=config.subplot_fs, frameon=False)
plt.savefig('../output/legend_DVS.svg', format='svg')
plt.show()



##  LSA ------------------------------
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
large_df['value'] = large_df['value'].astype(float)
colors = config.name_color_map
no_ofdays = len(large_df.day.unique())

# dfs_IND = []  # List to store DataFrames
# for i, key, value in zip(para_vals.keys(), keys, para_vals.values()):
#     # print(i, value)
#     with open(f'C:/Users/liu283/GitRepos/ch1_SA/output/LSA/sims_100/{i}_{key}.json', 'r') as f:
#         results = json.load(f)
#     df = pd.DataFrame(results)
#     df['key'] = key
#     df['value'] = value
#     dfs_IND.append(df)
# large_df_IND = pd.concat(dfs_IND)
# large_df_IND['value'] = large_df_IND['value'].astype(float)

# %%
# $t_{b\_pheno}$, TSUM1, TDWI, SPAN and $t_{phot-max}$.
DAPs = np.tile(np.arange(no_ofdays), config.LSA_sample_size * len(para))
large_df['DAP'] = DAPs[:len(large_df)]
large_df.set_index('DAP', inplace=True)

param_name = 't1_pheno'
output_var = 'LAI'
output_df = large_df[large_df['key'] == param_name].sort_values('day')
param_name_no_effect = 'TSUM1'
TSUM1 = large_df[large_df['key'] == 'TSUM1'].sort_values('day')
param_name_wirdo = 'te'
output_df_wirdo = large_df[large_df['key'] == param_name_wirdo].sort_values('day')
SPAN = large_df[large_df['key'] == 'SPAN'].sort_values('day')
TDWI = large_df[large_df['key'] == 'TDWI'].sort_values('day')
ylimt_upper = 6 # output_df.LAI.max() + 0.5
xlimt_upper = no_ofdays - 1
pointsize = 1
subplotlab_x = config.subplotlab_x
subplotlab_y = config.subplotlab_y
# relabel parameter names
param_name = config.label_map.get(param_name, param_name)
param_name_wirdo = config.label_map.get(param_name_wirdo, param_name_wirdo)
# DVS values 
emergence_date = output_df[output_df['DVS'] == 0]['DVS'].drop_duplicates()
tuberintiation_date = output_df[output_df['DVS'] == 1]['DVS'].drop_duplicates()

# %%

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

def create_plots(config, output_df, output_df_wirdo, TSUM1, SPAN, TDWI, param_name, param_name_wirdo, param_name_no_effect, output_var, pointsize, emergence_date, tuberintiation_date, subplotlab_x, subplotlab_y, ylimt_upper, no_ofdays):
    plt.rcParams.update({'font.size': config.subplot_fs})  # Adjust the font size as needed

    fig = plt.figure(figsize=(4, 12))

    # Create a GridSpec for the whole figure
    gs = gridspec.GridSpec(5, 1, figure=fig, hspace=0.2, wspace=0.3)

    def create_subplot(ax, data, output_var, param_name, label, index_label, pointsize, ylimt_upper, subplot_label):
        cmap = plt.get_cmap('viridis')
        norm = plt.Normalize(data['value'].min(), data['value'].max())
        sc = ax.scatter(data.index, data[output_var], c=data['value'], s=pointsize, cmap=cmap, norm=norm)
        # if not config.run_NL_conditions:
        fig.colorbar(sc, ax=ax, label=param_name)
        
        ax.set_xlabel('')
        # ax.set_ylabel(label)
        ax.set_ylim(0, ylimt_upper)
        ax.text(subplotlab_x, subplotlab_y - 0.1, subplot_label, transform=ax.transAxes, size=config.subplot_fs, weight='bold')

    # First subplot
    ax1 = fig.add_subplot(gs[0, 0])
    create_subplot(ax1, output_df, output_var, param_name, output_var, '', pointsize, ylimt_upper, 'f)')
    ax1.axvline(emergence_date.index, color='green', linestyle='-')
    ax1.axvline(tuberintiation_date.index, color='green', linestyle='-')
    ax1.set_xticklabels('')
    # Second subplot
    ax2 = fig.add_subplot(gs[1, 0])
    create_subplot(ax2, TSUM1, output_var, param_name_no_effect, output_var, '', pointsize + 4, ylimt_upper, 'g)')
    ax2.set_xticklabels('')
    # Third subplot
    ax3 = fig.add_subplot(gs[2, 0])
    create_subplot(ax3, SPAN, output_var, 'SPAN', output_var, 'DAP', pointsize + 4, ylimt_upper, 'h)')
    ax3.set_xticklabels('')
    # Fourth subplot
    ax4 = fig.add_subplot(gs[3, 0])
    create_subplot(ax4, TDWI, output_var, 'TDWI', '', '', pointsize + 4, ylimt_upper, 'i)')
    ax4.set_xticklabels('')
    # Fifth subplot
    ax5 = fig.add_subplot(gs[4, 0])
    create_subplot(ax5, output_df_wirdo, output_var, param_name_wirdo, '', 'DAP', pointsize + 4, ylimt_upper, 'j)')
    ax5.set_xlabel('DAP')
    # if config.run_NL_conditions:
    fig.text(0, 0.5, 'Leaf Area Index', va='center', rotation='vertical', fontsize=config.subplot_fs)

    # Save the figure
    scenario = "NL_" if config.run_NL_conditions else ""
    output_path = f'{config.p_out_LSA}/{scenario}{output_var}_mainText_{config.LSA_sample_size}'
    plt.savefig(f'{output_path}.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    # plt.savefig(f'{output_path}.svg', bbox_inches='tight', pad_inches=0.1)
    plt.show()

create_plots(config, output_df, output_df_wirdo, TSUM1, SPAN, TDWI, param_name, param_name_wirdo, param_name_no_effect, output_var, pointsize, emergence_date, tuberintiation_date, subplotlab_x, subplotlab_y, ylimt_upper, no_ofdays)

plt.rcParams['font.size'] = original_font_size

#  BACK TO GSA VISUALISATION
# %% # legend to rename and italicise
# import re

# # %%
print(f"Print 1st and total order Sobol indices for {col}.")
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6, 8), sharex=True, sharey=True)
df1 = normalize_sensitivity(df_sensitivity_S1)
df2 = normalize_sensitivity(df_sensitivity_ST)
df3 = normalize_sensitivity(df_pawn_median)
if col == 'LAI':
    df1 = df1.iloc[config.arbitrary_start:]
    df2 = df2.iloc[config.arbitrary_start:]
    df3 = df3.iloc[config.arbitrary_start:] 
# Combine the column names from both dataframes
# combined_columns = list(df1.columns) + [col for col in df2.columns if col not in df1.columns]
# Map the combined column names to colors
colors1 = [config.name_color_map.get(col, 'black') for col in df1.columns]
colors2 = [config.name_color_map.get(col, 'black') for col in df2.columns]
colors3 = [config.name_color_map.get(col, 'black') for col in df3.columns]
df1.plot.area(ax=axes[0],stacked=True, color=colors1, legend=False)
df2.plot.area(ax=axes[1],stacked=True, color=colors2, legend=False)
df3.plot.area(ax=axes[2],stacked=True, color=colors3, legend=False)
plt.ylim(0, 1)
plt.xlim(0, config.sim_period)
axes[0].set_xlabel('')
axes[1].set_xlabel('')
plt.xlabel('Day After Planting')
# plt.ylabel('Parameter sensitivity')
# Set the title in the middle of the figure
# fig.suptitle(f'First order and total Sobol Si for {col}')
fig.text(0, 0.5, 'Propostion of Sensitivity indices', va='center', rotation='vertical')
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
# Create a shared legend
lines, labels = fig.axes[-1].get_legend_handles_labels()

# Define a dictionary mapping old labels to new labels
label_map = {
    't1_pheno': r'$t_{b\_pheno}$',
    'te$': r'$t_{phot-max}$',
    'te_pheno': r'$t_{e\_pheno}$',
    't1': r'$t_1$',
    't2': r'$t_2$',
    'tm1': r'$t_{m1}$'
}

# Use a single list comprehension to apply replacements only for labels in label_map
labels = [label_map.get(label, label) for label in labels]
# fig.legend(lines, labels, loc='center left', bbox_to_anchor=(1.0, 0.5))
    # Add labels to the subplots
for i, ax in enumerate(axes.flatten(), start=1):
    ax.text(0.025, config.subplotlab_y, chr(96+i) + ")", transform=ax.transAxes, 
            size=config.subplot_fs, weight='bold')
plt.tight_layout()

# plt.savefig(f'{config.p_out}/Sobol_Salteli_PAWN_{col}_samplesize{GSA_sample_size}.svg', bbox_inches='tight')
plt.show()
plt.close()
# %% # test the code to plot the sensitivity indices after an arbitrary emergence date
# this is because the parameter values will affect the emergence date

df_sensitivity_ST, df_sensitivity_S1 = process_files('LAI')
df_pawn = create_dataframe_from_dict(load_PAWN('LAI'))

col = 'TWSO'
df_sensitivity_S1, df_sensitivity_ST = process_files(col)
df_pawn_long = create_dataframe_from_dict(load_PAWN(col))
df_pawn_long = df_pawn_long[df_pawn_long['median'] > Dummy_si[1][1]]
df_pawn_median = df_pawn_long.loc[:, ["DAP","median", "names"]].pivot_table(index='DAP', columns='names', values='median').reset_index()

df_pawn_median.set_index('DAP', inplace=True)
df_pawn_median.index.name = 'index'
df_sensitivity_ST, df_pawn_median = df_sensitivity_ST.align(df_pawn_median, axis=0, join='left')

# %%
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 7), sharex=True, sharey=True)
if col in ['LAI', 'TWSO']:
    start_date = emergence_date[0] if col == 'LAI' else tuber_initiation[0]
    df_sensitivity_ST = df_sensitivity_ST.iloc[start_date:]
    df_pawn = df_pawn_median.iloc[start_date:]
df2 = normalize_sensitivity(df_sensitivity_ST)
df3 = normalize_sensitivity(df_pawn)
colors_final = [config.name_color_map.get(col, 'black') for col in df3.columns]
df2.plot.area(ax=axes, stacked=True, color=colors_final, legend=False)

lines, labels = axes.get_legend_handles_labels()
axes.set_ylim(0, 1)
axes.set_xlim(0, config.sim_period)
axes.set_xlabel('Day After Planting', fontsize=config.subplot_fs)
axes.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
axes.invert_yaxis()
labels_final = [config.label_map.get(label, label) for label in labels]
# fig.legend(lines, labels_final, loc='center left', bbox_to_anchor=(1.0, 0.5), handlelength=1, borderpad=1, fontsize=18)
axes.set_yticks([0, 0.25, 0.5, 0.75, 1])
axes.set_yticklabels(['100%', '75%', '50%', '25%', '0%'])
scenario = 'NL_' if config.run_NL_conditions else ''
plt.tight_layout()
plt.savefig(os.path.join(config.p_out, f'{scenario}Sobol_Slide{col}_samplesize{GSA_sample_size}.svg'), bbox_inches='tight')
plt.show()
plt.close()
plt.rcParams['font.size'] = original_font_size




# %%

original_font_size = plt.rcParams['font.size']

# Set the new default font size
plt.rcParams['font.size'] = 22
emergence_date, tuber_initiation = process_dvs_files()
df_sensitivity_ST, df_sensitivity_S1 = process_files('LAI')
df_pawn = create_dataframe_from_dict(load_PAWN('LAI'))

# # # %% # test the code to fix the legend
col = 'TWSO'
df_sensitivity_S1, df_sensitivity_ST = process_files(col)
df_pawn_long = create_dataframe_from_dict(load_PAWN(col))
df_pawn_long = df_pawn_long[df_pawn_long['median'] > Dummy_si[1][1]]
df_pawn_median = df_pawn_long.loc[:, ["DAP","median", "names"]].pivot_table(index='DAP', columns='names', values='median').reset_index()

df_pawn_median.set_index('DAP', inplace=True)
df_pawn_median.index.name = 'index'
df_sensitivity_ST, df_pawn_median = df_sensitivity_ST.align(df_pawn_median, axis=0, join='left')
# %%
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 7), sharex=True, sharey=True)
if col in ['LAI', 'TWSO']:
    start_date = emergence_date[0] if col == 'LAI' else tuber_initiation[0]
    df_sensitivity_ST = df_sensitivity_ST.iloc[start_date:]
    df_pawn = df_pawn_median.iloc[start_date:]
df2 = normalize_sensitivity(df_sensitivity_ST)

df3 = normalize_sensitivity(df_pawn)
colors_final = [config.name_color_map.get(col, 'black') for col in df3.columns]
df2.plot.area(ax=axes, stacked=True, color=colors_final, legend=False)
# df3.plot.area(ax=axes, stacked=True, color=colors_final, legend=False)
lines, labels = axes.get_legend_handles_labels()
plt.ylim(0, 1)
plt.xlim(0, config.sim_period)
plt.xlabel('Day After Planting', fontsize = config.subplot_fs)
# fig.text(0, 0.5, 'Proportion of Sensitivity indices', va='center', rotation='vertical', fontsize = config.subplot_fs-4)
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
plt.gca().invert_yaxis()
labels_final = [config.label_map.get(label, label) for label in labels]
fig.legend(lines, labels_final, loc='center left', bbox_to_anchor=(1.0, 0.5), handlelength=1, borderpad=1, fontsize = 18)
axes.set_yticks([0, 0.25, 0.5, 0.75, 1])
axes.set_yticklabels(['100%', '75%', '50%', '25%', '0%'])
scenario = 'NL_' if config.run_NL_conditions else ''
plt.tight_layout()
plt.savefig(f'{config.p_out}/{scenario}Sobol_Slide{col}_samplesize{GSA_sample_size}.svg', bbox_inches='tight')
plt.show()
plt.close()
plt.rcParams['font.size'] = original_font_size



# %% 
# df_sensitivity_S1.head(20)
# normalize_sensitivity(df_sensitivity_S1).head(20)
# # df_pawn_long.head(20)
# # %%
# import json
# with open(f'{config.p_out_sims}/10.json', 'r') as f:
#     data = json.load(f)
#     data = pd.DataFrame(data)

# data[data['DVS'] == 0].day.unique()
