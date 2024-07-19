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
col_variable = "TWSO" 
file = os.path.join(base_path, f"output_NL_AUC_{col_variable}.csv") if config.run_NL_conditions else os.path.join(base_path, f"output_AUC_{col_variable}.csv")
df_pawn_ros, df_st_ros = ros.process_AUC_file(file)
df_ros = pd.merge(df_st_ros, df_pawn_ros, how='inner', on=['variable','label','country'])

col = 'TWSO'
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
# %%
DAPs = np.tile(np.arange(no_ofdays), config.LSA_sample_size * len(para))
large_df['DAP'] = DAPs[:len(large_df)]
large_df.set_index('DAP', inplace=True)
param_name = 't1_pheno'
output_var = 'LAI'
output_df = large_df[large_df['key'] == param_name].sort_values('day')
param_name_no_effect = 'RGRLAI'
output_df_no_effect = large_df[large_df['key'] == param_name_no_effect].sort_values('day')
param_name_wirdo = 'te'
output_df_wirdo = large_df[large_df['key'] == param_name_wirdo].sort_values('day')
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
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(7, 9))

# Create a GridSpec for the whole figure
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

# First subplot
ax1 = fig.add_subplot(gs[0, 0])
cmap = plt.get_cmap('viridis')
norm = plt.Normalize(output_df['value'].min(), output_df['value'].max())
sc = ax1.scatter(output_df.index, output_df[output_var], c=output_df['value'], s = pointsize, cmap=cmap, norm=norm)
# fig.colorbar(sc, ax=ax1)
ax1.set_xlabel('')
ax1.set_ylabel(output_var)
ax1.axvline(emergence_date.index, color='green', linestyle='-')
ax1.axvline(tuberintiation_date.index, color='green', linestyle='-')
ax1.set_ylim(0, ylimt_upper)
# ax1.set_xlim(0, xlimt_upper)
ax1.vlines(xlimt_upper, 0, ylimt_upper, color='red', linestyle='--')
# ax1.annotate('Illustration on the right', xy=(xlimt_upper, ylimt_upper),
#               xytext=(xlimt_upper, ylimt_upper + 1), 
#              arrowprops=dict(facecolor='black', shrink=0.0), ha = 'center')
ax1.text(subplotlab_x, subplotlab_y, 'a)', transform=ax1.transAxes, size=config.subplot_fs, weight='bold')
# Second subplot
ax2 = fig.add_subplot(gs[0, 1])
# Get the final day of output_df
final_day_df = output_df[output_df.index == output_df.index.max()]

# Plot LAI against parameter values
sc2 = ax2.scatter(final_day_df['value'], final_day_df[output_var], c=final_day_df['value'], s = pointsize + 4, cmap=cmap, norm=norm)
fig.colorbar(sc2, ax=ax2)
ax2.set_xlabel(param_name)
ax2.set_ylabel(output_var)
ax2.text(subplotlab_x, subplotlab_y, 'b)', transform=ax2.transAxes, size=config.subplot_fs, weight='bold')
# Third subplot
ax3 = fig.add_subplot(gs[1, 0])
cmap = plt.get_cmap('viridis')
norm = plt.Normalize(output_df_wirdo['value'].min(), output_df_wirdo['value'].max())
sc3 = ax3.scatter(output_df_wirdo.index, output_df_wirdo[output_var], c=output_df_wirdo['value'], s = pointsize + 4, cmap=cmap, norm=norm)
# fig.colorbar(sc, ax=ax1)
ax3.set_xlabel('')
ax3.set_ylabel(output_var)
ax3.set_ylim(0, ylimt_upper)
# ax3.set_xlim(0, xlimt_upper)
xlimt_upper = (no_ofdays - 1)/2 #if config.run_NL_conditions else no_ofdays - 1
ax3.vlines(xlimt_upper, 0, ylimt_upper, color='red', linestyle='--')
# ax3.annotate('Illustration on the right', xy=(xlimt_upper, ylimt_upper),
#               xytext=(xlimt_upper, ylimt_upper + 1), 
#              arrowprops=dict(facecolor='black', shrink=0.0), ha = 'center')
ax3.text(subplotlab_x, subplotlab_y, 'c)', transform=ax3.transAxes, size=config.subplot_fs, weight='bold')

# Fourth subplot
# from mpl_toolkits.axes_grid1 import make_axes_locatable

# Fourth subplot
ax4 = fig.add_subplot(gs[1, 1])
# Get the day of output_df for x and y scatter plot
selected_day = round((no_ofdays - 1)/2) # if config.run_NL_conditions else no_ofdays - 1
final_day_df_wirdo = output_df_wirdo[output_df_wirdo.index == selected_day]

# Plot LAI against parameter values
sc4 = ax4.scatter(final_day_df_wirdo['value'], final_day_df_wirdo[output_var], c=final_day_df_wirdo['value'], s = pointsize + 4, cmap=cmap, norm=norm)
fig.colorbar(sc4, ax=ax4)
ax4.set_xlabel(param_name_wirdo)
ax4.set_ylabel(output_var)
ax4.text(subplotlab_x, subplotlab_y, 'd)', transform=ax4.transAxes, size=config.subplot_fs, weight='bold')

# Fifth subplot output_df_wirdo
ax5 = fig.add_subplot(gs[2, 0])
cmap = plt.get_cmap('viridis')
norm = plt.Normalize(output_df_no_effect['value'].min(), output_df_no_effect['value'].max())
sc5 = ax5.scatter(output_df_no_effect.index, output_df_no_effect[output_var], c=output_df_no_effect['value'], s = pointsize + 4, cmap=cmap, norm=norm)
# fig.colorbar(sc, ax=ax1)
ax5.set_xlabel('DAP')
ax5.set_ylabel(output_var)
ax5.text(subplotlab_x, subplotlab_y, 'e)', transform=ax5.transAxes, size=config.subplot_fs, weight='bold')
xlimt_upper = (no_ofdays - 1)/2 if config.run_NL_conditions else no_ofdays - 1
ax5.vlines(xlimt_upper, 0, ylimt_upper, color='red', linestyle='--')
# ax5.annotate('Illustration on the right', xy=(xlimt_upper, ylimt_upper ),
            #   xytext=(xlimt_upper, ylimt_upper + 1), 
            #  arrowprops=dict(facecolor='black', shrink=0.0), ha = 'center')
ax5.set_ylim(0, ylimt_upper)
# ax5.set_xlim(0, xlimt_upper)
# Sixth subplot
ax6 = fig.add_subplot(gs[2, 1])
# Get the final day of output_df
final_day_df_no_effect = output_df_no_effect[output_df_no_effect.index == selected_day]

# Plot LAI against parameter values
sc6 = ax6.scatter(final_day_df_no_effect['value'], final_day_df_no_effect[output_var], c=final_day_df_no_effect['value'], s = pointsize + 4, cmap=cmap, norm=norm)
fig.colorbar(sc6, ax=ax6)
ax6.set_xlabel(param_name_no_effect)
ax6.set_ylabel(output_var)
ax6.text(subplotlab_x, subplotlab_y, 'f)', transform=ax6.transAxes, size=config.subplot_fs, weight='bold')

scenario = "NL_" if config.run_NL_conditions else ""
plt.savefig(f'{config.p_out_LSA}/{scenario}{output_var}_mainText_{config.LSA_sample_size}.png', dpi = 300, bbox_inches='tight', pad_inches=0.1)
plt.savefig(f'{config.p_out_LSA}/{scenario}{output_var}_mainText_{config.LSA_sample_size}.svg', bbox_inches='tight', pad_inches=0.1)
plt.show()


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
