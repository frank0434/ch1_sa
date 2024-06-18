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
# %%
# Replace the values in the 'key' column with the corresponding values from 'label_map'
# large_df['key'] = large_df['key'].replace(config.label_map)
# %%
def create_subplot(ax, output_df, output_var, param_name):
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(output_df['value'].min(), output_df['value'].max())
    sc = ax.scatter(output_df.index, output_df[output_var], 
                    c=output_df['value'], cmap=cmap, norm=norm, s = 1)
    return sc

def create_figure(large_df, output_var, param_names):
    fig, axs = plt.subplots(5, 3, figsize=(9, 12), sharex=True, sharey=True)    
    # fig.subplots_adjust(wspace=-.5, hspace= -0.5)
    axs = axs.flatten()

    for i, param_name in enumerate(param_names):
        output_df = large_df[large_df['key'] == param_name].sort_values('day')
        # output_df.set_index('DAP', inplace=True)
        sc = create_subplot(axs[i], output_df, output_var, param_name)
        fig.colorbar(sc, ax=axs[i])

        if i >= 12:
            axs[i].set_xlabel('DAP')
        if i % 3 == 0:
            axs[i].set_ylabel(output_var)    
        # Replace param_name with the corresponding value from label_map if it exists
        param_name = config.label_map.get(param_name, param_name)

        axs[i].set_title(f'{param_name}')
        if output_var == 'LAI':
            axs[i].set_ylim(0, 6)
        elif output_var == 'TWSO':
            axs[i].set_ylim(0, output_df.TWSO.max() + 5000)
    scenario = "NL_" if config.run_NL_conditions else ""
    plt.savefig(f'{config.p_out_LSA}/{scenario}{output_var}_timeseries_ss_{config.LSA_sample_size}.png', dpi = 300, pad_inches=0.1)
    plt.savefig(f'{config.p_out_LSA}/{scenario}{output_var}_timeseries_ss_{config.LSA_sample_size}.svg', pad_inches=0.1)
    plt.show()
# %%

for var in ['LAI', 'TWSO', 'DVS']:
    create_figure(large_df, var, config.params_of_interests)

# %% single plot for the main text --------------------------------
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
ax1.set_xlim(0, xlimt_upper)
# ax1.vlines(xlimt_upper, 0, ylimt_upper, color='red', linestyle='--')
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
ax3.set_xlim(0, xlimt_upper)
# xlimt_upper = (no_ofdays - 1)/2 #if config.run_NL_conditions else no_ofdays - 1
# ax3.vlines(xlimt_upper, 0, ylimt_upper, color='red', linestyle='--')
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
# xlimt_upper = (no_ofdays - 1)/2 if config.run_NL_conditions else no_ofdays - 1
# ax5.vlines(xlimt_upper, 0, ylimt_upper, color='red', linestyle='--')
# ax5.annotate('Illustration on the right', xy=(xlimt_upper, ylimt_upper ),
            #   xytext=(xlimt_upper, ylimt_upper + 1), 
            #  arrowprops=dict(facecolor='black', shrink=0.0), ha = 'center')
ax5.set_ylim(0, ylimt_upper)
ax5.set_xlim(0, xlimt_upper)
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
# %%  make plot for the slide 
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
# Save the original default font size
original_font_size = plt.rcParams['font.size']

# Set the new default font size
plt.rcParams['font.size'] = 22


import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(6, 10))

# Create a GridSpec for the whole figure
gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.15)

# First subplot
# Third subplot
ax3 = fig.add_subplot(gs[0, 0])
cmap = plt.get_cmap('viridis')
norm = plt.Normalize(output_df_wirdo['value'].min(), output_df_wirdo['value'].max())
sc3 = ax3.scatter(output_df_wirdo.index, output_df_wirdo[output_var], c=output_df_wirdo['value'], s = pointsize + 4, cmap=cmap, norm=norm)
# fig.colorbar(sc, ax=ax1)
ax3.set_xlabel('')
ax3.set_ylabel(output_var)
ax3.set_ylim(0, ylimt_upper)
ax3.set_xlim(0, xlimt_upper)
ax3.text(subplotlab_x, subplotlab_y, '$Base\ Temperature$', transform=ax3.transAxes, size=18, weight='bold')
fig.colorbar(sc3, ax=ax3)
# Fourth subplot

# Fifth subplot output_df_wirdo
ax5 = fig.add_subplot(gs[1, 0])
cmap = plt.get_cmap('viridis')
norm = plt.Normalize(output_df_no_effect['value'].min(), output_df_no_effect['value'].max())
sc5 = ax5.scatter(output_df_no_effect.index, output_df_no_effect[output_var], c=output_df_no_effect['value'], s = pointsize + 4, cmap=cmap, norm=norm)
# fig.colorbar(sc, ax=ax1)
ax5.set_xlabel('DAP')
ax5.set_ylabel(output_var)
ax5.text(subplotlab_x, subplotlab_y, '$Relative\ Growth\ Rate\ LAI$', transform=ax5.transAxes, size = 18, weight='bold')
ax5.set_ylim(0, ylimt_upper)
ax5.set_xlim(0, xlimt_upper)
fig.colorbar(sc5, ax=ax5)
# Sixth subplot

scenario = "NL_" if config.run_NL_conditions else ""
plt.savefig(f'{config.p_out_LSA}/{scenario}{output_var}_slide_{config.LSA_sample_size}.png', dpi = 300, bbox_inches='tight', pad_inches=0.1)
plt.savefig(f'{config.p_out_LSA}/{scenario}{output_var}_slide_{config.LSA_sample_size}.svg', bbox_inches='tight', pad_inches=0.1)
plt.show()

# Reset the default font size to its original value
plt.rcParams['font.size'] = original_font_size