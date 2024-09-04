# %%
# Figure one 
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
original_font_size = plt.rcParams['font.size']


AGFUN_DTSMTB = [0.0,  0.0,
                2.0,  0.0,
               13.0, 11.0,
               30.0, 28.0]
x = AGFUN_DTSMTB[::2]  # Values in odd positions
y = AGFUN_DTSMTB[1::2]  # Values in even positions

# %%
def combined_equation(Temp_mean, t1, te):
    """
    This is a smooth function for DTSMTB based on the value of Temp_mean.

    Parameters:
    Temp_mean (float): The mean air temperature.
    t1, te (float): Parameters of the piecewise function.

    Returns:
    float: The result of the piecewise function.
    """
    condition1 = Temp_mean <= t1
    condition2 = (Temp_mean > t1)  & (Temp_mean <= te)

    # Define the equations 
    equation1 = 0
    equation2 = Temp_mean - t1
    # Calculate the result of the piecewise function
    result = np.where(condition1, equation1, equation2)

    return result


# %%
# Initial guess: [t1, te]
p0 = [4, 30]

# Fit the curve
popt, pcov = curve_fit(combined_equation, x, y, p0, method='dogbox', maxfev=5000)

# Generate y data from the combined_equation function with the optimized parameters
x_smooth = np.linspace(min(x), max(x), 100)
y_smooth = combined_equation(x_smooth, *popt)

# Plot the original data and the fitted curve
plt.plot(x, y, 'o', label='Data points sampled from AFGEN')
plt.plot(x_smooth, y_smooth, label='Fitted curve', c = 'red')
plt.xlabel('Mean Air Temperature (°C)')
plt.ylabel('Effective Thermal Time (°C day)')
plt.legend()


plt.savefig('../output/DTSMTB_curving_fitting.svg')
plt.show()
plt.close()
plt.rcParams['font.size'] = original_font_size

#%% # alternative for DTSMTB
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Original data
AGFUN_DTSMTB = [0.0,  0.0,
                2.0,  0.0,
               13.0, 11.0,
               30.0, 28.0]

x = AGFUN_DTSMTB[::2]  # Values in odd positions (Temperature)
y = AGFUN_DTSMTB[1::2]  # Values in even positions (Effective Thermal Time)

# Sigmoid function definition
def sigmoid_equation(Temp_mean, L, x0, k, t1):
    """
    Sigmoid function for smooth transition between two states.

    Parameters:
    Temp_mean (float): The mean air temperature.
    L (float): The upper limit of the sigmoid function (max effective thermal time).
    x0 (float): The temperature at which the sigmoid transitions.
    k (float): The steepness of the sigmoid curve.
    t1 (float): The threshold temperature below which the thermal time is zero.

    Returns:
    float: The calculated effective thermal time.
    """
    # Sigmoid component
    sigmoid_component = L / (1 + np.exp(-k * (Temp_mean - x0)))
    
    # Piecewise linear with sigmoid for smooth transition
    result = np.where(Temp_mean <= t1, 0, sigmoid_component - sigmoid_component[Temp_mean <= t1].min())
    
    return result

# Initial guess for the parameters [L, x0, k, t1]
p0 = [30, 2, 1, 2]

# Fit the sigmoid function to the data
popt, pcov = curve_fit(sigmoid_equation, x, y, p0, method='dogbox', maxfev=5000)

# Generate smooth data from the fitted function
x_smooth = np.linspace(min(x), max(x), 100)
y_smooth = sigmoid_equation(x_smooth, *popt)

# Plot the original data and the fitted curve
plt.plot(x, y, 'o', label='Data points sampled from AFGEN')
plt.plot(x_smooth, y_smooth, label='Fitted sigmoid curve', c='red')
plt.xlabel('Mean Air Temperature (°C)')
plt.ylabel('Effective Thermal Time (°C day)')
plt.legend(loc = "upper left")

plt.savefig('DTSMTB_sigmoid_fitting.svg')
plt.show()
plt.close()

#%%

import numpy as np
import matplotlib.pyplot as plt
def combined_equation(Temp_daytime, tm1, t1, t2, te):
    return np.where(Temp_daytime < t1, 
                    1 * (1 + (t1 - Temp_daytime) / (t1 - tm1)) * (Temp_daytime / t1) ** (t1 / (t1 - tm1)),
                    np.where((Temp_daytime >= t1) & (Temp_daytime <= t2), 1, 
                             1 * ((te - Temp_daytime)/(te - t2))*((Temp_daytime + t1 - t2)/t1)**(t1/(te - t2))))
from scipy.optimize import curve_fit


AGFUN_TMPFTB = [0.0,0.01,3.0,0.01,10.0,0.75, 11, 0.8,  12, 0.85, 13, 0.9, 
                14, 0.95, 15.0,1.0,24.0,1.0,29.0,0.75,36.0,0.01]
x = AGFUN_TMPFTB[::2]  # Values in odd positions
y = AGFUN_TMPFTB[1::2]  # Values in even positions
plt.plot(x, y)
plt.xlabel('Temperature (°C)')
plt.ylabel('Reduction factor of \nMaximum leaf CO2 assimilation rate')

# Initial guess: [tm1, t1, t2, te, 1]
p0 = [0, 10, 20, 30]

# Fit the curve
popt, pcov = curve_fit(combined_equation, x, y, p0, method='dogbox', maxfev=5000)

# Generate y data from the combined_equation function with the optimized parameters
x_smooth = np.linspace(min(x), max(x), 100)
y_smooth = combined_equation(x_smooth, *popt)

# Plot the original data and the fitted curve
plt.plot(x, y, 'o', label='Data points sampled from AFGEN_TMPFTB', c = 'blue')
plt.plot(x_smooth, y_smooth, label='Fitted curve', c = 'red')
plt.xlabel('Mean Daytime Temperature (°C)')
plt.ylabel('Reduction factor of \nMaximum leaf CO2 assimilation rate')
plt.legend()

plt.savefig('../output/TMPFTB_curving_fitting.svg')
plt.show()
plt.close()


# %%
# Figure 2 
# %%
import pandas as pd
import matplotlib.pyplot as plt
import config
from pcse.fileinput import ExcelWeatherDataProvider
# import Todo_before_writing as daylength_data
from datetime import datetime
# Load the data
wdp_NL = ExcelWeatherDataProvider('../data_raw\\350_weatherfile_2021.xlsx')
wdp_IND = ExcelWeatherDataProvider('../data_raw\\India2022_23.xlsx')
# %%
df_NL = pd.DataFrame(wdp_NL.export())
df_IND = pd.DataFrame(wdp_IND.export())
config.SIMULATION_START_DATE 
config.SIMULATION_END_DATE
df_NL.set_index('DAY', inplace=True)
df_IND.set_index('DAY', inplace=True)
# %%
# Filter the DataFrame

# Now you can filter the DataFrame
df_IND = df_IND[(df_IND.index >= datetime.strptime("2022-11-10", '%Y-%m-%d').date()) & (df_IND.index <= datetime.strptime("2023-02-28",'%Y-%m-%d').date())]

df_NL = df_NL[(df_NL.index >= datetime.strptime("2021-04-20", '%Y-%m-%d').date()) & (df_NL.index <= datetime.strptime("2021-09-30",'%Y-%m-%d').date())]

# %%

df_IND['Tmean'] = (df_IND['TMAX'] + df_IND['TMIN']) / 2
df_NL['Tmean'] = (df_NL['TMAX'] + df_NL['TMIN']) / 2
df_IND['daytimeTemp'] = ((df_IND['TMAX'] + df_IND['TMIN']) / 2 + df_IND['TMAX'])/2
df_NL['daytimeTemp'] = ((df_NL['TMAX'] + df_NL['TMIN']) / 2 + df_NL['TMAX'])/2
max_daytime_NL = df_NL[df_NL['daytimeTemp'] > 24]['daytimeTemp'].max()
df_NL[df_NL['daytimeTemp'] > 24].index[0] - datetime.strptime("2021-04-20", '%Y-%m-%d').date()
df_NL[df_NL['daytimeTemp'] == max_daytime_NL].index - datetime.strptime("2021-04-20", '%Y-%m-%d').date()
# %%
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
indicatation_text_x = 0
indicatation_text_y = 1.01
j_to_mj = 1000000
fig = plt.figure(figsize=(10, 10))

# Create a GridSpec for the whole figure
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.1, wspace=0.1, width_ratios=[6,4])

# First subplot
ax1 = fig.add_subplot(gs[0, 1])
ax1.plot(df_IND.index, df_IND['TMAX'], label='Maximum Temperature', color='r')
ax1.plot(df_IND.index, df_IND['daytimeTemp'], label='Daytime Mean Temperature', color='black')
ax1.plot(df_IND.index, df_IND['Tmean'] , label='Mean Temperature', color='grey')
ax1.plot(df_IND.index, df_IND['TMIN'], label='Minimum Temperature', color='g')
ax1.xaxis.set_major_locator(mdates.MonthLocator())
ax1.set_ylabel('Temperature (°C)')
# ax1.set_xticklabels('')

ax1.set_title('India')
ax1.text(indicatation_text_x, indicatation_text_y, 'b)', transform=ax1.transAxes, size=config.subplot_fs, weight='bold')

ax2 = fig.add_subplot(gs[0, 0], sharey=ax1)
ax2.plot(df_NL.index, df_NL['TMAX'], label='Maximum Temperature', color='r')
ax2.plot(df_NL.index, df_NL['daytimeTemp'], label='Daytime Mean Temperature', color='black')
ax2.plot(df_NL.index, df_NL['Tmean'] , label='Mean Temperature', color='grey')
ax2.plot(df_NL.index, df_NL['TMIN'], label='Minimum Temperature', color='g')
# ax2.set_xticklabels('')
ax2.set_ylabel('Temperature (°C)')
ax2.legend()
ax2.set_title('The Netherlands')
ax2.text(indicatation_text_x, indicatation_text_y, 'a)', transform=ax2.transAxes, size=config.subplot_fs, weight='bold')

plt.savefig(f'../output/weather_data.png', dpi = 300, bbox_inches='tight')
plt.savefig(f'../output/weather_data.svg', dpi = 600, bbox_inches='tight')
plt.show()
# %%




#%%
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
with open(f'{config.p_out_LSAsims}/hash_dict_final.json', 'r') as f:
    para_vals = json.load(f)
para.extend([] * config.LSA_sample_size)

keys = list(itertools.chain.from_iterable(itertools.repeat(x, config.LSA_sample_size) for x in para))
with open(f'{config.p_out}/LSA_NL/sims_NL_100/hash_dict_final.json', 'r') as f:
    para_vals_nl = json.load(f)

# %%

dfs = []  # List to store DataFrames

for i, key, value in zip(para_vals.keys(), keys, para_vals.values()):
    # print(i, value)
    with open(f'{config.p_out_LSAsims}/{i}_{key}.json', 'r') as f:
        results = json.load(f)
    df = pd.DataFrame(results)
    df['key'] = key
    df['value'] = value
    df['value'] = df['value'].astype(float)
    dfs.append(df)
df_NL = []
for i, key, value in zip(para_vals_nl.keys(), keys, para_vals_nl.values()):
    # print(i, value)
    with open(f'{config.p_out}/LSA_NL/sims_NL_100/{i}_{key}.json', 'r') as f:
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
key_fig6 = ['t1_pheno', 'TSUM1', 'SPAN', 'TDWI', 'te']
df_fig6_IND = large_df[large_df['key'].isin(key_fig6)].loc[:,['day','LAI','DVS','key','value']]
df_fig6_IND['DAP'] = np.tile(np.arange(no_ofdays), config.LSA_sample_size * len(key_fig6))
df_fig6_NL = large_df_NL[large_df_NL['key'].isin(key_fig6)].loc[:,['day','LAI','DVS','key','value']]
df_fig6_NL['DAP'] = np.tile(np.arange(no_ofdays_NL), config.LSA_sample_size * len(key_fig6))
df_fig6_IND['country'] = 'IND'
df_fig6_NL['country'] = 'NL'
df_fig6 = pd.concat([df_fig6_IND, df_fig6_NL])
df_fig6.set_index('DAP', inplace=True)
countries = ['NL', 'IND']
df_fig6['key'] = pd.Categorical(df_fig6['key'], key_fig6)
df_fig6['country'] = pd.Categorical(df_fig6['country'], categories=countries, ordered=True)

# %%

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
fig = plt.figure(figsize=(8, 12))
gs = gridspec.GridSpec(5, 3, width_ratios=[4, 4, 0.5], wspace=0.2, hspace=0.2)  # Adjust the width ratio for colorbars
labels = string.ascii_lowercase[:len(key_fig6) * len(countries)]
pointsize = 1
subplotlab_x = config.subplotlab_x
subplotlab_y = config.subplotlab_y

# Flatten the array of axes for easy iteration
axes = [fig.add_subplot(gs[i, j]) for i in range(5) for j in range(2)]
colorbar_axes = [fig.add_subplot(gs[i, 2]) for i in range(5)]  # For colorbars

# Iterate over parameters and countries to create subplots
for i, param in enumerate(key_fig6):
    for j, country in enumerate(countries):
        ax = axes[i * len(countries) + j]
        data = df_fig6[(df_fig6['key'] == param) & (df_fig6['country'] == country)]
        cmap = plt.get_cmap('viridis')
        norm = plt.Normalize(data['value'].min(), data['value'].max())
        sc = ax.scatter(x=data.index, y=data['LAI'], c=data['value'], cmap=cmap, norm=norm, s=pointsize)
        
        emergence_date = df_fig6[(df_fig6['DVS'] == 0) & (df_fig6['country'] == country)]['DVS'].drop_duplicates()
        tuberinitiation_date = df_fig6[(df_fig6['DVS'] == 1) & (df_fig6['country'] == country)]['DVS'].drop_duplicates()
        
        # Only set x-axis label for the last row
        if i == len(key_fig6) - 1:
            ax.set_xlabel('DAP')
        else:
            ax.set_xticklabels([])
        if i == 0:
            ax.axvline(emergence_date.index, color='green', linestyle='-')
            ax.axvline(tuberinitiation_date.index, color='green', linestyle='-')
        
        # Add subplot label
        subplot_label = labels[j * len(key_fig6) + i]
        ax.text(subplotlab_x, subplotlab_y - 0.1, subplot_label + ")", transform=ax.transAxes, size=config.subplot_fs, weight='bold')
        ax.set_ylim(0, 6)

# Add shared y-axis label
fig.text(0.05, 0.5, 'Leaf Area Index', va='center', rotation='vertical', fontsize=config.subplot_fs)

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
output_path = f'{config.p_out}/mainTextFigLSA'
plt.savefig(f'{output_path}.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()
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
