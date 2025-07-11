# %%
# Graphical abstract
import os
import pandas as pd
import matplotlib.pyplot as plt
import config
import utilities as ros

# --- Load and process data ---
base_path = "C:/Users/liu283/GitRepos/ch1_SA/"
col = "TWSO"
file = os.path.join(base_path, f"output_AUC_{col}.csv")
file_nl = os.path.join(base_path, f"output_NL_AUC_{col}.csv")

df_pawn_ros, df_st_ros = ros.process_AUC_file(file)
df_ros = pd.merge(df_st_ros, df_pawn_ros, how='inner', on=['variable', 'label', 'country'])
df_pawn_ros_NL, df_st_ros_NL = ros.process_AUC_file(file_nl)
df_ros_NL = pd.merge(df_st_ros_NL, df_pawn_ros_NL, how='inner', on=['variable', 'label', 'country'])

df_ros_sorted = df_ros.sort_values(by='ST', ascending=False).set_index('variable')
df_ros_sorted_NL = df_ros_NL.sort_values(by='ST', ascending=False).set_index('variable')
df_ros_sorted['country'] = 'India'
df_ros_sorted_NL['country'] = 'NL'

# --- Define display parameters and color assignment ---
display = ['te', 'TDWI', 'TSUM1', 't1_pheno', 'SPAN']
growth_params = ['te', 'TDWI']
import matplotlib.pyplot as plt

import cmcrameri.cm as cmc
batlow = cmc.batlow

def assign_colors(index, growth_params):
    # Pick two visually distinct colors from batlow
    color_growth = "#e41a1c"  # e.g., yellowish
    color_other = "#377eb8"   # e.g., blueish
    return [color_growth if param in growth_params else color_other for param in index]
# def assign_colors(index, growth_params):
    # return ['#009E73' if param in growth_params else '#0072B2' for param in index]

# Prepare DataFrames for plotting
def prepare_df(df, params, value_col):
    out = df.loc[params, [value_col, 'country']].copy()
    out['color'] = assign_colors(out.index, growth_params)
    return out

Sis_ST_NLD = prepare_df(df_ros_sorted_NL, display, 'ST')
Sis_PAWN_NLD = prepare_df(df_ros_sorted_NL, display, 'PAWN')
Sis_ST_IND = prepare_df(df_ros_sorted, display, 'ST')
Sis_PAWN_IND = prepare_df(df_ros_sorted, display, 'PAWN')

# --- Plotting ---
def plot_colored_barh(ax, df, value_col):
    # Plot each bar individually to assign a color per bar
    for i, (param, row) in enumerate(df.iterrows()):
        ax.barh(i, row[value_col], color=row['color'])
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([config.label_map.get(idx, idx) for idx in df.index])
    # ax.invert_yaxis()  # Optional: to match pandas' default order

# --- Plotting ---
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 5), sharex=True)
size_ticklabels = 24
size_striplables = 22
plot_colored_barh(axes[0, 0], Sis_ST_NLD, 'ST')

axes[0, 0].set_title('Temperate', fontsize=size_ticklabels)
plot_colored_barh(axes[1, 0], Sis_PAWN_NLD, 'PAWN')
plot_colored_barh(axes[0, 1], Sis_ST_IND, 'ST')
axes[0, 1].set_title('Subtropical', fontsize=size_ticklabels)
plot_colored_barh(axes[1, 1], Sis_PAWN_IND, 'PAWN')
# increase label size on first column 
for i, ax in enumerate(axes.flatten()):
    ax.tick_params(axis='y', labelsize=22)
# Set y-tick labels only for left column, and relabel using config.label_map
for i, ax in enumerate(axes.flatten()):
    ax.set_ylabel('')
    if i % 2 != 0:
        ax.set_yticklabels([])

fig.text(0.5, 0.02, 'Cumulative Sensitivity Indices', ha='center', fontsize=size_striplables)
fig.text(0.91, 0.3, 'PAWN', va='center', rotation=270, fontsize=size_striplables)
fig.text(0.91, 0.7, 'Sobol', va='center', rotation=270, fontsize=size_striplables)
plt.subplots_adjust(hspace=0.1, wspace=0.05)

plt.savefig('../manuscript/AUC_barh_plots.svg', bbox_inches='tight', transparent=True)
plt.savefig('../manuscript/AUC_barh_plots.png', dpi=600, bbox_inches='tight', transparent=True)
plt.show()
plt.close()
# %%
# Figure one 
#%%
import numpy as np
import matplotlib.pyplot as plt
import os
import config
from scipy.optimize import curve_fit
original_font_size = plt.rcParams['font.size']
wd = os.getcwd()
# %%

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

# %% # Figure 1 - DTSMTB
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
plt.xlabel('Daily mean air temperature (°C)', fontsize = config.subplot_fs)
plt.ylabel('Effective thermal time (°C day)', fontsize = config.subplot_fs)
plt.legend()
plt.savefig(f'{wd}/../manuscript/Fig2-i.svg')
plt.savefig(f'{wd}/../manuscript/Fig2-i.png', dpi = 600)
plt.show()
plt.close()
plt.rcParams['font.size'] = original_font_size

#%% # Figure 1 - TMPFTB
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
# plt.xlabel('Temperature (°C)')
# plt.ylabel('Reduction factor of \nMaximum leaf CO2 assimilation rate')

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
plt.xlabel('Daily daytime mean air temperature (°C)', fontsize=config.subplot_fs)
plt.ylabel('TMPFTB', fontsize=config.subplot_fs)
plt.legend()

plt.savefig(f'{wd}/../manuscript/Fig2-ii.svg')
plt.savefig(f'{wd}/../manuscript/Fig2-ii.png', dpi = 600)
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

plt.savefig(f'{wd}/output/weather_data.png', dpi = 600, bbox_inches='tight')
plt.savefig(f'{wd}/output/weather_data.svg', dpi = 600, bbox_inches='tight')
plt.show()


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

# Parameters to be colored (in the order they should receive colors)
main_params = [
    'TSUM1', 'TSUM2', 'SPAN', 'TBASEM', 'TSUMEM', 'TEFFMX', 't1_pheno', 'te_pheno'  # First 8 params
]

secondary_params = [
    'TDWI', 'RGRLAI', 'tm1', 't1', 't2', 'te', 'Q10'  # Last 7 params
]

# Create the name_color_map with automated color assignments
name_color_map = {
    'mean': 'blue',
    'median': 'red',
    'minimum': 'grey',
    'maximum': 'grey',
    'CV': 'green',  # Keep these standard colors
}

name_color_map = {
    'mean': 'blue',
    'median': 'red',
    'minimum': 'grey',
    'maximum': 'grey',
    'CV': 'black',  # Adjust colors as needed
    'TSUM1' : 'blue',
    'TSUM2' : '#6AE1C7',  # Light brown TO #6AE1C7
    'SPAN' :  '#BD0026',      # Light red
    'Q10' : 'purple',
    'TBASEM' : 'tan',
    'TSUMEM' : 'orange',
    'TEFFMX' : 'navy',
    'TDWI' : '#66FF33',  # Lime green
    'RGRLAI' : 'pink',
    'tm1' : 'cyan',
    't1' : 'magenta',
    't2' : '#66CC99',      # Medium aquamarine
    'te' : 'yellow',
    't1_pheno' : 'black',
    'te_pheno' : 'lightblue'
}
# %% visualise the color map 

for i, color in enumerate(name_color_map.values()):
    plt.axvspan(i, i + 1, color=color)
    plt.text(i + 0.5, 0.5, list(name_color_map.keys())[i], ha='center', va='center', fontsize=10, rotation=90)

plt.xlim(0, len(name_color_map))
plt.axis('off')  # Turn off axes
plt.show()

# After creating df2 and df3 but before plotting
# Reorder columns to group secondary_params together
def reorder_columns(df, secondary_params):
    # Create a list of column names with secondary_params first, then others
    secondary_cols = [col for col in df.columns if col in secondary_params]
    other_cols = [col for col in df.columns if col not in secondary_params]
    ordered_cols = secondary_cols + other_cols
    
    # Return the dataframe with reordered columns
    return df[ordered_cols]
# %%  Figure 3 DVS, LAI, TWSO in both NL and IND - manual run with modifing config.py
import utilities as ros

base_path = "C:/Users/liu283/GitRepos/ch1_SA/"
col = "DVS" 
# col = "LAI"
# col = "TWSO"
file = os.path.join(base_path, f"output_NL_AUC_{col}.csv")
# file = os.path.join(base_path, f"output_AUC_{col}.csv")
# 
df_pawn_ros, df_st_ros = ros.process_AUC_file(file)
df_ros = pd.merge(df_st_ros, df_pawn_ros, how='inner', on=['variable','label','country'])
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
# Reorder the columns in both dataframes
df2 = reorder_columns(df2, secondary_params)
df3 = reorder_columns(df3, secondary_params)

# Make sure the dataframes have numeric data types
df2 = df2.apply(pd.to_numeric, errors='coerce')
df3 = df3.apply(pd.to_numeric, errors='coerce')
hatch_patterns = ['.']

# colors_final = [config.name_color_map.get(col, 'black') for col in df2.columns]
colors2 = [name_color_map.get(col, 'black') for col in df2.columns]
colors3 = [name_color_map.get(col, 'black') for col in df3.columns]
# df2.plot.area(ax=axes[0],stacked=True, color=colors2, legend=False)
# df3.plot.area(ax=axes[1],stacked=True, color=colors3, legend=False)
# Plot df2 with hatching (Total order Sobol indices)
# Plot df2 with hatching (Total order Sobol indices)
alpha = 0.8  # Set the transparency level
for i, column in enumerate(df2.columns):
    y1 = df2.iloc[:, :i].sum(axis=1).values  # Convert to numpy array
    y2 = (y1 + df2[column].values)  # Convert to numpy array
    
    # Only apply hatch pattern if the column is in secondary_params
    hatch = hatch_patterns[i % len(hatch_patterns)] if column in secondary_params else None
    diff_colour = '#ffff66' if column in secondary_params else '#b301b3'
    axes[0].fill_between(
        df2.index, y1, y2, 
        color= colors2[i], 
        edgecolor='black',
        linewidth=0.5,
        hatch=hatch, 
        # alpha=alpha,  # Set the transparency level
        label=column
    )

# Plot df3 with hatching (PAWN indices)
for i, column in enumerate(df3.columns):
    y1 = df3.iloc[:, :i].sum(axis=1).values  # Convert to numpy array
    y2 = (y1 + df3[column].values)  # Convert to numpy array
    
    # Only apply hatch pattern if the column is in secondary_params
    hatch = hatch_patterns[i % len(hatch_patterns)] if column in secondary_params else None
    
    axes[1].fill_between(
        df3.index, y1, y2, 
        color=colors3[i], 
        edgecolor='black',
        linewidth=0.5,
        hatch=hatch, 
        label=column
    )

lines, labels = fig.axes[0].get_legend_handles_labels()
plt.ylim(0, 1.05)
plt.xlim(0, config.sim_period)
plt.xlabel('DAP', fontsize = config.subplot_fs)
# fig.text(0, 0.5, 'Proportion of Sensitivity indices', va='center', rotation='vertical', fontsize = config.subplot_fs)
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
plt.gca().invert_yaxis()
labels_final = [config.label_map.get(label, label) for label in labels]
# fig.legend(lines, labels_final, loc='center left', bbox_to_anchor=(1.0, 0.5), handlelength=1, borderpad=1, fontsize = 8)
labels_AUC = df_ros.variable.unique()

colors_AUC = [name_color_map.get(col, 'black') for col in labels_AUC]
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
if col == 'DVS' and config.run_NL_conditions:
    plt.savefig(f'../manuscript/Fig4-i.svg', bbox_inches='tight')
    plt.savefig(f'../manuscript/Fig4-i.png', dpi=600, bbox_inches='tight')
elif col == 'DVS' and not config.run_NL_conditions:
    plt.savefig(f'../manuscript/Fig4-ii.svg', bbox_inches='tight')
    plt.savefig(f'../manuscript/Fig4-ii.png', dpi=600, bbox_inches='tight')
elif col == 'LAI' and config.run_NL_conditions:
    plt.savefig(f'../manuscript/Fig5-i.svg', bbox_inches='tight')
    plt.savefig(f'../manuscript/Fig5-i.png', dpi=600, bbox_inches='tight')
elif col == 'LAI' and not config.run_NL_conditions:
    plt.savefig(f'../manuscript/Fig5-ii.svg', bbox_inches='tight')
    plt.savefig(f'../manuscript/Fig5-ii.png', dpi=600, bbox_inches='tight')
elif col == 'TWSO' and config.run_NL_conditions:
    plt.savefig(f'../manuscript/Fig6-i.svg', bbox_inches='tight')
    plt.savefig(f'../manuscript/Fig6-i.png', dpi=600, bbox_inches='tight')
elif col == 'TWSO' and not config.run_NL_conditions:
    plt.savefig(f'../manuscript/Fig6-ii.svg', bbox_inches='tight')
    plt.savefig(f'../manuscript/Fig6-ii.png', dpi=600, bbox_inches='tight')
plt.show()
plt.close()

# %% figures legends
import matplotlib.pyplot as plt

def create_legend_figure(params, colors, labels, output_path, ncol, dvs = False):
    fig, ax = plt.subplots(figsize=(12.5, 1), frameon=False)
    ax.axis('off')  # Hide the axes
    # get the colors for the secondary params
    colors = [name_color_map.get(col, 'black') for col in params]
    # Create a legend with colored lines
    if dvs:
        lines = [plt.Line2D([0], [0], color=c, linewidth=15, linestyle='-') for c in colors]
    else:
        lines = [plt.Line2D([0], [0], color=c, linewidth=15, linestyle='-', 
                        marker = '.' , markersize=1, markevery=0.2,
                         markerfacecolor='black', markeredgecolor='black' ) for c in colors]
    fig.legend(lines, labels, loc='center', ncol=ncol, handlelength=1, handleheight=2, borderpad=1,
               markerscale=3, handletextpad=1, columnspacing=1.5, fontsize=config.subplot_fs, frameon=False)
    plt.savefig(output_path, format='png', transparent=True)
    plt.savefig(output_path.replace('.png', '.svg'), format='svg', transparent=True)
    plt.show()
    plt.close()

# DVS legend
DVS_params = ['t1_pheno', 'TSUM1', 'TSUM2', 'TSUMEM', 'TBASEM', 'TEFFMX']
colors_DVS = [name_color_map.get(col, 'black') for col in DVS_params]
labels_DVS = [config.label_map.get(label, label) for label in DVS_params]
create_legend_figure(DVS_params, colors_DVS, labels_DVS, '../manuscript/Fig4_legend.png', ncol=6, dvs=True)

# # LAI legend
# LAI_params = config.params_of_interests
# colors_LAI = [name_color_map.get(col, 'black') for col in LAI_params]
# labels_LAI = [config.label_map.get(label, label) for label in LAI_params]
# create_legend_figure(LAI_params, colors_LAI, labels_LAI, '../output/legend_LAI.png', ncol=8)
# %%

# TWSO legend
TWSO_params = config.params_of_interests
colors_TWSO = [name_color_map.get(col, 'black') for col in TWSO_params]
labels_TWSO = [config.label_map.get(label, label) for label in TWSO_params]
# create_legend_figure(TWSO_params, colors_TWSO, labels_TWSO, '../output/legend_TWSO.png', ncol=8)
fig, ax = plt.subplots(figsize=(12.5, 1), frameon=False)
ax.axis('off')  # Hide the axes
    # get the colors for the secondary params
pattern_colors = [name_color_map.get(col, 'black') for col in secondary_params]
cool_colors = [name_color_map.get(col, 'black') for col in main_params]
    # Create a legend with colored lines
lines_warm = [plt.Line2D([0], [0], color=c, linewidth=15, linestyle='-', 
                        marker = '.' , markersize=3, markevery=0.2,
                         markerfacecolor='black', markeredgecolor='black' ) for c in pattern_colors]
lines_cool = [plt.Line2D([0], [0], color=c, linewidth=15, linestyle='-') for c in cool_colors]
labels = [config.label_map.get(label, label) for label in secondary_params + main_params]
lines = lines_warm + lines_cool
fig.legend(lines, labels, loc='center', ncol=8, handlelength=1, handleheight=2, borderpad=1,
               markerscale=3, handletextpad=1, columnspacing=1.5, fontsize=config.subplot_fs, frameon=False)
plt.savefig('../manuscript/Fig5-legend.png', format='png', transparent=True)
plt.savefig('../manuscript/Fig6-legend.png', format='png', transparent=True)
plt.savefig('../output/Fig5_legend.png'.replace('.png', '.svg'), format='svg', transparent=True)
plt.show()


##  LSA figure 6 ------------------------------
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
# fig.text(0.3, 0.04, 'The Netherlands', ha='center', fontsize=config.subplot_fs)
# fig.text(0.65, 0.04, 'India', ha='center', fontsize=config.subplot_fs)
units = ['°C', '°Cd', 'Days', r'DM kg ha$^{-1}$', '°C'] 

# Create colorbars on the right for each parameter
for i, (param, unit) in enumerate(zip(key_fig6, units)):
    ax_cbar = colorbar_axes[i]
    data = df_fig6[df_fig6['key'] == param]
    cmap = plt.get_cmap('viridis')
    min_val = data['value'].min()
    max_val = data['value'].max()
    norm = Normalize(min_val, max_val)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Only needed for older versions of Matplotlib
    # print(i, param, unit)

    labels = f'{config.label_map.get(param, param)} ({unit})'  # Combine the parameter name with its unit
    # combine the labels with units based on the order of the parameters
    
    cbar = fig.colorbar(sm, cax=ax_cbar, label=labels, orientation='vertical')
    cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))  # Ensure integer ticks
    cbar.ax.yaxis.set_major_formatter('{x:.0f}')  # Format ticks without decimals

# Save and show the plot
output_path = f'{wd}/../manuscript/Fig7'
plt.savefig(f'{output_path}.png', dpi=600, bbox_inches='tight', pad_inches=0.1)
plt.show()
plt.rcParams['font.size'] = original_font_size

# %% 
# Figure 7 illustration more on the te relative changes 
te_lai = data.loc[(data['key'] == 'te')&(data['country'] ==  'IND') &(data.index == 75), ['LAI', 'value']]
import seaborn as sns

sns.set_palette("pastel")
plt.figure(figsize=(10, 6))
sns.scatterplot(x='value', y='LAI', data=te_lai, hue = 'value')
te_lai['value'].min(), te_lai['value'].max()
plt.axhline(y=te_lai['LAI'].min(), color='r', linestyle='--', label='Min LAI')
plt.axhline(y=te_lai['LAI'].max(), color='g', linestyle='--', label='Max LAI')
te_lai['LAI'].max()/te_lai['LAI'].min()
# non-linear effect of te on LAI
# try other parameters
# %%
parameters = ['TDWI', 't1_pheno', 'TSUM1', 'SPAN', 'te']
for param in parameters:
    param_lai = df_fig6.loc[(df_fig6['key'] == param) & (df_fig6['country'] == 'IND') & (df_fig6.index == 75), ['LAI', 'value']]

    sns.set_palette("pastel")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='value', y='LAI', data=param_lai, hue='value')
    ratio = param_lai['LAI'].max() / param_lai['LAI'].min()
    plt.axhline(y=param_lai['LAI'].min(), color='r', linestyle='--', label='Min LAI')
    plt.axhline(y=param_lai['LAI'].max(), color='g', linestyle='--', label='Max LAI')
    plt.text(0.5, 0.5, f"Max/min LAI ratio: {ratio}", transform=plt.gca().transAxes)
    plt.title(f"Effect of {param} on LAI")
    plt.legend()
    plt.show()


# %% --------------------Figure 7
# Temp effect on tmpftb
# ---------------------
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import config
from pcse.fileinput import ExcelWeatherDataProvider
import config
# Define the data
temps = [0, 3, 10, 15, 24, 29, 36]
effeciency = [0.01, 0.01, 0.75, 1, 1, 0.75, 0.01]
tmpftb = pd.DataFrame({'Temperature': temps, 'TMPFTB': effeciency})


# load weather data
wdp = ExcelWeatherDataProvider('../data_raw/India2022_23.xlsx')
wdp_nl = ExcelWeatherDataProvider('../data_raw/350_weatherfile_2021.xlsx')
# %%
# Assuming weather is a DataFrame with the specified columns
weather = pd.DataFrame(wdp.export())
weather['Date'] = pd.to_datetime(weather['DAY'])
# weather= weather_nl[(weather_nl['Date'] >= pd.to_datetime("2021-04-20")) & (weather_nl['Date'] <= pd.to_datetime("2021-09-30"))]

weather['Temperature...C.'] = (((weather['TMIN'] + weather['TMAX']) / 2) + weather['TMAX'])/2
ws_temp = weather[['Date', 'Temperature...C.']]
weather_nl = pd.DataFrame(wdp_nl.export())

weather_nl['Date'] = pd.to_datetime(weather_nl['DAY'])
# Assuming df_NL is a DataFrame with a datetime index
weather_nl = weather_nl[(weather_nl['Date'] >= pd.to_datetime("2021-04-20")) & (weather_nl['Date'] <= pd.to_datetime("2021-09-30"))]
weather_nl['Temperature...C.'] = (((weather_nl['TMIN'] + weather_nl['TMAX']) / 2) + weather_nl['TMAX'])/2


# Interpolate
f = interpolate.interp1d(temps, effeciency, fill_value="extrapolate")

weather['TMPFTB'] = f(weather['Temperature...C.'])
weather_nl['TMPFTB'] = f(weather_nl['Temperature...C.'])
# Add DAP column
weather['DAP'] = np.arange(1, len(weather) + 1)
weather_nl['DAP'] = np.arange(1, len(weather_nl) + 1)
# %%
def plot_data(df, scenario):
    # plt.style.use('classic')  # Use grayscale style for black and white theme

    # Plot the data
    fig, ax1 = plt.subplots(facecolor='w')

    # color = 'tab:blue'
    ax1.set_xlabel('DAP', fontsize=config.subplot_fs)
    ax1.set_ylabel('TMPFTB', fontsize=config.subplot_fs)
    ax1.plot(df['DAP'], df['TMPFTB'], color='blue', linewidth=2)
    ax1.tick_params(axis='y')
    # ax1.set_facecolor('white')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    # color = 'tab:red'
    # we already handled the x-label with ax1
    ax2.set_ylabel('Temperature', color='red', fontsize=config.subplot_fs)  
    ax2.plot(df['DAP'], df['Temperature...C.'], color='red', linewidth=2)
    ax2.tick_params(axis='y')
    
    # Add horizontal lines
    ax2.axhline(15, color='green', linestyle='--')
    ax2.axhline(25, color='green', linestyle='--')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    ax1.set_ylim([0, 1.1])  # Set limits for first y-axis
    ax2.set_ylim(0, 35)  # Set limits for second y-axis
    ax1.set_xlim(0, )  # Set x-axis limits

    # ax1.set_xlabel('Day After Planting', fontsize = config.subplot_fs)
    if scenario == 'India':
        fig_number = 'Fig3-ii'
    else:
        fig_number = 'Fig3-i'
    plt.savefig(f'../manuscript/{fig_number}.svg', dpi = 600, bbox_inches='tight')
    plt.savefig(f'../manuscript/{fig_number}.png', dpi = 600, bbox_inches='tight')
    # plt.savefig(f'../output/weather_data.svg', dpi = 600, bbox_inches='tight')
    plt.show()

# Call the function with the dataframes
plot_data(weather, 'India')
plot_data(weather_nl, 'Netherlands')

# %%
# Calculate the number of days with temperature below 15 or above 24 in weather DataFrame
days_below_15_or_above_24_weather = weather[(weather['Temperature...C.'] < 15) | (weather['Temperature...C.'] > 24)].shape[0]

# Calculate the number of days with temperature below 15 or above 24 in weather_nl DataFrame
days_below_15_or_above_24_weather_nl = weather_nl[(weather_nl['Temperature...C.'] < 15) | (weather_nl['Temperature...C.'] > 24)].shape[0]

print(f"Number of days with temperature below 15 or above 24 in weather DataFrame: {days_below_15_or_above_24_weather}")
print(f"Number of days with temperature below 15 or above 24 in weather_nl DataFrame: {days_below_15_or_above_24_weather_nl}")

# Calculate the number of rows in the first half of the DataFrame
half_length = len(weather) // 2

# Slice the DataFrame to only include the first half of the rows
first_half_weather = weather.iloc[:half_length]

# Calculate the number of days with temperature below 15 or above 24 in the first half of the DataFrame
days_below_15_or_above_24_first_half_weather = first_half_weather[(first_half_weather['Temperature...C.'] < 15) | (first_half_weather['Temperature...C.'] > 24)].shape[0]
emergence = 8
print(f"Number of days with temperature below 15 or above 24 in the first half of weather DataFrame: {days_below_15_or_above_24_first_half_weather - emergence}")
first_half_weather_nl = weather_nl.iloc[:half_length]
days_below_15_or_above_24_first_half_weather_nl = first_half_weather_nl[(first_half_weather_nl['Temperature...C.'] < 15) | (first_half_weather_nl['Temperature...C.'] > 24)].shape[0]
emergence_nl = 24
print(f"Number of days with temperature below 15 or above 24 in the first half of weather_nl DataFrame: {days_below_15_or_above_24_first_half_weather_nl - emergence_nl}")
# %%
# FigS1.17 The last figure 
# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# %%
with open('../output/daysims_32768/day_105.json', 'r') as f:
    data = json.load(f)
# %%
day105 = pd.DataFrame(data).T
# %%
# Set a theme
sns.set_theme()

fig, ax = plt.subplots(3, 1, figsize=(6, 8))

# Define the column names and their corresponding titles
columns = ['DVS', 'LAI', 'TWSO']
titles = ['DVS', 'LAI', 'TWSO']

for i, (col, title) in enumerate(zip(columns, titles)):
    # Use seaborn for a nicer histogram
    sns.histplot(day105[col], bins=100, kde=True, ax=ax[i])
    
    # Set the title and labels
    ax[i].set_title(col, fontsize=14)
    ax[i].set_xlabel('Value', fontsize=12)
    ax[i].set_ylabel('Frequency', fontsize=12)

# Improve layout
fig.tight_layout()

# Save the figure
fig.savefig('../output/Distribution_105_TWSO.png', dpi=600)
fig.savefig('../output/Distribution_105_TWSO.svg')

plt.show()

# visualise the leaf life span changes 
# %%
import pandas as pd
threshold = 35
Tbase = 2
T = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
     16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
     26, 27, 28, 29, 30, 31,32 ,33 ,34 ,35 ,36 ,37 ,38 ,39 ,40]

def lvage(T, Tbase, threshold):
    if T < Tbase:
        return 0
    elif T > threshold:
        return 1
    else:
        return (T - Tbase) / (threshold - Tbase)
    
lvage_values = [lvage(t, Tbase, threshold) for t in T]

lvage_values = pd.DataFrame({'Temperature': T, 'lvage': lvage_values})
lvage_values.plot(x='Temperature', y='lvage', kind='line')