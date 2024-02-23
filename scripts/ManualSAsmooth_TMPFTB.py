#%%
from re import S
import re
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
plt.plot(x, y, 'o', label='Original data')
plt.plot(x_smooth, y_smooth, label='Fitted curve')
plt.legend()

plt.savefig('../output/TMPFTB_curving_fitting.png', dpi=600, format='png', bbox_inches='tight')
plt.show()
plt.close()
# %%
AGFUN_TMPFTB = [0.0,0.01,3.0,0.01,10.0,0.75,15.0,1.0,24.0,1.0,29.0,0.75,36.0,0.01]
x = AGFUN_TMPFTB[::2]  # Values in odd positions
y = AGFUN_TMPFTB[1::2]  # Values in even positions
plt.plot(x, y)
plt.xlabel('Temperature (°C)')
plt.ylabel('Reduction factor of AMAX')

# Initial guess: [tm1, t1, t2, te, 1]
p0 = [0, 10, 20, 30]

# Fit the curve
popt, pcov = curve_fit(combined_equation, x, y, p0, method='dogbox', maxfev=5000)

# Generate y data from the combined_equation function with the optimized parameters
x_smooth = np.linspace(min(x), max(x), 100)
y_smooth = combined_equation(x_smooth, *popt)

# Plot the original data and the fitted curve
plt.plot(x, y, 'o', label='Original data')
plt.plot(x_smooth, y_smooth, label='Fitted curve')
plt.legend()
plt.show()
# %%
## have a look the optimized parameters and the covariance matrix
print(popt)
pcov
# Calculate standard deviation
perr = np.sqrt(np.diag(pcov))

# Plot standard deviation over optimized parameters
plt.figure(figsize=(10, 6))
plt.errorbar(range(len(popt)), popt, yerr=perr, fmt='o')
plt.title('Standard Deviation Over Optimized Parameters')
plt.xlabel('Parameter Index')
plt.ylabel('Parameter Value')
plt.show()
# %%
perr * 3
popt - perr * 3
popt + perr * 3

# two ways to moving forward: 1. use the optimised parameters with x time of the standard deviation; 2. use the optimised parameters with x time of the standard deviation
# 2. find potential values from the literature. literature values from Tamara's paper? but the papers only has optimal values 
# %% 
# try a GSA
from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.analyze import pawn
from SALib.test_functions import Ishigami
from functools import partial
import config
import pandas as pd

weather = pd.DataFrame(config.wdp.export())
weather['Temp_daytime'] = ((weather['TMIN'] + weather['TMAX'])/2 + weather['TMAX']) / 2
weather['Temp_daytime'][0]
# %%
# Define the model inputs
problem = {
    'num_vars': 4,
    'names': ['tm1', 't1', 't2', 'te'],
    'bounds': [[5, 10], [10, 20], [20, 24], [25, 40]]
}

# Generate samples
param_values = saltelli.sample(problem, config.GSA_sample_size, calc_second_order=True)


# %%
day_Si = {}
day_Si_pawn = {}
day_values = {}
for day in range(config.sim_period):
    Y = np.zeros(param_values.shape[0])
    for i, X in enumerate(param_values):
        Y[i] = combined_equation(weather['Temp_daytime'][day], *X)
        
    day_values[day] = Y   
    day_Si[day] = sobol.analyze(problem, Y, calc_second_order=True, print_to_console=False)
    day_Si_pawn[day] = pawn.analyze(problem, param_values, Y, seed = 42)
# %%
# Perform analysis
len(day_values[0])
day_values[0]
param_values[0]
param_values[1]
param_values[200]
param_values[300]
day_Si_pawn[100]

# %%
dfs = []
for day in day_Si_pawn:
    df = pd.DataFrame(day_Si_pawn[day])
    df['day'] = day
    dfs.append(df)
df_pawn = pd.concat(dfs)
# %%
name_color_map = {
        'mean': 'blue',
        'median': 'red',
        'minimum': 'grey',
        'maximum': 'grey',
        'CV': 'green',  # Adjust colors as needed
        'tm1' : 'cyan',
        't1' : 'magenta',
        't2' : 'lime',
        'te' : 'indigo',
    }
fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize = (12, 6))
    
    # Extract unique names from the 'names' column
unique_names = df_pawn['names'].unique()

# Plot each unique name with its corresponding color
# for col, ax in zip(col, axes.flatten()):
    # output_df = df_pawn[df_pawn['Output'] == col]
for name in unique_names:
    name_data = df_pawn[df_pawn['names'] == name]
    color = name_color_map.get(name, 'black')  # Default to black if not found in the map
    axes.plot(name_data['day'], name_data['median'], label=name, color=color)
    # Add area plots for 'minimum' and 'maximum'
    axes.fill_between(name_data['day'], name_data['maximum'], name_data['minimum'], color=color, alpha=0.3)
    # axes.set_title(col)
fig.suptitle('PWAN median, maximum and minimum for nine parameters and eight output over time')
fig.text(0.5, 0.06, 'Day after planting', ha='center', va='center')
fig.text(0.08, 0.5, 'Sensitivity indices', ha='center', va='center', rotation='vertical')
plt.legend()
plt.subplots_adjust(hspace=0.2, wspace=0)

# %%
weather.plot(y=['Temp_daytime'])
# %% are they added up to 1?
day_Si[0]['S1'].sum()
day_Si[0]['ST'].sum()
day_Si[1]['S1'].sum()
df_pawn.loc[:, ['day', 'mean']].groupby('day').sum()

df_pawn.groupby('day').sum().plot()
# %%
df_sensitivity_S1['row_SUM'] = df_sensitivity_S1.sum(axis=1)
df_sensitivity_S1
df_sensitivity_ST['row_SUM'] = df_sensitivity_ST.sum(axis=1)
df_sensitivity_ST
# %%
day_Si[0]
# type(results)
# day_values = [[item[i] for item in results if item] for i in range(config.sim_period)]
# results
# results[116]
# %%
df_sensitivity_S1 = pd.DataFrame(columns=[f"S1_{variable}" for variable in ['tm1', 't1', 't2', 'te']],index=range(116))
df_sensitivity_ST = pd.DataFrame(columns=[f"ST_{variable}" for variable in ['tm1', 't1', 't2', 'te']],index=range(116))

for i in range(116):
    # Check if the values are NaN
    df_sensitivity_S1.loc[i, :] = list(day_Si[i]['S1'])
    df_sensitivity_ST.loc[i, :] = list(day_Si[i]['ST'])

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharey=True)
df_sensitivity_S1.plot(ax=axes[0])
df_sensitivity_ST.plot(ax=axes[1])
plt.ylim(0, 1)

plt.tight_layout()
