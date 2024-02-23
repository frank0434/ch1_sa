#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sympy import plot


AGFUN_DTSMTB = [0.0,  0.0,
                2.0,  0.0,
               13.0, 11.0,
               30.0, 28.0]
x = AGFUN_DTSMTB[::2]  # Values in odd positions
y = AGFUN_DTSMTB[1::2]  # Values in even positions
plt.plot(x, y)
plt.xlabel('Temperature (°C)')
plt.ylabel('Daily increase in temperature sum \nas function of average temperature (°C day)')

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
plt.plot(x, y, 'o', label='Original data')
plt.plot(x_smooth, y_smooth, label='Fitted curve')
plt.legend()

plt.savefig('../output/DTSMTB_curving_fitting.png', dpi=600, format='png', bbox_inches='tight')
plt.show()
plt.close()

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
weather['Temp_daytime'] = (weather['TMIN'] + weather['TMAX'])/2 
weather['Temp_daytime'][0]
# %%
# Define the model inputs
problem = {
    'num_vars': 2,
    'names': [ 't1', 'te'],
    'bounds': [[2, 5], [25, 35]]
}

# Generate samples
param_values = saltelli.sample(problem, 512, calc_second_order=True)


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
len(day_values[0])
wide = pd.DataFrame(day_values).transpose()
wide.iloc[:, 1:10].plot()
# %%
plt.plot(param_values[:, 0])
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
day_Si[100]
# type(results)
# day_values = [[item[i] for item in results if item] for i in range(config.sim_period)]
# results
# results[116]
#%%
df_sensitivity_S1.plot()

# %%
df_sensitivity_S1 = pd.DataFrame(columns=[f"S1_{variable}" for variable in ['t1',  'te']],index=range(116))
df_sensitivity_ST = pd.DataFrame(columns=[f"ST_{variable}" for variable in ['t1',  'te']],index=range(116))

for i in range(116):
    # Check if the values are NaN
    df_sensitivity_S1.loc[i, :] = list(day_Si[i]['S1'])
    df_sensitivity_ST.loc[i, :] = list(day_Si[i]['ST'])

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharey=True)
df_sensitivity_S1.plot(ax=axes[0])
df_sensitivity_ST.plot(ax=axes[1])
plt.ylim(0, 1)

plt.tight_layout()

# %%
