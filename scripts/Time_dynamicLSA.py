# %%
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
large_df['day'] = pd.to_datetime(large_df['day'])
large_df['value'] = large_df['value'].astype(float)
# %%
large_df.sort_values(by=['day','key', 'value'], inplace=True)

te = large_df.loc[large_df['key'] == 'te', ['day', 'value', 'LAI']]
te['value'] = te['value'].round(6)
te['LAI'] = te['LAI'].round(6)
te['derivative'] = np.gradient(te['LAI'], np.diff(te['value'], prepend=te['value'].iloc[0]))
te['derivative_2nd'] = np.gradient(te['derivative'], np.diff(te['value'], prepend=te['value'].iloc[0]))
# Replace 0 with NaN
te = te.replace(0, np.nan)

# Remove rows with NaN
te = te.dropna()

# %%


import seaborn as sns

from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 1, figsize=(10, 10))

# Create a line plot with a colorbar
plot = sns.lineplot(data=te, x='day', y='derivative', hue='value', palette='viridis', ax=ax[0])
plot = sns.lineplot(data=te, x='day', y='derivative_2nd', hue='value', palette='viridis', ax=ax[1])

plt.xticks(rotation=45)
plt.show()
# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_derivative(df, column, value_column):
    """
    Calculate the first and second derivatives of a column in a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame.
    column (str): The name of the column to calculate the derivatives of.
    value_column (str): The name of the column to use for the x values in the derivative calculation.

    Returns:
    pd.DataFrame: The DataFrame with the first and second derivatives added as new columns.
    """
    df = df.copy()
    df['derivative'] = np.gradient(df[column], np.diff(df[value_column], prepend=df[value_column].iloc[0]))
    df['derivative_2nd'] = np.gradient(df['derivative'], np.diff(df[value_column], prepend=df[value_column].iloc[0]))
    return df

def plot_data(df, x, y1, y2, hue, palette='viridis', col ='LAI', param ='te'):
    """
    Create two line plots in subplots.

    Parameters:
    df (pd.DataFrame): The DataFrame.
    x (str): The name of the column to use for the x values.
    y1 (str): The name of the column to use for the y values in the first plot.
    y2 (str): The name of the column to use for the y values in the second plot.
    hue (str): The name of the column to use for the hue values.
    palette (str): The name of the palette to use for the hue values.

    Returns:
    None
    """
    fig, ax = plt.subplots(2, 1, figsize=(10, 7))

    # Create the first line plot with a colorbar
    sns.scatterplot(data=df, x=x, y=y1, hue=hue, palette=palette, ax=ax[0])
    ax[0].set_xlabel('')  # Remove the x-axis label
    
    ax[0].set_xticklabels([])  # Rotate the x-axis tick labels
    ax[0].get_legend().remove()  
    # Create the second line plot with a colorbar
    sns.scatterplot(data=df, x=x, y=y2, hue=hue, palette=palette, ax=ax[1])
    plt.rcParams['axes.labelsize'] = 14  # Font size of the x and y labels
    plt.rcParams['xtick.labelsize'] = 12  # Font size of the tick labels on the x-axis
    plt.rcParams['ytick.labelsize'] = 12  # Font size of the tick labels on the y-axis
    plt.rcParams['legend.fontsize'] = 12  # Font size of the legend
    plt.legend(bbox_to_anchor=(1.0, 1.3), loc=2, borderaxespad=0.)

    plt.xticks(rotation=45)
    plt.savefig(f'{config.p_out_LSA}/{col}_{param}.png')
    plt.show()

# Use the functions
te = large_df.loc[large_df['key'] == 'te', ['day', 'value', 'LAI']]
te['value'] = te['value'].round(6)
te['LAI'] = te['LAI'].round(6)

# Calculate the derivatives
te = calculate_derivative(te, 'LAI', 'value')

# Replace 0 with NaN and remove rows with NaN
# te = te.replace(0, np.nan).dropna()

# Plot the data
plot_data(te, 'day', 'derivative', 'derivative_2nd', 'value', col='LAI', param='te')

# %%
import config



# Get the parameters and columns of interest from the config
params_of_interests = config.params_of_interests
cols_of_interests = ['DVS', 'LAI', 'TWSO']


# Iterate over the columns of interest
for col in cols_of_interests:
    # Iterate over the parameters of interest
    for param in params_of_interests:
        # Select the data for the current parameter and column of interest
        data = large_df.loc[large_df['key'] == param, ['day', 'value', col]]

        # Round the data to 6 decimal places
        data[col] = data[col].round(7)

        # Calculate the derivatives for the column of interest
        data = calculate_derivative(data, 'value', col)

        # Replace 0 with NaN and remove rows with NaN
        # data = data.replace(0, np.nan).dropna()

        # Plot the data for the column of interest
        plot_data(data, 'day', 'derivative', 'derivative_2nd', 'value', col=col, param=param)
# %%
