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
te['value'] = te['value'].round(5)
te['LAI'] = te['LAI'].round(5)
te['derivative'] = np.gradient(te['LAI'], np.diff(te['value'], prepend=te['value'].iloc[0]))
te['derivative_2nd'] = np.gradient(te['derivative'], np.diff(te['value'], prepend=te['value'].iloc[0]))
# Replace 0 with NaN
te = te.replace(0, np.nan)

# Remove rows with NaN
te = te.dropna()

# %%
import seaborn as sns
from matplotlib.cm import ScalarMappable

fig, ax = plt.subplots(2, 1, figsize=(10, 10))
# Create a scatter plot with a colorbar
plot = sns.scatterplot(data=te, x='day', y='derivative', hue='value', palette='viridis', ax=ax[0])
plot = sns.scatterplot(data=te, x='day', y='derivative_2nd', hue='value', palette='viridis', ax=ax[1])
plt.xticks(rotation=45)
plt.show()

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
te = large_df.loc[large_df['key'] == 'TSUM1', ['day', 'value', 'DVS']]

# %%
te['derivative'] = np.gradient(te['DVS'], np.diff(te['value'], prepend=te['value'].iloc[0]))

# Create a scatter plot with a colorbar
plot = sns.scatterplot(data=te, x='day', y='derivative', hue='value', palette='viridis')
plt.xticks(rotation=45)
plt.show()
# %%


# Group data by day
grouped = te.groupby("day")

# Define function to calculate derivative using central difference method
def calculate_derivative(df):
    derivative_values = []
    # for i in range(1, len(df) - 1):
        # dx = df.iloc[i + 1]["LAI"] - df.iloc[i - 1]["LAI"]
        # dy = df.iloc[i + 1]["value"] - df.iloc[i - 1]["value"]
    df['derivative'] =  np.gradient(df['LAI'], np.diff(df['value'], prepend=df['value'].iloc[0]))
        # derivative_values.append(derivative)
    # return derivative_values
# %% 
# # Calculate derivative for each group (day)
# derivative_results = {}
# for day, group_df in grouped:
#     derivative_results[day] = calculate_derivative(df = group_df)
# Calculate derivative for each group (day)
for name, group in grouped:
    calculate_derivative(group)
# Print derivative results
for day, derivatives in derivative_results.items():
    print("Day:", day)
    print("Derivative values:", derivatives)
