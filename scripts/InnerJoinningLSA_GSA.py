# %%
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
import config
import pickle
import json
import itertools
import numpy as np
import config
with open('DummySi_results.pkl', 'rb') as f:
    Dummy_si = pickle.load(f)
planting = "2022-11-10"
harvest = ['2022-12-19', '2023-01-16', '2023-02-24']
from pcse.fileinput import ExcelWeatherDataProvider
# load weather data
wdp = ExcelWeatherDataProvider(config.Weather_real)
# %%
weather_df = pd.DataFrame(wdp.export() )
weather_df = weather_df.loc[:, ['DAY','TMAX', 'TMIN','RAIN','IRRAD']]
weather_df.plot(x='DAY', subplots=True, layout=(2,2), figsize=(15,10))
weather_df.describe()
weather_df['IRRAD'] = weather_df['IRRAD']/100

weather_df['IRRAD'].cumsum()
# %%
def calculate_days_difference(planting, harvest):
    # Convert planting date to datetime
    planting_date = datetime.strptime(planting, "%Y-%m-%d")
    # Convert harvest dates to datetime and calculate differences
    harvest_dates = [datetime.strptime(date, "%Y-%m-%d") for date in harvest]
    differences = [(date - planting_date).days for date in harvest_dates]
    # Subtract 1 from each value in differences
    differences = [difference - 1 for difference in differences]
    # differences.append(105)
    return differences
# %%
# Calculate days difference
differences = calculate_days_difference(planting, harvest)
# %% 
# Load the results of the LSA
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

colors = config.name_color_map
colors

# %%
DAPs = np.tile(np.arange(107), config.LSA_sample_size * len(para))
large_df['DAP'] = DAPs[:len(large_df)]
final_output = large_df[large_df['day'] == '2023-02-24'].drop(columns=['DAP','day','WWLOW','RD','SM'])
final_output.loc[:, 'value'] = final_output['value'].astype(float)

final_long = final_output.melt(id_vars = ['key', 'value'],  value_name='vals')
final_long
# Define a function that will sort the values in each group
def sort_group(group):
    return group.sort_values(by='value', ascending=True)

# Group the DataFrame by 'group', apply the function, and reset the index
sorted_df = final_long.groupby(['key', 'variable']).apply(sort_group).reset_index(drop=True)

# sorted_df.plot(x='value', y='vals', kind='scatter', c=sorted_df['key'].map(colors), cmap='viridis')

sorted_df.groupby(['key', 'variable'])
# Define a function that will calculate the first derivative of 'vals' with respect to 'value' in each group
def calculate_second_derivative(group):
    group['first_derivative'] = np.gradient(group['vals'], group['value'])
    group['second_derivative'] = np.gradient(group['first_derivative'], group['value'])
    return group

# Group the DataFrame by 'group', apply the function, and reset the index
df = sorted_df.groupby(['key', 'variable']).apply(calculate_second_derivative).reset_index(drop=True)

print(df)
# %%
df[~(df['second_derivative'] == 0)]['key'].unique()

# make a matrix 
pre_m = df.loc[:,['key', 'variable', 'second_derivative']].drop_duplicates()
pre_m = pre_m.groupby(['key', 'variable']).apply(lambda second_derivative: np.mean(np.abs(second_derivative))).reset_index()
pre_m['effects'] = np.where(abs(pre_m[0]) > 0.0001, 1, 0)
pre_m = pre_m.pivot(index='key', columns='variable', values='effects')
pre_m = pre_m.fillna(0).sort_values(by=['DVS', 'LAI', 'TWSO'], ascending=False)

# %%
def load_data(day, output_var='TWSO', method='Saltelli'):
    # Load data
    with open(f'{config.p_out_daySi}/{method}_{day}_{output_var}.pkl', 'rb') as f:
        Si = pickle.load(f)
    return Si

def to_df(self):
    """Convert dict structure into Pandas DataFrame."""
    keys_to_include = ['S1', 'S1_conf', 'ST', 'ST_conf']
    return pd.DataFrame(
        {k: v for k, v in self.items() if k in keys_to_include}, index = config.params_of_interests
    )

# %% 
# Start with DVS IN SALTELLI
differences = calculate_days_difference(planting, harvest)
samplesize = 32768
config.set_variables(samplesize, local=True)
cols = len(differences)
def find_common_indices(d, var):
    """
    Find the common indices between LSA, Saltelli, and PAWN results.

    Parameters:
    d (int): The day for which to find the common indices.
    var (str): The variable for which to find the common indices.

    Returns:
    DataFrame: The common indices and their sources.
    """
    Si = load_data(d, var, 'Saltelli')
    df = to_df(Si[f'si_day_{d}_{var}'])
    data_PAWN = load_data(d, var, 'PAWN')
    df_PAWN = data_PAWN[f'si_day_{d}_{var}'].to_df()
    index_lsa = pre_m[pre_m['DVS'] == 1].index
    index_lsa.rename("param", inplace=True)
    index_Saltelli = df[df['ST']> 0.05 ].index

    index_pawn = df_PAWN[df_PAWN['median']>Dummy_si[1][1]].index
    df_lsa = pd.DataFrame(index_lsa, columns=['param'])
    df_lsa['source'] = 'LSA'
    df_Saltelli = pd.DataFrame(index_Saltelli, columns=['param'])
    df_Saltelli['source'] = 'Saltelli'
    df_pawn = pd.DataFrame(index_pawn, columns=['param'])
    df_pawn['source'] = 'PAWN'
    df_all = pd.concat([df_lsa, df_Saltelli, df_pawn])
    df_all = df_all.drop_duplicates().pivot(index='param', columns='source', values='param')
    df_all['day'] = d
    df_all['var'] = var
    df_all
    return df_all
# %%
d = 105
var = 'LAI'

# %%
# List of variables and days to iterate over
variables = ['DVS', 'LAI', 'TWSO']  # Replace with your actual variables

# Initialize an empty list to store the DataFrames
df_list = []

# Iterate over variables and days
for var in variables:
    for d in differences:
        common_indices = find_common_indices(d, var)

        # Add the current DataFrame to the list
        df_list.append(common_indices)

# Concatenate all the DataFrames in the list
common_indices_df = pd.concat(df_list, ignore_index=True)

# Print the DataFrame
table = common_indices_df.sort_values(by = ['day', 'var']).dropna()
table_wide = table.melt(id_vars = ['day', 'var'], value_name='param').pivot(index = ['day', 'var','source'], columns='param', values='param').reset_index()
table_wide.iloc[:, -4:].applymap(lambda x: 1 if pd.notnull(x) else 0)
# Apply the function to the last four columns and assign the result back to those columns
table_wide.iloc[:, -4:] = table_wide.iloc[:, -4:].applymap(lambda x: 1 if pd.notnull(x) else 0)


table_wide



# %%
def find_common_indices_GSA(d, var):
    """
    Find the common indices between LSA, Saltelli, and PAWN results.

    Parameters:
    d (int): The day for which to find the common indices.
    var (str): The variable for which to find the common indices.

    Returns:
    DataFrame: The common indices and their sources.
    """
    Si = load_data(d, var, 'Saltelli')
    df = to_df(Si[f'si_day_{d}_{var}'])
    data_PAWN = load_data(d, var, 'PAWN')
    df_PAWN = data_PAWN[f'si_day_{d}_{var}'].to_df()
    index_Saltelli = df[df['ST']> 0.05 ].index

    index_pawn = df_PAWN[df_PAWN['median']>Dummy_si[1][1]].index
    df_Saltelli = pd.DataFrame(index_Saltelli, columns=['param'])
    df_Saltelli['source'] = 'Saltelli'
    df_pawn = pd.DataFrame(index_pawn, columns=['param'])
    df_pawn['source'] = 'PAWN'
    df_all = pd.concat([df_Saltelli, df_pawn])
    df_all = df_all.drop_duplicates().pivot(index='param', columns='source', values='param')
    df_all['day'] = d
    df_all['var'] = var
    df_all
    return df_all
# List of variables and days to iterate over
variables = ['DVS', 'LAI', 'TWSO']  # Replace with your actual variables

# Initialize an empty list to store the DataFrames
df_list = []

# Iterate over variables and days
for var in variables:
    for d in differences:
        common_indices = find_common_indices_GSA(d, var)

        # Add the current DataFrame to the list
        df_list.append(common_indices)

# Concatenate all the DataFrames in the list
common_indices_df = pd.concat(df_list, ignore_index=True)

# Print the DataFrame
table = common_indices_df.sort_values(by = ['day', 'var']).dropna()
table_wide = table.melt(id_vars = ['day', 'var'], value_name='param').pivot(index = ['day', 'var','source'], columns='param', values='param').reset_index()
table_wide.iloc[:, -4:].applymap(lambda x: 1 if pd.notnull(x) else 0)
# Apply the function to the last four columns and assign the result back to those columns
table_wide.iloc[:, 3:] = table_wide.iloc[:, 3:].applymap(lambda x: 1 if pd.notnull(x) else 0)


table_wide

