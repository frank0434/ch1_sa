# %% # try to add DVS into graph
import json
import itertools
import config
import pandas as pd
import os
import matplotlib.pyplot as plt
GSA_sample_size = config.GSA_sample_size
config.set_variables(GSA_sample_size)
para = config.params_of_interests
path_json = f'{config.p_out_sims}/hash_dict_{config.GSA_sample_size}'
# Get a list of all JSON files in the directory
json_files = [os.path.join(path_json, pos_json) for pos_json in os.listdir(path_json) if pos_json.endswith('.json')]

#%%
# Initialize an empty list to hold DataFrames
df_list = []

# Iterate over the JSON files
for index, js in enumerate(json_files):
    with open(os.path.join(path_json, js)) as json_file:
        data = pd.read_json(json_file, typ='series')

    # Convert the data to a DataFrame and append it to the list
    df_list.append(pd.DataFrame(data))
# %%
# Concatenate all the DataFrames in the list
para_vals = pd.concat(df_list, ignore_index=True)
# Assuming df is your DataFrame and 'column_name' is the column with the string
df = para_vals[0].str.split('_', expand=True)
# %%

df.columns = para
# %%
y = pd.read_json(f'{config.p_out_daysims}/day_105.json')
# %% 
df_y = y.T
df_y = df_y.apply(pd.to_numeric, errors='coerce')
df_y.drop('day', axis = 1 , inplace = True)
df = df.apply(pd.to_numeric, errors='coerce')
for output in ['DVS', 'LAI', 'TWSO']:

    fig, ax = plt.subplots(3, 5, figsize=(9, 10), sharey=True)
    ax = ax.flatten()
    for i, col in enumerate(para):
        ax[i].scatter(df[col], df_y[output], alpha=0.5, s=2)
        ax[i].set_title(col)
    fig.text(0.5, 0.06, 'Parameter value', ha='center', va='center')
    fig.text(0.08, 0.5, f'{output}', ha='center', va='center', rotation='vertical')
    plt.savefig(f'{config.p_out}/xy_pair_{output}_samplesize{GSA_sample_size}.png', dpi = 300, bbox_inches='tight')

    plt.show()
# %%
