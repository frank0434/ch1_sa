# graph for EAPR conference slides
# session id = 703
# title = Efforts to model crop response to hot and dry environments
# Programme website and abstract booklet https://nibio.pameldingssystem.no/eapr2024#/program
# The code below is currently not working because path is one level lower than the main directory

# %%
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from datetime import datetime
import matplotlib.patches as mpatches
import config

config.set_variables(config.GSA_sample_size)
def load_data(day, output_var='TWSO', method='Saltelli'):
    # Load data
    with open(f'{config.p_out_daySi}/{method}_{day}_{output_var}.pkl', 'rb') as f:
        Si = pickle.load(f)
    return Si
# %%
def to_df(self):
    """Convert dict structure into Pandas DataFrame."""
    keys_to_include = ['S1', 'S1_conf', 'ST', 'ST_conf']
    return pd.DataFrame(
        {k: v for k, v in self.items() if k in keys_to_include}, index = config.params_of_interests
    )

# %% Produce a fig that shows the last day of TWSO for both conditions with bar chart 
# load NLD manually
with open(f'../output_NL/daySi_32768/Saltelli_160_TWSO.pkl', 'rb') as f:
    Si_NLD = pickle.load(f)
Si_df_NLD = to_df(Si_NLD['si_day_160_TWSO'])
df_sorted_NLD = Si_df_NLD.sort_values(by='ST', ascending=True)
order = df_sorted_NLD.index
conf_cols = df_sorted_NLD.columns.str.contains('_conf')
confs = df_sorted_NLD.loc[:, conf_cols]
confs.columns = [c.replace('_conf', "") for c in confs.columns]

display = ['t2', 'te', 'TDWI','Q10', 'TSUM1', 't1_pheno', 'SPAN']
Sis_NLD = df_sorted_NLD.loc[display, ['ST']]

# %%
# load indian 
Si_IND = load_data(105, 'TWSO', 'Saltelli')
# %%
Si_df_IND = to_df(Si_IND['si_day_105_TWSO'])

df_sorted_new = Si_df_IND.reindex(order)
conf_cols = df_sorted_new.columns.str.contains('_conf')
confs = df_sorted_new.loc[:, conf_cols]
confs.columns = [c.replace('_conf', "") for c in confs.columns]
Sis_IND = df_sorted_new.loc[display, ['ST']]
# Save the original default font size
original_font_size = plt.rcParams['font.size']
# Set the new default font size
plt.rcParams['font.size'] = 18
color = ['red', 'blue']
fig, axs = plt.subplots(1, 2, figsize=(7, 6), sharex=True)

plt.subplots_adjust(wspace=0.15, hspace=0.05)
# NLD
barplot = Sis_NLD.plot(kind='barh' , width = 0.9, ax=axs[0],
                   legend=False, color = color)
# indian
barplot = Sis_IND.plot(kind='barh' , width = 0.9, ax=axs[1],
                   legend=False, color = color)
                   
# Define the label mapping
label_map = {
    't2': '$T_{opt}$ for $A_{max}$',
    'te': '$T_{max}$ for $A_{max}$',
    'TDWI': 'Seed DW',
    'Q10': 'Q10',
    'TSUM1': 'TSUM1',
    't1_pheno': '$T_b$',
    'SPAN': 'Leaf Lifespan'
}

yticklabels =  [item.get_text() for item in axs[0].get_yticklabels()]
new_yticklabels = [label_map.get(label, label) for label in yticklabels]
axs[0].set_yticklabels(new_yticklabels)
axs[1].set_yticklabels([])
# barplot
plt.xlim(0, 1)
plt.show()
plt.rcParams['font.size'] = original_font_size

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
ax3.text(subplotlab_x, subplotlab_y, '$T_{phot-max}$', transform=ax3.transAxes, size=18, weight='bold')
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