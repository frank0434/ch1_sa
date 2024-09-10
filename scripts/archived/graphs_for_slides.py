# graph for EAPR conference slides
# session id = 703
# title = Efforts to model crop response to hot and dry environments
# Programme website and abstract booklet https://nibio.pameldingssystem.no/eapr2024#/program

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


# %%

# %% Produce a fig that shows the last day of TWSO for both conditions with bar chart 
# load NLD manually
with open(f'C:/Users/liu283/GitRepos/ch1_SA/output_NL/daySi_32768/Saltelli_160_TWSO.pkl', 'rb') as f:
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

# %%
