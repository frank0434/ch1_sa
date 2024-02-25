# %%
import pandas as pd
import numpy as np

import config
import pickle
import matplotlib.pyplot as plt
import psutil
import os
import multiprocessing as mp
import config
from tqdm import tqdm
from matplotlib.ticker import FuncFormatter
import run_vis_Si as vis
import pickle

# %%

def find_crossing_points(df):
    crossing_points = {}
    # Remove the first 10 rows to avoid the initialisation phase
    df2 =  df.iloc[10:].copy()  
    columns = df2.columns

    for i in range(len(columns)):
        for j in range(i+1, len(columns)):
            col1 = columns[i]
            col2 = columns[j]

            df2['diff'] = pd.to_numeric(df2[col1]) - pd.to_numeric(df2[col2])
            crossing_indices = np.where(np.diff(np.sign(df2['diff'])))[0]
            crossing_x_values = df2.index[crossing_indices].tolist()
            crossing_points[(col1, col2)] = crossing_x_values

    return crossing_points
# %%
# try to add a composition graph over time 
# %% 
col = 'DVS'
df_sensitivity_S1, df_sensitivity_ST = vis.process_files(col)
df_pawn_long = vis.create_dataframe_from_dict(vis.load_PAWN(col))
df_pawn_long = df_pawn_long[df_pawn_long['median'] > Dummy_si[1][1]]
df_pawn_median = df_pawn_long.loc[:, ["DAP","median", "names"]].pivot_table(index='DAP', columns='names', values='median').reset_index()
df_pawn_median.set_index('DAP',inplace =True)
df_pawn_median.rename_axis("index", axis='index', inplace=True)
# %%
# vis.plot_sensitivity_indices(df_sensitivity_S1, df_sensitivity_ST, df_pawn_median, col)
# %%
df_pawn_normal = vis.normalize_sensitivity(df_pawn_median)
df_s1_normal = vis.normalize_sensitivity(df_sensitivity_S1)
df_st_normal = vis.normalize_sensitivity(df_sensitivity_ST)
# %%
visualise = ['t1_pheno', 'TSUM1', 'TSUM2','TSUMEM','TEFFMX']
main_param_s1 = df_s1_normal.loc[:, visualise]
main_param_st = df_st_normal.loc[:, visualise]
main_pawn = df_pawn_normal.loc[:, visualise]

crossing_points_s1 = find_crossing_points(main_param_s1)
crossing_points_st = find_crossing_points(main_param_st)
crossing_points_pawn = find_crossing_points(main_pawn)
print(crossing_points_s1), print(crossing_points_st), print(crossing_points_pawn)
# Usage:
# %%

# %%
# df_pawn_normal = normalize_sensitivity(df_pawn_median)
# %%

df_normal.plot()
for x_values in selected_numbers:
    plt.axvline(x=x_values, color='r', linestyle='--')  # plot as red dashed vertical line

plt.show()