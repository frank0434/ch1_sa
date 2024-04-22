# %%
import pandas as pd
import numpy as np
import config
import pickle
import matplotlib.pyplot as plt
import config
import run_vis_GSA as vis
import pickle
from sklearn.cluster import DBSCAN
with open('DummySi_results.pkl', 'rb') as f:
    Dummy_si = pickle.load(f)

# %%
    # , cols = ['t1_pheno', 'TSUM1', 'TSUM2','TSUMEM','TEFFMX']
def process_dataframe(df):
    df_normal = vis.normalize_sensitivity(df)
    main_param = df_normal
    # .loc[:, cols]
    crossing_points = find_crossing_points(main_param)
    selected_numbers = process_crossing_points(crossing_points)
    return main_param, selected_numbers, crossing_points

def find_crossing_points(df):
    crossing_points = {}
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

def process_crossing_points(crossing_points):
    points = [item for sublist in crossing_points.values() for item in sublist]
    X = np.array(points).reshape(-1, 1)
    model = DBSCAN(eps=3, min_samples=2)
    model.fit(X)
    labels = model.labels_
    selected_numbers = [int(X[np.where(model.labels_ == label)[0]][0]) for label in labels]
    return selected_numbers


# %% 
# I want to know all the crossing points
col = 'DVS'
# key parameters are manually selected from the area graphs
if config.run_NL_conditions == True:
    key_paras =  ['t1_pheno', 'TSUM1', 'TSUM2','TSUMEM','TBASEM'] 
else:
    key_paras = ['t1_pheno', 'TSUM1', 'TSUM2','TSUMEM','TEFFMX']

df_sensitivity_S1, df_sensitivity_ST = vis.process_files(col)
df_sensitivity_S1_normal = vis.normalize_sensitivity(df_sensitivity_S1)
df_sensitivity_ST_normal = vis.normalize_sensitivity(df_sensitivity_ST)
df_pawn_long = vis.create_dataframe_from_dict(vis.load_PAWN(col))
df_pawn_long = df_pawn_long[df_pawn_long['median'] > Dummy_si[1][1]]
df_pawn_median = df_pawn_long.loc[:, ["DAP","median", "names"]].pivot_table(index='DAP', columns='names', values='median').reset_index()
df_pawn_median.set_index('DAP',inplace =True)
df_pawn_median.rename_axis("index", axis='index', inplace=True)

df_pawn_median_normal = vis.normalize_sensitivity(df_pawn_median)

# %%
_, _ , crossing_s1 = process_dataframe(df_sensitivity_S1_normal.loc[:, key_paras])
_, _ , crossing_st = process_dataframe(df_sensitivity_ST_normal.loc[:, key_paras])
_, _ , crossing_pawn = process_dataframe(df_pawn_median_normal.loc[:, key_paras])

crossing_s1 = {k: v for k, v in crossing_s1.items() if v}
crossing_st = {k: v for k, v in crossing_st.items() if v}
crossing_pawn = {k: v for k, v in crossing_pawn.items() if v}
df_s1 = pd.DataFrame(list(crossing_s1.items()), columns=['Keys', 'Values'])
df_st = pd.DataFrame(list(crossing_st.items()), columns=['Keys', 'Values'])
df_pawn = pd.DataFrame(list(crossing_pawn.items()), columns=['Keys', 'Values'])


# %%
df_merged = df_s1.merge(df_st, on='Keys', how='outer').merge(df_pawn, on='Keys', how='outer')
df_merged.rename(columns={'Values_x': 'Values_S1', 'Values_y': 'Values_ST', 'Values': 'Values_PAWN'}, inplace=True)

print(df_merged)
# Output the DataFrame to LaTeX
# Output the DataFrame to LaTeX
latex_output = df_merged.to_latex(index=False)

# Add the necessary LaTeX commands
latex_output = '\\begin{table}[]\n\\centering\n' + latex_output
latex_output += '\n\\caption{Caption}\n\\label{tab:my_label}\n\\end{table}'

# Write the LaTeX output to a file
with open(f'df_merged_{col}.tex', 'w') as f:
    f.write(latex_output)
# %%
# examine the crossing points
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# Plot the normalized dataframes and the crossing points as vertical lines
for ax, df, crossing in zip(axs, [df_sensitivity_S1_normal.loc[: , key_paras],
                                  df_sensitivity_ST_normal.loc[: , key_paras],
                                  df_pawn_median_normal.loc[: , key_paras]], 
                                  [crossing_s1, crossing_st, crossing_pawn]):
    df.plot(ax=ax)
    for key, values in crossing.items():
        for value in values:
            ax.vlines(x=value, ymin=0, ymax=1, colors='r')
            ax.text(x=value, y=0.5, s=str(value), color='r', ha='right')

plt.tight_layout()
plt.show()
# %% # do the area under the curve 
from scipy import integrate

# Calculate the area under the curve for each column in df_sensitivity_S1_normal
areas_pawn = df_pawn_median_normal.fillna(0).apply(integrate.trapz).sort_values(ascending=False)
areas_ST = df_sensitivity_ST_normal.fillna(0).apply(integrate.trapz).sort_values(ascending=False)

combined_areas = pd.concat([areas_pawn, areas_ST], axis=1, join='outer', keys=['PAWN', 'ST'])

combined_areas.round().to_csv(f'{config.p_out}_AUC_{col}.csv')
# combined_areas


# %%
# bring back the crossing points clustering
clusters_s1 = process_crossing_points(crossing_s1)
clusters_st = process_crossing_points(crossing_st)
clusters_pawn = process_crossing_points(crossing_pawn)
# %%
# examine the crossing points
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# Plot the normalized dataframes and the crossing points as vertical lines
for ax, df, crossing in zip(axs, [df_sensitivity_S1_normal, df_sensitivity_ST_normal, df_pawn_median_normal], [clusters_s1, clusters_st, clusters_pawn]):
    df.plot(ax=ax)
    values = list(set(crossing))
    for value in values:
        ax.vlines(x=value, ymin=0, ymax=1, colors='r')
        ax.text(x=value, y=0.5, s=str(value), color='r', ha='right')


plt.tight_layout()
fig.suptitle('Crossing points after clustering - could be used as a supplementary figure in the manuscript')

plt.savefig(f'{config.p_out}/{col}_Si_crossing_points_afterClustering.png', dpi = 300, bbox_inches = 'tight')
plt.show()
# %%
col = 'LAI'
# key parameters are manually selected from the area graphs
if config.run_NL_conditions == True:
    key_paras = ['t1_pheno', 'SPAN', 'TSUM1','TSUM2', 'TDWI', 'TSUMEM', 'TBASEM']
else:
    key_paras = ['SPAN', 'te',  'TDWI']

df_sensitivity_S1, df_sensitivity_ST = vis.process_files(col)
df_sensitivity_S1_normal = vis.normalize_sensitivity(df_sensitivity_S1)
df_sensitivity_ST_normal = vis.normalize_sensitivity(df_sensitivity_ST)
df_pawn_long = vis.create_dataframe_from_dict(vis.load_PAWN(col))
df_pawn_long = df_pawn_long[df_pawn_long['median'] > Dummy_si[1][1]]
df_pawn_median = df_pawn_long.loc[:, ["DAP","median", "names"]].pivot_table(index='DAP', columns='names', values='median').reset_index()
df_pawn_median.set_index('DAP',inplace =True)
df_pawn_median.rename_axis("index", axis='index', inplace=True)

df_pawn_median_normal = vis.normalize_sensitivity(df_pawn_median)

# %%
_, _ , crossing_s1 = process_dataframe(df_sensitivity_S1_normal.loc[:, key_paras])
_, _ , crossing_st = process_dataframe(df_sensitivity_ST_normal.loc[:, key_paras])
_, _ , crossing_pawn = process_dataframe(df_pawn_median_normal.loc[:, key_paras])

crossing_s1 = {k: v for k, v in crossing_s1.items() if v}
crossing_st = {k: v for k, v in crossing_st.items() if v}
crossing_pawn = {k: v for k, v in crossing_pawn.items() if v}
df_s1 = pd.DataFrame(list(crossing_s1.items()), columns=['Keys', 'Values'])
df_st = pd.DataFrame(list(crossing_st.items()), columns=['Keys', 'Values'])
df_pawn = pd.DataFrame(list(crossing_pawn.items()), columns=['Keys', 'Values'])


# %%
df_merged = df_s1.merge(df_st, on='Keys', how='outer').merge(df_pawn, on='Keys', how='outer')
df_merged.rename(columns={'Values_x': 'Values_S1', 'Values_y': 'Values_ST', 'Values': 'Values_PAWN'}, inplace=True)

print(df_merged)
# Output the DataFrame to LaTeX
# Output the DataFrame to LaTeX
latex_output = df_merged.to_latex(index=False)

# Add the necessary LaTeX commands
latex_output = '\\begin{table}[]\n\\centering\n' + latex_output
latex_output += '\n\\caption{Caption}\n\\label{tab:my_label}\n\\end{table}'

# Write the LaTeX output to a file
with open(f'df_merged_{col}.tex', 'w') as f:
    f.write(latex_output)
# %%
# examine the crossing points
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# Plot the normalized dataframes and the crossing points as vertical lines
for ax, df, crossing in zip(axs, [df_sensitivity_S1_normal, df_sensitivity_ST_normal, df_pawn_median_normal], [crossing_s1, crossing_st, crossing_pawn]):
    df.plot(ax=ax)
    for key, values in crossing.items():
        for value in values:
            ax.vlines(x=value, ymin=0, ymax=1, colors='r')
            ax.hlines(y = 0.1, xmin = 0, xmax = 100, colors = 'b')
            ax.text(x=value, y=0.5, s=str(value), color='r', ha='right')


plt.tight_layout()
plt.show()
# %%
from scipy import integrate

# Calculate the area under the curve for each column in df_sensitivity_S1_normal
areas_pawn = df_pawn_median_normal.fillna(0).apply(integrate.trapz).sort_values(ascending=False)
areas_ST = df_sensitivity_ST_normal.fillna(0).apply(integrate.trapz).sort_values(ascending=False)
combined_areas = pd.concat([areas_pawn, areas_ST], axis=1, join='outer', keys=['PAWN', 'ST'])

combined_areas.round().to_csv(f'{config.p_out}_AUC_{col}.csv')
print("Areas under the curve for each column in df_sensitivity_S1_normal:", areas_pawn)
print("Areas under the curve for each column in df_sensitivity_ST_normal:", areas_ST)
# %%
# bring back the crossing points clustering
clusters_s1 = process_crossing_points(crossing_s1)
clusters_st = process_crossing_points(crossing_st)
clusters_pawn = process_crossing_points(crossing_pawn)
# %%
# examine the crossing points
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# Plot the normalized dataframes and the crossing points as vertical lines
for ax, df, crossing in zip(axs, [df_sensitivity_S1_normal, df_sensitivity_ST_normal, df_pawn_median_normal], [clusters_s1, clusters_st, clusters_pawn]):
    df.plot(ax=ax)
    values = list(set(crossing))
    for value in values:
        ax.vlines(x=value, ymin=0, ymax=1, colors='r')
        ax.text(x=value, y=0.5, s=str(value), color='r', ha='right')


plt.tight_layout()
fig.suptitle('Crossing points after clustering - could be used as a supplementary figure in the manuscript')
plt.show()
plt.savefig(f'{config.p_out}/{col}_Si_crossing_points_afterClustering.png', dpi = 300, bbox_inches = 'tight')


# %%
col = 'TWSO'
# key parameters are manually selected from the area graphs
if config.run_NL_conditions == True:
    key_paras = ['t1_pheno', 'TSUM1', 'TSUM2','TSUMEM','TBASEM', 'Q10']
else:
    key_paras = ['SPAN', 'te', 't1_pheno', 'TSUM1', 'TDWI', 'TSUMEM', 't2']

df_sensitivity_S1, df_sensitivity_ST = vis.process_files(col)
df_sensitivity_S1_normal = vis.normalize_sensitivity(df_sensitivity_S1)
df_sensitivity_ST_normal = vis.normalize_sensitivity(df_sensitivity_ST)
df_pawn_long = vis.create_dataframe_from_dict(vis.load_PAWN(col))
df_pawn_long = df_pawn_long[df_pawn_long['median'] > Dummy_si[1][1]]
df_pawn_median = df_pawn_long.loc[:, ["DAP","median", "names"]].pivot_table(index='DAP', columns='names', values='median').reset_index()
df_pawn_median.set_index('DAP',inplace =True)
df_pawn_median.rename_axis("index", axis='index', inplace=True)

df_pawn_median_normal = vis.normalize_sensitivity(df_pawn_median)

# %%
_, _ , crossing_s1 = process_dataframe(df_sensitivity_S1_normal.loc[:, key_paras])
_, _ , crossing_st = process_dataframe(df_sensitivity_ST_normal.loc[:, key_paras])
_, _ , crossing_pawn = process_dataframe(df_pawn_median_normal.loc[:, key_paras])

crossing_s1 = {k: v for k, v in crossing_s1.items() if v}
crossing_st = {k: v for k, v in crossing_st.items() if v}
crossing_pawn = {k: v for k, v in crossing_pawn.items() if v}
df_s1 = pd.DataFrame(list(crossing_s1.items()), columns=['Keys', 'Values'])
df_st = pd.DataFrame(list(crossing_st.items()), columns=['Keys', 'Values'])
df_pawn = pd.DataFrame(list(crossing_pawn.items()), columns=['Keys', 'Values'])


# %%
df_merged = df_s1.merge(df_st, on='Keys', how='outer').merge(df_pawn, on='Keys', how='outer')
df_merged.rename(columns={'Values_x': 'Values_S1', 'Values_y': 'Values_ST', 'Values': 'Values_PAWN'}, inplace=True)

print(df_merged)
# Output the DataFrame to LaTeX
# Output the DataFrame to LaTeX
latex_output = df_merged.to_latex(index=False)

# Add the necessary LaTeX commands
latex_output = '\\begin{table}[]\n\\centering\n' + latex_output
latex_output += '\n\\caption{Caption}\n\\label{tab:my_label}\n\\end{table}'

# Write the LaTeX output to a file
with open(f'df_merged_{col}.tex', 'w') as f:
    f.write(latex_output)
# %%
# examine the crossing points
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# Plot the normalized dataframes and the crossing points as vertical lines
for ax, df, crossing in zip(axs, [df_sensitivity_S1_normal, df_sensitivity_ST_normal, df_pawn_median_normal], [crossing_s1, crossing_st, crossing_pawn]):
    df.plot(ax=ax)
    for key, values in crossing.items():
        for value in values:
            ax.vlines(x=value, ymin=0, ymax=1, colors='r')
            ax.text(x=value, y=0.5, s=str(value), color='r', ha='right')


plt.tight_layout()
plt.show()

# Calculate the area under the curve for each column in df_sensitivity_S1_normal
areas_pawn = df_pawn_median_normal.fillna(0).apply(integrate.trapz).sort_values(ascending=False)
areas_ST = df_sensitivity_ST_normal.fillna(0).apply(integrate.trapz).sort_values(ascending=False)
combined_areas = pd.concat([areas_pawn, areas_ST], axis=1, join='outer', keys=['PAWN', 'ST'])

combined_areas.round().to_csv(f'{config.p_out}_AUC_{col}.csv')
print("Areas under the curve for each column in df_sensitivity_pawn_normal:", areas_pawn.round(0))
print("Areas under the curve for each column in df_sensitivity_ST_normal:", areas_ST.round(0))
#
# %%
# bring back the crossing points clustering
clusters_s1 = process_crossing_points(crossing_s1)
clusters_st = process_crossing_points(crossing_st)
clusters_pawn = process_crossing_points(crossing_pawn)
# %%
# examine the crossing points
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# Plot the normalized dataframes and the crossing points as vertical lines
for ax, df, crossing in zip(axs, [df_sensitivity_S1_normal, df_sensitivity_ST_normal, df_pawn_median_normal], [clusters_s1, clusters_st, clusters_pawn]):
    df.plot(ax=ax)
    values = list(set(crossing))
    for value in values:
        ax.vlines(x=value, ymin=0, ymax=1, colors='r')
        ax.text(x=value, y=0.5, s=str(value), color='r', ha='right')


plt.tight_layout()
fig.suptitle('Crossing points after clustering - could be used as a supplementary figure in the manuscript')
plt.show()
plt.savefig(f'{config.p_out}/{col}_Si_crossing_points_afterClustering.png', dpi = 300, bbox_inches = 'tight')

