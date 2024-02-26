# %%
import pandas as pd
import numpy as np
import config
import pickle
import matplotlib.pyplot as plt
import config
import run_vis_Si as vis
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
col = 'LAI'
df_sensitivity_S1, df_sensitivity_ST = vis.process_files(col)
df_pawn_long = vis.create_dataframe_from_dict(vis.load_PAWN(col))
df_pawn_long = df_pawn_long[df_pawn_long['median'] > Dummy_si[1][1]]
df_pawn_median = df_pawn_long.loc[:, ["DAP","median", "names"]].pivot_table(index='DAP', columns='names', values='median').reset_index()
df_pawn_median.set_index('DAP',inplace =True)
df_pawn_median.rename_axis("index", axis='index', inplace=True)

df_pawn_median_normal = vis.normalize_sensitivity(df_pawn_median)

# %%
_, _ , crossing_s1 = process_dataframe(df_sensitivity_S1)
_, _ , crossing_st = process_dataframe(df_sensitivity_ST)
_, _ , crossing_pawn = process_dataframe(df_pawn_median_normal)

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

col_variables = ['DVS', 'LAI', 'TWSO']
results = []

for col in col_variables:
    df_sensitivity_S1, df_sensitivity_ST = vis.process_files(col)
    df_pawn_long = vis.create_dataframe_from_dict(vis.load_PAWN(col))
    df_pawn_long = df_pawn_long[df_pawn_long['median'] > Dummy_si[1][1]]
    df_pawn_median = df_pawn_long.loc[:, ["DAP","median", "names"]].pivot_table(index='DAP', columns='names', values='median').reset_index()
    df_pawn_median.set_index('DAP',inplace =True)
    df_pawn_median.rename_axis("index", axis='index', inplace=True)
    df_pawn_normal = vis.normalize_sensitivity(df_pawn_median)
    print(df_pawn_normal.columns)
    lai_cols = ['SPAN', 'te', 't1_pheno', 'TSUM1', 'TDWI', 'TSUM2', 'TSUMEM', 't2','TEFFMX', 'TBASEM', 'Q10', 't1']
    main_param_s1, selected_numbers_s1 = process_dataframe(df_sensitivity_S1)
    main_param_st, selected_numbers_st = process_dataframe(df_sensitivity_ST)
    main_pawn, selected_numbers_pawn = process_dataframe(df_pawn_median)

    results.append({
        'variable': col,
        'selected_numbers_s1': list(set(selected_numbers_s1)),
        'selected_numbers_st': list(set(selected_numbers_st)),
        'selected_numbers_pawn': list(set(selected_numbers_pawn))
    })

    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    plot_data(main_param_s1, selected_numbers_s1, f'main_param_s1 for {col}', axs[0])
    plot_data(main_param_st, selected_numbers_st, f'main_param_st for {col}', axs[1])
    plot_data(main_pawn, selected_numbers_pawn, f'main_pawn for {col}', axs[2])

    plt.tight_layout()
    plt.show()

df_results = pd.DataFrame(results)
# %%
df_results