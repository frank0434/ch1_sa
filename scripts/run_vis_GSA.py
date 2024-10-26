#%%
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
import argparse
import pickle
import json
import glob
with open('DummySi_results.pkl', 'rb') as f:
    Dummy_si = pickle.load(f)

try:
    # Create the parser
    parser = argparse.ArgumentParser(description='Run GSA visualisation.')
    # Add the arguments
    parser.add_argument('--GSA_sample_size', type=int, help='The GSA sample size')
    parser.add_argument('--CPUs', type=int, help='The number of CPUs')
    # Parse the arguments
    args = parser.parse_args()
    # Get the GSA_sample_size
    GSA_sample_size = args.GSA_sample_size
    # Get the number of CPUs
    CPUs = args.CPUs
except:
    # Set default values if parsing command line arguments fails
    GSA_sample_size = config.GSA_sample_size
    CPUs = psutil.cpu_count(logical=False)
# Set the variables in config
config.set_variables(GSA_sample_size)

# Generator to create the file names
def generate_file_names(col, period = config.sim_period, path=config.p_out_daySi):
    for day in range(period):
        filename = f"{path}/Saltelli_{day}_{col}.pkl"
        yield day, filename 

# S2
def plot_heatmaps(days, col):
    """
    This function plots heatmaps for the specified days and column
    Parameters:
    dfs (dict): The dictionary storing the data.
    days (list): The list of days to plot.
    col (str): The column name to be plotted
    Returns:
    None
    """
    
    dfs = {}
    for day in days:
        file_name = f"{config.p_out_daySi}/Saltelli_{day}_{col}.pkl"
        with open(file_name, 'rb') as f:
            # Read the pickle file and store the data in the dictionary
            dfs.update(pickle.load(f))

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(8, 6))
    # plt.subplots_adjust(hspace=0.5, wspace=0.5)
    axs = axs.ravel()  # Flatten the array to make indexing easier
    print(f"Print heatmap for Second order Sobol indices {col}.")
    # Remove unnecessary subplots
    # for i in range(2, 2*3):
    #     fig.delaxes(axs[i])
    parameters = config.params_of_interests
    parameters = [config.label_map.get(param, param) for param in parameters]
    for i, day in enumerate(days):
        # Check if dfs[f'si_day_{day}_{col}'] is a dictionary
        if isinstance(dfs.get(f'si_day_{day}_{col}', {}), dict):
            S2_values = dfs.get(f'si_day_{day}_{col}', {}).get('S2', np.nan)
        else:
            S2_values = np.nan


        if not np.isnan(S2_values).all():
            im = axs[i].imshow(dfs[f'si_day_{day}_{col}']['S2'], cmap='viridis', interpolation='nearest')
            fig.colorbar(im, ax=axs[i])
            axs[i].set_title(f'Day {day}')
            # Set the x and y tick labels to the parameter names
            axs[i].set_xticks(np.arange(len(parameters)))
            axs[i].set_yticks(np.arange(len(parameters)))
            axs[i].set_xticklabels(parameters, rotation=90)
            axs[i].set_yticklabels(parameters)
        else:
            axs[i].axis('off')  # Hide empty subplots

    fig.suptitle(f'S2 Heatmaps for {col} in Different Days')  # Adjust the vertical position of the title
    scenario = 'NL_' if config.run_NL_conditions else ''
    plt.tight_layout()
    plt.savefig(f'{config.p_out}/{scenario}Sobol_Salteli_S2_{col}_samplesize{GSA_sample_size}.png', dpi  = 300, pad_inches=0.3)
    plt.close()
# Call the function

# %%
def process_files(col, period = config.sim_period, path=config.p_out_daySi):
    """
    This function reads data from pickle files and stores them in two pandas DataFrames.

    Parameters:
    col (str): The column name to be processed.

    Returns:
    df_sensitivity_S1 (pd.DataFrame): DataFrame storing the S1 sensitivity indices.
    df_sensitivity_ST (pd.DataFrame): DataFrame storing the ST sensitivity indices.
    """
    # Dictionary to store the data from the pickle files
    dfs = {}

    # Initialize two DataFrames to store the S1 and ST sensitivity indices
    df_sensitivity_S1 = pd.DataFrame(columns=[f"{variable}" for variable in config.params_of_interests], index=range(config.sim_period))
    df_sensitivity_ST = pd.DataFrame(columns=[f"{variable}" for variable in config.params_of_interests], index=range(config.sim_period))

    # Loop over the file names
    for day, file_name in generate_file_names(col, period, path):
        # print(day)
        # print(type(day))
        with open(file_name, 'rb') as f:
            # Read the pickle file and store the data in the dictionary
            dfs.update(pickle.load(f))
        # Update the S1 and ST DataFrames with the sensitivity indices from the current day

        try:
        # Try to assign the S1 values to the df_sensitivity_S1 DataFrame
            df_sensitivity_S1.loc[day, :] = list(dfs[f'si_day_{day}_{col}']['S1'])
            df_sensitivity_ST.loc[day, :] = list(dfs[f'si_day_{day}_{col}']['ST'])

        except TypeError:
        # If the values in the dfs object are NaN, fill all columns in the df_sensitivity_S1 DataFrame with NaN
            df_sensitivity_S1.loc[day, :] = [np.nan for _ in range(len(df_sensitivity_S1.columns))]
            df_sensitivity_ST.loc[day, :] = [np.nan for _ in range(len(df_sensitivity_ST.columns))]

    return df_sensitivity_S1, df_sensitivity_ST

def normalize_sensitivity(df, threshold=0):
    # df.reset_index(inplace=True)
    melted_df = df.reset_index().melt(id_vars='index', var_name='variable', value_name='Percentage')

    # Remove the negative values
    filtered_df = melted_df.query('Percentage >= 0')

    # Cast the long format DataFrame back to wide format
    wide_df = filtered_df.pivot(index='index', columns='variable', values='Percentage')

    wide_df.fillna(0, inplace=True)
    normalized_df = wide_df.div(wide_df.sum(axis=1), axis=0)

    # Filter out the columns where the sum is 0
    filtered_df2 = normalized_df.loc[:, (normalized_df.sum() >= threshold)]

    ordered_df = filtered_df2[filtered_df2.iloc[-1].sort_values(ascending=False).index]
    return ordered_df


# Call the function
def plot_sobol_Si_multiprocessing():
    """
    This function uses multiprocessing to calculate the Sobol sensitivity indices (Si) for each day.

    Returns:
        None
    """
    # Create a multiprocessing Pool
    # cols = config.cols_of_interests
    cols = ['DVS', 'LAI', 'TWSO']
    with mp.Pool(len(cols)) as pool:
        # Create a progress bar
        with tqdm(total=len(cols)) as pbar:
            for i in pool.imap_unordered(worker_plot, cols):
                # Update the progress bar
                pbar.update()

    print("Sobol sensitivity indices Plotted for all days.")


indices = GSA_sample_size* (len(config.params_of_interests) * 2 + 2)

# %% 
def generate_file_names_PAWN(col, period = config.sim_period, path=config.p_out_daySi):
    for day in range(period):
        filename = f"{path}/PAWN_{day}_{col}.pkl"
        yield day, filename 

def load_PAWN(col, period = config.sim_period, path=config.p_out_daySi):
    """
    This function loads data from pickle files for the specified column.

    Parameters:
    col (str): The column name.

    Returns:
    dict: A dictionary containing the loaded data.
    """
    dfs = {}
    for day, file_name in generate_file_names_PAWN(col, period, path):
        with open(file_name, 'rb') as f:
            # Read the pickle file and store the data in the dictionary
            dfs.update(pickle.load(f))
    return dfs

# %%

def create_dataframe_from_dict(pawn_si):
    dfs = []
    # Iterate over the keys in si_pawn
    for key, values in pawn_si.items():
        if pd.isna(values) or pd.isna(key):
            continue
        df = pd.DataFrame(values)
        df['DAP'] = int(key.split('_')[2])
        df['Output'] = key.split('_')[3]
        # Append the DataFrame to the list
        dfs.append(df)
    # Concatenate the list of DataFrames
    result_df = pd.concat(dfs)
    return result_df

#%%
def plot_pawn_indices(result_df, col): 
    # Create a figure and axis
    fig, ax = plt.subplots(figsize = (10, 7))
    
    # Extract unique names from the 'names' column
    unique_names = result_df['names'].unique()
    print(f"Print PAWN indices for {col}.")
    # Plot each unique name with its corresponding color
    for name in unique_names:
        name_data = result_df[result_df['names'] == name]
        color = config.name_color_map.get(name, 'black')  # Default to black if not found in the map
        ax.plot(name_data['DAP'], name_data['median'], label=name, color=color)
        # Add area plots for 'minimum' and 'maximum'
        ax.fill_between(name_data['DAP'], name_data['maximum'], name_data['minimum'], color=color, alpha=0.3)
    
    ax.set_title(col)
    fig.suptitle('PWAN median, maximum and minimum for nine parameters and eight output over time')
    fig.text(0.5, 0.06, 'Day after planting', ha='center', va='center')
    fig.text(0.08, 0.5, 'Sensitivity indices', ha='center', va='center', rotation='vertical')
    plt.legend()
    plt.savefig(f'{config.p_out}/Sobol_PAWN_Si_{col}_samplesize{GSA_sample_size}.png', dpi = 300, bbox_inches='tight')
    plt.close()
def process_and_plot_PAWN(col):
    """
    This function runs the process_files and plot_sensitivity_indices functions for the specified column.

    Parameters:
    col (str): The column name.
    """
    dfs = load_PAWN(col)
    result_df = create_dataframe_from_dict(dfs)
    plot_pawn_indices(result_df, col)


# %%

def worker_plot(col):
    """
    This function runs the process_files and plot_sensitivity_indices functions for the specified column.
    Parameters:
    col (str): The column name.
    """
    emergence_date, tuber_initiation = process_dvs_files()
    df_sensitivity_S1, df_sensitivity_ST = process_files(col)
    df_pawn_long = create_dataframe_from_dict(load_PAWN(col))
    df_pawn_median = df_pawn_long.loc[:, ["DAP","median", "names"]].pivot_table(index='DAP', columns='names', values='median').reset_index()
    df_pawn_median.set_index('DAP', inplace=True)
    df_pawn_median.index.name = 'index'
    df_sensitivity_ST, df_pawn_median = df_sensitivity_ST.align(df_pawn_median, axis=0, join='left')
    plot_sensitivity_indices(df_sensitivity_S1, df_sensitivity_ST,df_pawn_median,
                             emergence_date, tuber_initiation, col)
    plot_heatmaps(config.days_s2, col)

# %%

def process_dvs_files(base_path=config.p_out_daysims, planting_date=config.planting):
    # Define the pattern
    pattern = f'{base_path}/day_dvs_*.json'

    # Use glob to find files that match the pattern
    files = glob.glob(pattern)
    df = pd.DataFrame()

    # Load data from each file and concatenate into a DataFrame
    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
            data = pd.DataFrame(data)
            df = pd.concat([df, data])

    # Convert 'day' column to datetime and calculate 'DAP'
    df['day'] = pd.to_datetime(df['day'], format='%Y-%m-%d')
    df['DAP'] = (df['day'] - pd.to_datetime(planting_date, format='%Y-%m-%d')).dt.days

    # Calculate emergence_date and tuber_initiation
    emergence_date = df[df['DVS'] == 0]
    emergence_date = [emergence_date['DAP'].min(), emergence_date['DAP'].max()]
    tuber_initiation = df[df['DVS'] == 1]
    tuber_initiation = [tuber_initiation['DAP'].min(), tuber_initiation['DAP'].max()]

    return emergence_date, tuber_initiation
def plot_sensitivity_indices(df_sensitivity_S1, df_sensitivity_ST, df_pawn, 
                             emergence_date, tuber_initiation, col):
    """
    This function plots the S1 and ST sensitivity indices.
    Parameters:
    df_sensitivity_S1 (pd.DataFrame): DataFrame storing the S1 sensitivity indices.
    df_sensitivity_ST (pd.DataFrame): DataFrame storing the ST sensitivity indices.
    col (str): The column name to be plotted.
    Returns:
    None
    """
    print(f"Print 1st and total order Sobol indices for {col}.")
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(3, 4), sharex=True, sharey=True)



    df2 = normalize_sensitivity(df_sensitivity_ST)
    df3 = normalize_sensitivity(df_pawn)
    if col in ['LAI', 'TWSO']:
        start_date = emergence_date[0] if col == 'LAI' else tuber_initiation[0]
        df2 = df2.iloc[start_date:]
        df3 = df3.iloc[start_date:]
    colors2 = [config.name_color_map.get(col, 'black') for col in df2.columns]
    colors3 = [config.name_color_map.get(col, 'black') for col in df3.columns]
    df2.plot.area(ax=axes[0],stacked=True, color=colors2, legend=False)
    df3.plot.area(ax=axes[1],stacked=True, color=colors3, legend=False)
    lines, labels = fig.axes[0].get_legend_handles_labels()
    plt.ylim(0, 1.05)
    plt.xlim(0, config.sim_period)

    plt.xlabel('Day After Planting', fontsize = config.subplot_fs)
    fig.text(0, 0.5, 'Proportion of Sensitivity indices', va='center', rotation='vertical', fontsize = config.subplot_fs-4)

    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
    plt.gca().invert_yaxis()


    labels_final = [config.label_map.get(label, label) for label in labels]
    fig.legend(lines, labels_final, loc='center left', bbox_to_anchor=(1.0, 0.5), handlelength=1, borderpad=1, fontsize = 8)
    # labels_AUC = ['te', 'TDWI', 'TSUM1', 't1_pheno', 'TSUMEM', 'TEFFMX', 'SPAN', 't2', 'TSUM2', 'TBASEM']
    # colors_AUC = [config.name_color_map.get(col, 'black') for col in labels_AUC]
    # labels_AUC = [config.label_map.get(label, label) for label in labels_AUC]
    # labels_AUC = [f"{i+1}. {label}" for i, label in enumerate(labels_AUC)]

    # lines_AUC = [plt.Line2D([0], [0], color=c, linewidth=8, linestyle='-') for c in colors_AUC]
    # fig.legend(lines_AUC, labels_AUC, loc='upper center',  bbox_to_anchor=(0.7, 1.1), handlelength=0.3,ncol=len(labels_AUC)/2)
    for i, ax in enumerate(axes.flatten(), start=1):
        i = i if config.run_NL_conditions else i+2
        ax.text(0.01, config.subplotlab_y, chr(96+i) + ")", transform=ax.transAxes, 
                size=config.subplot_fs - 4, weight='bold')
        ax.fill_betweenx([1, 1.05], emergence_date[0], emergence_date[1], color='dimgray')
        ax.fill_betweenx([1, 1.05], tuber_initiation[0], tuber_initiation[1], color='dimgray')
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_yticklabels(['100%', '75%', '50%', '25%', '0%'])
    scenario = 'NL_' if config.run_NL_conditions else ''
    plt.tight_layout()
    plt.savefig(f'{config.p_out}/{scenario}Sobol_Salteli_PAWN_{col}_samplesize{GSA_sample_size}.svg', bbox_inches='tight')
    plt.show()
    plt.close()


# %%

if __name__ == "__main__":
    # Iterate over each column of interest and run process_and_plot_Saltelli
    initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    plot_sobol_Si_multiprocessing()

    final_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    # Calculate the maximum memory usage
    max_memory_usage = max(initial_memory, final_memory)
   
    print(f"Maximum memory usage: {max_memory_usage} MB")
    # plot_colorized_time_course(GSA_simulations, config.cols_of_interests, indices)
    # plot_colorized_time_course(GSA_simulations, config.cols_of_interests, indices)

# %%   test
# col = 'LAI'
# emergence_date, tuber_initiation = process_dvs_files()

# df_sensitivity_S1, df_sensitivity_ST = process_files(col)
# df_pawn_long = create_dataframe_from_dict(load_PAWN(col))
# df_pawn_long = df_pawn_long[df_pawn_long['median'] > Dummy_si[1][1]]
# df_pawn_median = df_pawn_long.loc[:, ["DAP","median", "names"]].pivot_table(index='DAP', columns='names', values='median').reset_index()
# # df_pawn_median.drop('names', axis=1,inplace=True)
# df_pawn_median.set_index('DAP', inplace=True)
# df_pawn_median.index.name = 'index'
# plot_sensitivity_indices(df_sensitivity_S1, df_sensitivity_ST,df_pawn_median, 
#                          emergence_date, tuber_initiation, col)
