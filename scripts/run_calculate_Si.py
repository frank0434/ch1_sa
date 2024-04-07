# %%
from SALib.sample import saltelli
from SALib.analyze import sobol, pawn
import pandas as pd
import numpy as np
import config
import pickle
import time
import psutil
import json
import os
import multiprocessing as mp
from tqdm import tqdm
import warnings
import argparse


try:
    # Create the parser
    parser = argparse.ArgumentParser(description='Run GSA Si Calculator.')
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
    GSA_sample_size = 32
    CPUs = psutil.cpu_count(logical=False)
# Set the variables in config
config.set_variables(GSA_sample_size, local = True)
# %%
# Function to calculate Sobol indices
start_time = time.time()

def calculate_indices(day,col):
    """
    This function calculates the Sobol and Pawn indices for a given day and column.

    Args:
        day (int): The day number in the simulation period.
        col (str): The column name.

    Returns:
        an array of Y values for the day and column over all the samples
    """
    # Load the day values from the JSON file
    with open(f'{config.p_out_daysims}/day_{day}.json', 'r') as file:
        day_values = json.load(file)

    # Convert the day values to a DataFrame and then to a numpy array
    df = pd.DataFrame(day_values).T
    # print(day_values)
    arraryday = df.loc[:, col].to_numpy()

    param_values =  saltelli.sample(config.problem, GSA_sample_size, 
                                    calc_second_order=True)


    saltelli_indices = {}
    pawn_indices = {}

    try:
        # Suppress runtime warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # Calculate the Sobol and Pawn indices
            saltelli_indices[f"si_day_{day}_{col}"] = sobol.analyze(config.problem, arraryday, calc_second_order=True, print_to_console=False)
            pawn_indices[f"si_day_{day}_{col}"] = pawn.analyze(config.problem, param_values, arraryday, seed = 42)
    except (TypeError, ZeroDivisionError):
        # Handle the case where the item is None
        saltelli_indices[f"si_day_{day}_{col}"] = np.nan
        pawn_indices[f"si_day_{day}_{col}"] = np.nan

    return saltelli_indices, pawn_indices
# %%
def worker(day, column = config.cols_of_interests):
    """
    This function calculates the Sobol sensitivity indices (Si) for a single day and saves it in a pickled file.

    Args:
        day (int): The day for which to calculate the Si.

    Returns:
        None
    """
    # Loop over each column of interest
    print(f'Start to calculate for {day} Si for all col of interests!') 
    staltelli_indices = {}
    pawn_indices = {}

    for  col in column:   
    
        # Call the calculate_indices function
        day_saltelli_indices, day_pawn_indices = calculate_indices(day, col)
        # Update the dictionary
        staltelli_indices.update(day_saltelli_indices)
        pawn_indices.update(day_pawn_indices)
        # Save the Si in a pickled file
        with open(f'{config.p_out_daySi}/Saltelli_{day}_{col}.pkl', 'wb') as f:
            pickle.dump(day_saltelli_indices, f)
        with open(f'{config.p_out_daySi}/PAWN_{day}_{col}.pkl', 'wb') as f:
            pickle.dump(day_pawn_indices, f)


def calculate_sobol_Si_multiprocessing():
    """
    This function uses multiprocessing to calculate the Sobol sensitivity indices (Si) for each day.

    Returns:
        None
    """
    # Create a multiprocessing Pool
    with mp.Pool(CPUs) as pool:
        # Create a progress bar
        with tqdm(total=config.sim_period) as pbar:
            for i in pool.imap_unordered(worker, range(config.sim_period)):
                # Update the progress bar
                pbar.update()

    print("Sobol sensitivity indices calculated for all days.")

# %%
if __name__ == "__main__":
    # Get the initial memory usage
    initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
       # Call the multiprocessing function
    calculate_sobol_Si_multiprocessing()
    # Get the final memory usage
    final_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2

    # Calculate the maximum memory usage
    max_memory_usage = max(initial_memory, final_memory)
    print(f'Calculate total simulation number: {config.Total_sims} for {config.sim_period} days!')

    print(f"Maximum memory usage: {max_memory_usage} MB")
# %%
