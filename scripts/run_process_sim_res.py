# %%
import json
import os
import config
import psutil
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import argparse
import pandas as pd
# %%
try:
    # Create the parser
    parser = argparse.ArgumentParser(description='Run GSA result processor.')
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
def total_sims_generator():
    """
    This function creates a generator that yields the numbers from 0 to config.Total_sims - 1.

    Returns:
        generator: A generator that yields the numbers from 0 to config.Total_sims - 1.
    """
    for i in range(config.Total_sims):
        yield i

# %%

def process_day(day):
    """
    This function processes simulations for a single day.
    It reads the simulation results from JSON files, selects the data of the current day from each file,
    and saves the results to a new JSON file for the day.

    Args:
        day (int): The day to process.

    Returns:
        None
    """
    daysim = {}

    for i in total_sims_generator():
        with open(f'{config.p_out_sims}/{i}.json', 'r') as file:
            data = json.load(file)

        daysim.update({i: data[day]})

    with open(f'{config.p_out_daysims}/day_{day}.json', 'w') as outfile:
        json.dump(daysim, outfile)

def process_all_days():
    """
    This function uses multiprocessing to process simulations for each day in the simulation period in parallel.
    It calls the process_day function for each day.

    Returns:
        None
    """
    with Pool(CPUs) as pool:
        list(tqdm(pool.imap(process_day, range(config.sim_period)), total=config.sim_period))

# %%  # extracting days when DVS = 0 or 1 
# Create the generator

# Create the generator
gen = total_sims_generator()

# Get the first item
first_item = next(gen)
data = json.load(open(f'{config.p_out_sims}/{first_item}.json'))
# Assuming 'DVS' is the key in the JSON data
dvs_data = [item for item in data if item['DVS'] in [0, 1]]
dvs_data
daydvs = {}
daydvs.update({first_item: dvs_data})
# %%
def process_file(simulations):
    df = pd.DataFrame()
    for i in simulations:
        with open(f'{config.p_out_sims}/{i}.json', 'r') as file:
            data = json.load(file)
        temp_df = pd.DataFrame([item for item in data if item['DVS'] in [0, 1]])
        temp_df['simulation'] = i
        df = pd.concat([df, temp_df])

    # Drop duplicates
    df = df.drop_duplicates(subset=[col for col in df.columns if col != 'simulation'])
    # print(df)

    # Convert DataFrame to dictionary and save as JSON
    with open(f'{config.p_out_daysims}/day_dvs_{simulations[0]}_{simulations[-1]}.json', 'w') as outfile:
        json.dump(df.to_dict(), outfile)
# process_file(range(10))
#%%

def process_files_daydvs():
    # Get a list of all simulations
    total_sims =  range(config.Total_sims)
    # Divide the simulations into chunks
    chunks = np.array_split(total_sims, CPUs)

    # Convert the list of numpy arrays into a list of lists
    chunks = [chunk.tolist() for chunk in chunks]
    with Pool(CPUs) as pool:
        list(tqdm(pool.imap(process_file, chunks), total = len(chunks)))

if __name__ == "__main__":
    # Get the initial memory usage
    initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2

    # Process all days in parallel
    process_files_daydvs()
    # process_all_days()

    final_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2

    # Calculate the maximum memory usage
    max_memory_usage = max(initial_memory, final_memory)
    print(f"Total simulation processed: {config.Total_sims}.")
    print(f"Maximum memory usage: {max_memory_usage} MB.")
# %%
