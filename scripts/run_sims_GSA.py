# %% 
import json
import datetime
import config
import run_wofost
import multiprocessing as mp
import time
from functools import partial
from collections import namedtuple
from tqdm import tqdm
from SALib.sample import saltelli
import psutil
import os
import argparse

try:
    # Create the parser
    parser = argparse.ArgumentParser(description='Run GSA simulations.')
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
config.set_variables(GSA_sample_size)
# %%
# Define a named tuple to hold all details of this run
RunDetails = namedtuple("RunDetails", ['crop_name','variety_name', 'campaign_start_date', 
                                       'crop_start_date', 'crop_end_date'])
from pcse.fileinput import ExcelWeatherDataProvider
# load weather data
wdp = ExcelWeatherDataProvider(config.Weather_real)
print(f"Read weather data fine.")

def create_run_details():
    """
    Create a `RunDetails` object to define the agronomic details.
    
    This function uses the `config` module to get the necessary details.
    The `RunDetails` object is a named tuple with the following fields:
    - crop_name: The name of the crop. Currently hard-coded to 'potato'.
    - variety_name: The name of the variety. Taken from `config.variety_name`.
    - campaign_start_date: The start date of the campaign. Taken from `config.SIMULATION_START_DATE`.
    - crop_start_date: The start date of the crop. Taken from `config.SIMULATION_CROP_START`.
    - crop_end_date: The end date of the crop. Taken from `config.SIMULATION_END_DATE_real`.
    
    Returns:
        A `RunDetails` object with the agronomic details.
    """
    d = dict(
        crop_name = 'potato',  # The crop name is currently hard-coded to 'potato'
        variety_name = config.variety_name,  # The variety name is taken from the config
        campaign_start_date = config.SIMULATION_START_DATE,  # The campaign start date is taken from the config
        crop_start_date = config.SIMULATION_CROP_START,  # The crop start date is taken from the config
        crop_end_date = config.SIMULATION_END_DATE_real  # The crop end date is taken from the config
    )
    run_details = RunDetails(**d)  # Create the RunDetails object
    return run_details  # Return the RunDetails object
# %%
def date_to_string(obj):
    if isinstance(obj, datetime.date):
        return obj.isoformat()
    return obj
# Instead of creating a list all at once, create a generator that yields one item at a time
def generate_paramsets(problem, nsamples):
    for paramset in saltelli.sample(problem, nsamples, config.calc_second_order):
        yield paramset

   # create a partial functions that pre-defines the target_variable, problem and run_details in run_wofost_simulation()
    # because they are static for each function call in the loop. Only the paramset changes.
run_details = create_run_details()
print(f"Agronomic details created successfully.")   
problem = config.problem
def run_wofost_partial(args):
    id, paramset = args
    return run_wofost.run_wofost_simulation(id, paramset, run_details, wdp, problem=problem, local=False)
 
def main():
    # Define location, crop and season
    start_time = time.time()
    memory_info = psutil.virtual_memory()
    print(f"Memory usage at start: {memory_info.percent}%")
    initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2

    nsamples = GSA_sample_size
   # run_wofost_partial = partial(run_wofost.run_wofost_simulation, id, run_details=run_details, wdp=wdp, problem=problem, local=False)

    # Create a dictionary to store the hashes and paramsets
    hash_dict = {}
    print("Start multi-processes")
    # tracemalloc.start()
    # Create a progress bar
    with tqdm(total=config.Total_sims) as pbar:
        with mp.Pool(CPUs) as pool:
            paramsets = list(generate_paramsets(problem, nsamples))
            results = pool.imap(run_wofost_partial, enumerate(paramsets))
            # for i, (result, paramset) in enumerate(pool.imap(run_wofost_partial, generate_paramsets(problem, nsamples))):
            for i, result in enumerate(results):
                if isinstance(result, str) and result.startswith("An error occurred:"):
                    print(result)
                else:                   
                    paramset_str = '_'.join(map(str, result))       # Convert paramset to a string
                    # Use the safe string as part of the filename

                    hash_dict[i] = paramset_str

                pbar.update(1)
                # If hash_dict has 1000 rows, write it to a file and clear it
                if len(hash_dict) == config.len_hash_dict:
                    with open(f'{config.p_out_sims_hash}/hash_dict_{i // config.len_hash_dict}.json', 'w') as file:
                        json.dump(hash_dict, file, default=date_to_string)
                    hash_dict.clear()

    # tracemalloc.stop()
    print(f'Multi-processes finished and dump results to folder {config.p_out_sims}.')

    # Write any remaining items in hash_dict to a file
    if hash_dict:
        with open(f'{config.p_out_sims_hash}/hash_dict_final.json', 'w') as file:
            json.dump(hash_dict, file, default=date_to_string)
    end_time = time.time()
    duration = end_time - start_time

    print(f"Simulation took {duration/60} minutes for sample size {nsamples}")    
    memory_info = psutil.virtual_memory()
    print(f"Memory usage at end: {memory_info.percent}%")
    final_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2

    # Calculate the maximum memory usage
    max_memory_usage = max(initial_memory, final_memory)

    print(f"Maximum memory usage: {max_memory_usage} MB")

if __name__ == "__main__":
    main()
