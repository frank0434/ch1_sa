#%%
from functools import partial
import json
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import run_wofost
import pickle
from collections import namedtuple
import pandas as pd
import config
from pcse.fileinput import ExcelWeatherDataProvider

### Input files 
# this file path should be as an input parameter or hard code in 
# %%
# with open(config.Weather_AgERA5, 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    # wdp = pickle.load(f)
RunDetails = namedtuple("RunDetails", ['crop_name','variety_name', 'campaign_start_date', 
                                       'crop_start_date', 'crop_end_date'])

def load_parameter_values(path=config.parameter_list,  variety_name=config.variety_name):
    """
    Load the parameter values from the excel file specified in the config.

    Args:
        path: The file path to the excel file containing the parameter values.

    Returns:
        A pandas DataFrame containing the parameter values.
    """
    para_list = pd.read_excel(path, sheet_name='Sheet1')
    Scalar_para_list = para_list[(para_list['Type'] == 'Scalar') & (para_list['ExpertOpinion'].isin([1, 2]))]
    param_df = Scalar_para_list[~Scalar_para_list['Key'].isin(['IDSL','DVSEND','DVSI','DLC','DLO'])]
    param_df = param_df.loc[:, ['Key', 'Max', 'Min']]
    param_df['Variety'] = variety_name
    return param_df

def prepare_simulation_parameters(key_range_df):
    """
    Prepare the parameters for running simulations.

    This function takes a DataFrame of keys and their ranges, generates incremental values for each key   

    Args:
        key_range_df: A pandas DataFrame that contains 'Key', 'Max', and 'Min' columns.

    Returns:
        A tuple of two numpy arrays. The first array contains the keys, and the second array contains the values.
    """
    # Generate config.LSA_step_size incremental values for each key

    keys, values = [], []
    for _, row in key_range_df.iterrows():
        key, max_value, min_value = row
        generated_values = np.linspace(min_value, max_value, config.LSA_sample_size)
        generated_values = np.round(generated_values, 4)
        keys.extend([key] * config.LSA_sample_size)
        values.extend(generated_values)

    keys = np.array(keys)
    values = np.array(values)

    return keys, values

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

#%%
# Create a RunDetails object with the agronomic details from the config
run_details = create_run_details()
print(f"Agronomic details created successfully.")   
wdp = ExcelWeatherDataProvider(config.Weather_real)
def run_wofost_partial(args):
    id, paramset = args
    return run_wofost.run_wofost_simulation(id, paramset, run_details, wdp)
print("Partial function prepared successfully.")
param_df = pd.DataFrame({'Key' : config.problem['names'],
                        'Min' : [item[0] for item in config.problem['bounds']],
                        'Max' : [item[1] for item in config.problem['bounds']]})

key, value = prepare_simulation_parameters(param_df)
transformed_paramsets = np.column_stack([key, value])

# %%
def main():


    # parameters = {}
    # for item in zip(param_df['Key'], param_df['Min'], param_df['Max']):
    #     key, min_val, max_val = item[0], item[1], item[2]
    #     parameters[key] = (float(min_val), float(max_val))

    # Define the number of samples
    # n_samples = config.LSA_sample_size

    # Sample each parameter value
    # samples = {}
    # for key, (min_val, max_val) in parameters.items():
    #     samples[key] = np.random.uniform(min_val, max_val, n_samples)
    key, value = prepare_simulation_parameters(param_df)
    print(key, value)
    transformed_paramsets = np.column_stack([key, value])
    num_of_rows = len(transformed_paramsets)
    hash_dict = {}
    print("Starting multi-process simulations.")
    # Use a progress bar to track the progress of the simulations
    with tqdm(total=num_of_rows) as pbar:
        # Use a multiprocessing Pool to run the simulations in parallel
        with mp.Pool(10) as pool:
            for i, result in enumerate(pool.imap(run_wofost_partial, enumerate(transformed_paramsets))):
                # Append the result of each simulation to the results list
                if isinstance(result, str) and result.startswith("An error occurred:"):
                    print(result)
                else:
                    hash_dict[i] = result
                pbar.update(1)
    if hash_dict:
        with open(f'{config.p_out_LSAsims}/hash_dict_final.json', 'w') as file:
            json.dump(hash_dict, file)
if __name__ == '__main__':
    main()
# %%



# # %%
# # Load the parameter values from the CSV file specified in the config
# param_df = load_parameter_values()
# print(f"Parameter range data loaded successfully.")

# # Extract the 'Key' and 'Value' columns from the parameter DataFrame
# keys, values = prepare_simulation_parameters(param_df)

# # Combine keys and values into a structured array for further processing
# paramsets = np.column_stack((keys, values))
# num_of_rows = paramsets.shape[0]
# # %%

# # %%

# # Parse the parameter list
# import numpy as np

# parameters = {}
# for item in zip(param_df['Key'], param_df['Min'], param_df['Max']):
#     key, min_val, max_val = item[0], item[1], item[2]
#     parameters[key] = (float(min_val), float(max_val))

# # Define the number of samples
# n_samples = 1000

# # Sample each parameter value
# samples = {}
# for key, (min_val, max_val) in parameters.items():
#     samples[key] = np.random.uniform(min_val, max_val, n_samples)
# transformed_paramsets = np.vstack([[key, value] for key, values in samples.items() for value in values])

# # %%
# wdp = ExcelWeatherDataProvider(config.Weather_real)

# run_wofost_partial = partial(run_wofost.run_wofost_simulation, run_details=run_details, wdp=wdp)
# run_wofost_partial(1, ['TSUM1',samples['TSUM1'][0]])
