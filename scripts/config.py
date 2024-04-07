# %%
import pathlib
from datetime import datetime
# %%
# Resolve the parent directory
p = pathlib.Path().resolve().parent
# %% 
# Switch to choose between NL and Indian conditions
# If run_NL_conditions is True, the script will run the NL conditions
# If run_NL_conditions is False, the script will run the Indian conditions
# To switch between the two, simply change the value of run_NL_conditions
run_NL_conditions = True
local = True
GSA_sample_size = 32
LSA_sample_size = 100

calc_second_order = True
len_hash_dict = 1000
#%%
# Define paths - input 
p_dat_raw = p / 'data_raw'
p_dat_processed = p / 'data_processed'  
parameter_list = p_dat_raw / "TempRelateParams.xlsx"
potato_paramters = p_dat_raw / "potato.yaml"
Weather_AgERA5 = p_dat_processed / "wdp_ind.pickle"
SOIL_DATA_PATH = p_dat_raw / "ec3 sandyloam.soil"
CROP_DATA_PATH = p_dat_raw 

## GSA PROBLEM configuration
problem = {
    'num_vars': 15,
    'names': [
        'TSUM1',
        'TSUM2',
        'SPAN',
        'Q10',
        'TBASEM',
        'TSUMEM',
        'TEFFMX',
        'TDWI',
        'RGRLAI',
        'tm1',
        't1',
        't2',
        'te',
        't1_pheno',
        'te_pheno'
        ],
   'bounds': [
       [150, 280], 
       [1550, 2100], 
       [20, 50],
       [2, 3],
       [2, 4],
       [170, 255],
       [18, 32],
       [75, 700],
       [0.008, 0.02],
       [5, 10],
       [10, 20],
       [20, 25],
       [25, 40],
       [2,8],
       [25,35]
       ]
}



# run NL conditions
if run_NL_conditions:
    SIMULATION_START_DATE = "2021-06-08"
    SIMULATION_CROP_START = '2021-06-08'
    SIMULATION_END_DATE = "2021-09-24"
    SIMULATION_END_DATE_real = "2021-09-24"  # Real weather station data ends at 24th
    variety_name = "Fontane"  # Other available cultivars: ["Fontane", "Markies","Premiere", "Festien", "Innovator"]
    planting = "2021-06-08"
    harvest = ['2021-07-17', '2021-08-14', '2021-09-12']
    # run the NL conditions
    Weather_real = p_dat_raw / "350_weatherfile_2021.xlsx"
else:
    # Model configuration for Indian conditions
    SIMULATION_START_DATE = "2022-11-10"
    SIMULATION_CROP_START = '2022-11-10'
    SIMULATION_END_DATE = "2023-02-28"
    SIMULATION_END_DATE_real = "2023-02-24"  # Real weather station data ends at 24th
    variety_name = "Fontane"  # Other available cultivars: ["Fontane", "Markies","Premiere", "Festien", "Innovator"]
    planting = "2022-11-10"
    harvest = ['2022-12-19', '2023-01-16', '2023-02-14']
    Weather_real = p_dat_raw / "India2022_23.xlsx"


def set_variables(GSA_sample_size, local = local, run_NL_conditions = run_NL_conditions):
    global p_out, p_out_sims, p_out_sims_hash, p_out_daysims, p_out_daySi, Total_sims

    # output directories 
    if local:
        p_out = p / 'output_NL' if run_NL_conditions else p / 'output'
    else:
        p_out = pathlib.Path('/lustre/nobackup/INDIVIDUAL/liu283/NL_output') if run_NL_conditions else pathlib.Path('/lustre/nobackup/INDIVIDUAL/liu283/')
    p_out_LSA = p / 'output/LSA'
    p_out_LSAsims = p_out_LSA / f'sims_NL_{LSA_sample_size}' if run_NL_conditions else p_out_LSA / f'sims_{LSA_sample_size}'    
    p_out_sims =  p_out / f'sims_{GSA_sample_size}'
    p_out_sims_hash = p_out_sims / f'hash_dict_{GSA_sample_size}'
    p_out_daysims = p_out / f'daysims_{GSA_sample_size}'
    p_out_daySi = p_out / f'daySi_{GSA_sample_size}'

    # Create directories if they don't exist
    
    p_out_LSA.mkdir(parents=True, exist_ok=True)
    p_out_LSAsims.mkdir(parents=True, exist_ok=True)
    p_out.mkdir(parents=True, exist_ok=True)
    p_out_sims.mkdir(parents=True, exist_ok=True)
    p_out_daysims.mkdir(parents=True, exist_ok=True)
    p_out_sims_hash.mkdir(parents=True, exist_ok=True)
    p_out_daySi.mkdir(parents=True, exist_ok=True)

    # Calculate Total_sims
    Total_sims = GSA_sample_size * (2*len(problem['names']) + 2)  # replace 'problem' with the actual variable
# Calculate the difference in days
sim_period = (datetime.strptime(SIMULATION_END_DATE_real, "%Y-%m-%d") - datetime.strptime(SIMULATION_CROP_START, "%Y-%m-%d")).days
# input and output columns

cols_of_interests = ['DVS', 'LAI', 'TAGP', 'TWSO', 'TWLV', 'TWST', 'TWRT', 'TRA']
params_of_interests = problem['names']

# Create a dictionary to map names to colors
name_color_map = {
    'mean': 'blue',
    'median': 'red',
    'minimum': 'grey',
    'maximum': 'grey',
    'CV': 'green',  # Adjust colors as needed
    'TSUM1' : 'blue',
    'TSUM2' : 'green',
    'SPAN' : 'red',
    'Q10' : 'purple',
    'TBASEM' : 'tan',
    'TSUMEM' : 'orange',
    'TEFFMX' : 'navy',
    'TDWI' : 'brown', 
    'RGRLAI' : 'pink',
    'tm1' : 'cyan',
    't1' : 'magenta',
    't2' : 'lime',
    'te' : 'yellow',
    't1_pheno' : 'black',
    'te_pheno' : 'lightblue'
}

# %%
def calculate_days_difference(planting, harvest):
    # Convert planting date to datetime
    planting_date = datetime.strptime(planting, "%Y-%m-%d")
    # Convert harvest dates to datetime and calculate differences
    harvest_dates = [datetime.strptime(date, "%Y-%m-%d") for date in harvest]
    differences = [(date - planting_date).days for date in harvest_dates]
    # Subtract 1 from each value in differences
    differences = [difference - 1 for difference in differences]
    differences.append(105)
    return differences
days_s2 = calculate_days_difference(planting, harvest)
