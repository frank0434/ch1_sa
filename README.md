# Time-dependent sensitivity analysis is necessary to improve crop model performance under high temperatures (Submitted)

### Documentation for Scripts

#### `run_sims_GSA.py`
This script runs Global Sensitivity Analysis (GSA) simulations using the WOFOST model. It reads weather data, sets up simulation parameters, and runs the simulations in parallel using multiple CPUs.

- **Arguments**:
  - `--GSA_sample_size`: The GSA sample size.
  - `--CPUs`: The number of CPUs to use.

- **Key Functions**:
  - `create_run_details()`: Creates a `RunDetails` object with agronomic details.
  - `generate_paramsets(problem, nsamples)`: Generates parameter sets for the simulations.
  - `run_wofost_partial(args)`: Runs a single WOFOST simulation.
  - `main()`: Main function to run the simulations in parallel and save results.

#### `run_sims_LSA.py`
This script runs Local Sensitivity Analysis (LSA) simulations. It prepares simulation parameters, runs the simulations in parallel, and saves the results.

- **Key Functions**:
  - `load_parameter_values(path, variety_name)`: Loads parameter values from an Excel file.
  - `prepare_simulation_parameters(key_range_df)`: Prepares parameters for simulations.
  - `create_run_details()`: Creates a `RunDetails` object with agronomic details.
  - `run_wofost_partial(args)`: Runs a single WOFOST simulation.
  - `main()`: Main function to run the simulations in parallel and save results.

#### `run_process_sim_res.py`
This script processes the results of GSA simulations. It reads simulation results, processes them for each day, and saves the processed results.

- **Key Functions**:
  - `total_sims_generator()`: Generates numbers from 0 to `config.Total_sims - 1`.
  - `process_day(day)`: Processes simulations for a single day.
  - `process_all_days()`: Processes simulations for all days in parallel.
  - `process_file(simulations)`: Processes simulation files to extract specific data.
  - `process_files_daydvs()`: Processes files in parallel to extract specific data.

#### `run_calculate_Si.py`
This script calculates Sobol sensitivity indices (Si) for GSA simulations. It uses multiprocessing to calculate indices for each day and saves the results.

- **Key Functions**:
  - `calculate_indices(day, col)`: Calculates Sobol and Pawn indices for a given day and column.
  - `worker(day, column)`: Calculates Sobol indices for a single day.
  - `calculate_sobol_Si_multiprocessing()`: Uses multiprocessing to calculate Sobol indices for each day.

#### `run_wofost.py`
This script runs WOFOST simulations with given parameters and agronomic details. It defines agromanagement, retrieves model parameters, and runs the simulation.

- **Key Functions**:
  - `define_agromanagement(run_details)`: Defines agromanagement for the crop.
  - `get_modelparameters(run_details)`: Retrieves parameter sets for crop, soil, and site.
  - `run_wofost_simulation(id, paramset, run_details, wdp, problem, local)`: Runs a WOFOST simulation and saves the output.

#### `slurm_script.sh`
This SLURM script runs GSA simulations on a high-performance computing cluster. It sets up the environment, specifies job requirements, and runs the `run_sims_GSA.py` script with different sample sizes.

#### `slurm_script_calculateSi.sh`
This SLURM script processes GSA simulation results and calculates Sobol indices. It sets up the environment, specifies job requirements, and runs the `run_process_sim_res.py` script.

#### `run_vis_GSA.py`
This script visualizes the results of GSA simulations. It generates heatmaps and plots sensitivity indices over time.

- **Key Functions**:
  - `generate_file_names(col)`: Generates file names for each day and column.
  - `plot_heatmaps(days, col)`: Plots heatmaps for specified days and column.
  - `process_files(col)`: Reads data from pickle files and stores them in DataFrames.
  - `normalize_sensitivity(df, threshold)`: Normalizes sensitivity indices.
  - `plot_sobol_Si_multiprocessing()`: Uses multiprocessing to plot Sobol indices for each day.
  - `process_dvs_files()`: Processes files to extract specific data.
  - `plot_sensitivity_indices(df_sensitivity_S1, df_sensitivity_ST, df_pawn, emergence_date, tuber_initiation, col)`: Plots sensitivity indices.

#### `slurm_script_visSi.sh`
This SLURM script visualizes Sobol indices for GSA simulations. It sets up the environment, specifies job requirements, and runs the `run_vis_GSA.py` script.

---

Feel free to let me know if you need any further details or modifications!

## Acknowledgements

This Readme documentation used the assistance of Microsoft Copilot, an AI model based on the GPT-4 architecture. 


