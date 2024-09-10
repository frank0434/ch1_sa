# %%
### Test the dummy variable calculation for PAWN over different:
### sample sizes, variables, and days - all the same Si values
### Because the dummy values are sampled from the same distribution as YU
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import config
import json
import pandas as pd
from SALib.sample import saltelli

class DummyVarCalculation:
    """
    A class used to perform dummy variable calculation and comparison with other variables' Si.

    ...

    Attributes
    ----------
    problem : dict
        a dictionary defining the problem
    GSA_sample_size : int
        the sample size for the GSA

    Methods
    -------
    import_data(day, col):
        Imports the data for the given day and column.
    calculate_and_store_statistics(dummy, var_pos):
        Calculates and stores the statistics for the given dummy variable and variable position.
    calculate_errors(medians, errors):
        Calculates the errors for the given medians and errors.
    calculate_for_all_variables():
        Calculates the statistics for all variables in the problem.
    plot_results():
        Plots the results.
    """
    def __init__(self, problem, GSA_sample_size):
        self.problem = problem
        self.GSA_sample_size = GSA_sample_size
        self.X = None
        self.Y = None
        self.medians = {}
        self.errors = {}

    def import_data(self, day, col):
        """
        Imports the data for the given day and column.

        Parameters
        ----------
            day : int
                the day for which to import the data
            col : str
                the column for which to import the data
        """
        with open(f'{config.p_out_daysims}/day_{day}.json') as f:
            df = pd.DataFrame(json.load(f)).T
        self.Y = df.loc[:, col].to_numpy()
        self.X = saltelli.sample(self.problem, self.GSA_sample_size, calc_second_order=True)
    def calculate_ks_statistics(self, bootstrap_samples, Y_latin):
        ks_statistics = []
        for sample in bootstrap_samples:
            ks_stat = stats.ks_2samp(sample, Y_latin)
            ks_statistics.append(ks_stat.statistic)
        return ks_statistics

    def calculate_statistics(self, X, Y_latin, dummy=True, n=10, n_bootstrap=50, subset_fraction=0.8, var_pos=0):
        """
        Calculate the Kolmogorov-Smirnov statistics for the given data.

        Parameters
        ----------
        X : numpy.ndarray
            The input data.
        Y_latin : numpy.ndarray
            The output data.
        dummy : bool, optional
            Whether the variable is a dummy variable. Defaults to True.
        n : int, optional
            The number of intervals to divide the data into. Defaults to 10.
        n_bootstrap : int, optional
            The number of bootstrap samples to generate. Defaults to 50.
        subset_fraction : float, optional
            The fraction of the data to include in each bootstrap sample. Defaults to 0.8.
        var_pos : int, optional
            The position of the variable in the problem. Defaults to 0.

        Returns
        -------
        float
            The median of the Kolmogorov-Smirnov statistics.
        list
            The 2.5 and 97.5 percentiles of the Kolmogorov-Smirnov statistics.
        """
        np.random.seed(42)  # Set the seed here to ensure reproducibility
        all_samples = []
        bootstrap_samples = []
        if n == 1:
            # If n is 1, generate bootstrap samples from the entire data
            bootstrap_samples = [np.random.choice(Y_latin, size=len(Y_latin), replace=False) for _ in range(n_bootstrap)]
        else:
            # If n is not 1, divide the data into n intervals and generate bootstrap samples from each interval
            step = 1 / n
            seq = np.arange(0,  1 + step/2, step)
            X_di = X[:, var_pos]
            X_q = np.nanquantile(X_di, seq)
            for s in range(n):
                Y_sel = Y_latin[(X_di >= X_q[s]) & (X_di < X_q[s + 1])]
                subset_size = int(len(Y_sel) * subset_fraction)
                bootstrap_samples = [np.random.choice(Y_sel if not dummy else Y_latin, size=subset_size, replace=False) for _ in range(n_bootstrap)]
        all_samples.extend(bootstrap_samples)
        # Calculate the Kolmogorov-Smirnov statistics for the bootstrap samples
        ks_statistics_var = self.calculate_ks_statistics(all_samples, Y_latin)
        ks_median_var = np.median(ks_statistics_var)
        lower_bound_ks_var = np.percentile(ks_statistics_var, 2.5)
        upper_bound_ks_var = np.percentile(ks_statistics_var, 97.5)
        return ks_median_var, [lower_bound_ks_var, upper_bound_ks_var]

    def calculate_and_store_statistics(self, dummy, var_pos):
        medians = []
        errors = []
        for n in range(1, 15):
            ks_median_var, error = self.calculate_statistics(self.X, self.Y, dummy, n=n, var_pos=var_pos)
            medians.append(ks_median_var)
            errors.append(error)
        return medians, errors

    def calculate_errors(self, medians, errors):
        lower_errors = medians - np.array([error[0] for error in errors])
        upper_errors = np.array([error[1] for error in errors]) - medians
        return [lower_errors, upper_errors]
    def calculate_for_dummy_variable(self, var_number):
        self.medians['dummy'], _ = self.calculate_and_store_statistics(True, var_pos=var_number)

    def calculate_for_all_variables(self):

        for i in range(self.problem['num_vars']):
            self.medians[f'x{i+1}'], self.errors[f'x{i+1}'] = self.calculate_and_store_statistics(False, i)

    def plot_results(self, ax):
        """
        Plots the results on the given axes.

        Parameters
        ----------
            ax : matplotlib.axes.Axes
                The axes on which to plot the results.
        """

        offset = 0.2
        for i in range(self.problem['num_vars']):
            errors = self.calculate_errors(self.medians[f'x{i+1}'], self.errors[f'x{i+1}'])
            ax.errorbar(np.array(range(1, 15)) - offset, self.medians[f'x{i+1}'], yerr=errors, fmt="o", label=f'x{i+1}')
        ax.plot(range(1, 15), self.medians['dummy'], label='dummy', color='red', linestyle='dotted')
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.1, 0.1))  # Set y-axis increments to 0.1
        ax.set_xticks(range(1, 15))
        ax.set_xlabel('n value')
        ax.set_ylabel('Median and 95% Confidence Interval')
        # ax.legend()

# Usage
# %% 
GSA_sample_size = 2 ** 15
config.set_variables(GSA_sample_size, local=True)

# %%
col = 'DVS' 
calculation_dummy = DummyVarCalculation(config.problem, GSA_sample_size)
calculation_dummy.import_data(160, 'LAI')
dummySi, dummyCI = calculation_dummy.calculate_statistics(calculation_dummy.X, calculation_dummy.Y, dummy=True, n=10, var_pos=0)
# %% 
import pickle
# Save the results to a pickle file
with open(f'{config.p_out}/DummySi_results.pkl', 'wb') as f:
    pickle.dump((dummySi, dummyCI), f)
