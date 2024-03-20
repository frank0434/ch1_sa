# %%

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
#%%


# Generate random numbers from a normal distribution
data = np.random.normal(size=1000)

# Create a kernel density estimate
density = gaussian_kde(data)
xs = np.linspace(np.min(data), np.max(data), 1000)
density.covariance_factor = lambda : .25
density._compute_covariance()

# Define the boundaries of the sections
boundaries = [-3, -1, 1, 3]

# Define the colors for the sections
colors = ['red', 'blue', 'yellow']

# Create a figure and axis
fig, ax = plt.subplots()

# Plot each section with a different color
for low, high, color in zip(boundaries[:-1], boundaries[1:], colors):
    mask = (xs > low) & (xs <= high)
    ax.fill_between(xs[mask], density(xs[mask]), color=color)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xticklabels([])
ax.set_yticklabels([])
# Show the plot
plt.show()
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Define the means and standard deviations for the synthetic data
means = [-1, 0, 1]
stds = [0.5, 1, 1.5]

# Generate three sets of synthetic data from normal distributions
data_sets = [np.random.normal(loc=mean, scale=std, size=1000) for mean, std in zip(means, stds)]

# Define the colors for the plots
colors = ['red', 'blue', 'yellow']

# Create a figure and three subplots
fig, axs = plt.subplots(3, 1, sharex=True, sharey=True)

# Calculate the density for each set of data and plot it on a separate subplot
for ax, data, color in zip(axs, data_sets, colors):
    density = gaussian_kde(data)
    xs = np.linspace(np.min(data), np.max(data), 1000)
    density.covariance_factor = lambda : .25
    density._compute_covariance()
    ax.fill_between(xs, density(xs), color=color)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Show the plot
plt.show()
# %%
