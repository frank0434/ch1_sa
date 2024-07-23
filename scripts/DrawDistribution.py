# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import config

import json

# %%

with open('../output/daysims_32768/day_105.json', 'r') as f:
    data = json.load(f)
# %%
day105 = pd.DataFrame(data).T

# %%


# Set a theme
sns.set_theme()

fig, ax = plt.subplots(3, 1, figsize=(6, 8))

# Define the column names and their corresponding titles
columns = ['DVS', 'LAI', 'TWSO']
titles = ['DVS', 'LAI', 'TWSO']

for i, (col, title) in enumerate(zip(columns, titles)):
    # Use seaborn for a nicer histogram
    sns.histplot(day105[col], bins=100, kde=True, ax=ax[i])
    
    # Set the title and labels
    ax[i].set_title(col, fontsize=14)
    ax[i].set_xlabel('Value', fontsize=12)
    ax[i].set_ylabel('Frequency', fontsize=12)

# Improve layout
fig.tight_layout()

# Save the figure
fig.savefig('../output/Distribution_105_TWSO.png', dpi=300)
fig.savefig('../output/Distribution_105_TWSO.svg')

plt.show()
# %%
# x and y scatter plot
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
sns.scatterplot(data=day105, x='DVS', ax=ax)
# %%
