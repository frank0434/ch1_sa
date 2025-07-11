# %%
import pandas as pd
import matplotlib.pyplot as plt
import config
from pcse.fileinput import ExcelWeatherDataProvider
# import Todo_before_writing as daylength_data
from datetime import datetime
# Load the data
wdp_NL = ExcelWeatherDataProvider('../data_raw\\350_weatherfile_2021.xlsx')
wdp_IND = ExcelWeatherDataProvider('../data_raw\\India2022_23.xlsx')
# %%
df_NL = pd.DataFrame(wdp_NL.export())
df_IND = pd.DataFrame(wdp_IND.export())
config.SIMULATION_START_DATE 
config.SIMULATION_END_DATE
df_NL.set_index('DAY', inplace=True)
df_IND.set_index('DAY', inplace=True)

# %% 
import pcse
import pcse.fileinput 
from pcse.fileinput import NASAPowerWeatherDataProvider
from scipy import stats
# In terminal/command prompt:
# conda create -n pcse_new python=3.x
# conda activate pcse_new
# pip install pcse==newer_version
pcse.__version__
# wdp = NASAPowerWeatherDataProvider(latitude=52, longitude=5)
# print(wdp)
wdp = NASAPowerWeatherDataProvider(51.54, 5.86)
print(wdp)
df = pd.DataFrame(wdp.export())
# filter the data from 2021-01-01 to 2021-10-11
df['DAY'] = pd.to_datetime(df['DAY'])
df = df[(df['DAY'] >= '2021-01-01') & (df['DAY'] <= '2021-10-11')].set_index('DAY')

# plot the data
df_nl_NASA = df.copy()
# compare with the original data
df_nl_orig = df_NL.copy()
# %% plot the data in each column as x and y pairs
import matplotlib.pyplot as plt
plt.figure(figsize=(5, 5))
plt.scatter(df_nl_orig['IRRAD'], df_nl_NASA['IRRAD'], alpha=0.6)
plt.xlabel('Weather Station measured solar radiation')
plt.ylabel('NASA\'s solar radiation')
plt.title('NASA vs Weather Station IRRAD Comparison in the Netherlands')

# Add 1:1 line
min_val = min(df_nl_orig['IRRAD'].min(), df_nl_NASA['IRRAD'].min())
max_val = max(df_nl_orig['IRRAD'].max(), df_nl_NASA['IRRAD'].max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 line')

# Calculate R² and linear regression equation
slope, intercept, r_value, p_value, std_err = stats.linregress(df_nl_orig['IRRAD'], df_nl_NASA['IRRAD'])
r_squared = r_value**2

# Add R² and equation as text on the plot
plt.text(0.05, 0.95, f'R² = {r_squared:.2f}\ny = {slope:.2f}x + {intercept:.2f}', 
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()