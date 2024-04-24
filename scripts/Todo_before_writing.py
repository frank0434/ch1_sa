# %%
from pcse.util import astro 
from pcse.util import doy
from pcse.util import daylength
from pcse.fileinput import ExcelWeatherDataProvider
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
# %%
# Checking photoperiod differences 
# Read the weather data from the excel file
wdp_ind = ExcelWeatherDataProvider('../data_raw/India2022_23.xlsx')
wdp_nl = ExcelWeatherDataProvider('../data_raw/350_weatherfile_2021.xlsx')
df_ind = pd.DataFrame(wdp_ind.export())
df_nl = pd.DataFrame(wdp_nl.export())
df_ind.DAY = pd.to_datetime(df_ind.DAY)
df_nl.DAY = pd.to_datetime(df_nl.DAY)

NL_START = '2021-04-22'
NL_END = "2021-09-30"
df_nl = df_nl[(df_nl.DAY >= NL_START) & (df_nl.DAY <= NL_END)]
df_ind = df_ind[(df_ind.DAY >= '2022-11-11')]

# %% 
LAT_ind = df_ind['LAT'].unique()[0]
LAT_nl = df_nl['LAT'].unique()[0]
# astro(day = df_ind['DAY'][0], latitude = LAT_ind, radiation = df_ind['IRRAD'][0])
# %%
# doy(day = df_ind['DAY'][0])
df_ind['daylength'] = df_ind['DAY'].apply(lambda day: daylength(day=day, latitude=LAT_ind))
df_nl['daylength'] = df_nl['DAY'].apply(lambda day: daylength(day=day, latitude=LAT_nl))
# %%
df_ind['DAP'] = np.tile(np.arange(len(df_ind)), 1)
df_ind.set_index('DAP', inplace=True)
df_nl['DAP'] = np.tile(np.arange(len(df_nl)), 1)
df_nl.set_index('DAP', inplace=True)

df_nl.set_index('DAY', inplace=True)
df_ind.set_index('DAY', inplace=True)
# %%

# df_ind['daylength'].plot(label='India')
# df_nl['daylength'].plot(label='Netherlands')
# plt.legend()
# plt.ylim(0, 24)
# plt.ylabel('Daylength (hours)')
# plt.xlabel('Days after planting')
# plt.title('Daylength differences between India and Netherlands')
# plt.yticks(np.arange(0, 25, 1))
# plt.show()