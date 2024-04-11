# %%
import pandas as pd
import matplotlib.pyplot as plt
import config
from pcse.fileinput import ExcelWeatherDataProvider

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
# Filter the DataFrame
from datetime import datetime

# Convert the strings to datetime.date objects
SIMULATION_START_DATE = datetime.strptime(config.SIMULATION_START_DATE, '%Y-%m-%d').date()
SIMULATION_END_DATE_real = datetime.strptime(config.SIMULATION_END_DATE_real, '%Y-%m-%d').date()

# Now you can filter the DataFrame
df_IND = df_IND[(df_IND.index >= SIMULATION_START_DATE) & (df_IND.index <= SIMULATION_END_DATE_real)]
df_NL = df_NL[(df_NL.index >= datetime.strptime("2021-04-20", '%Y-%m-%d').date()) & (df_NL.index <= datetime.strptime("2021-09-30",'%Y-%m-%d').date())]
# %%

df_IND['Tmean'] = (df_IND['TMAX'] + df_IND['TMIN']) / 2
df_NL['Tmean'] = (df_NL['TMAX'] + df_NL['TMIN']) / 2
df_IND['daytimeTemp'] = ((df_IND['TMAX'] + df_IND['TMIN']) / 2 + df_IND['TMAX'])/2
df_NL['daytimeTemp'] = ((df_NL['TMAX'] + df_NL['TMIN']) / 2 + df_NL['TMAX'])/2
# %%
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates

fig = plt.figure(figsize=(10, 10))

# Create a GridSpec for the whole figure
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.1, wspace=0.2)

# First subplot
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(df_IND.index, df_IND['TMAX'], label='Maximum Temperature', color='r')
ax1.plot(df_IND.index, df_IND['daytimeTemp'], label='Daytime Mean Temperature', color='black')
ax1.plot(df_IND.index, df_IND['Tmean'] , label='Mean Temperature', color='grey')
ax1.plot(df_IND.index, df_IND['TMIN'], label='Minimum Temperature', color='g')
ax1.xaxis.set_major_locator(mdates.MonthLocator())
ax1.set_ylabel('Temperature (°C)')
ax1.set_xticklabels('')
ax1.set_title('Indian Weather Data')
ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
ax2.plot(df_NL.index, df_NL['TMAX'], label='Maximum Temperature', color='r')
ax2.plot(df_NL.index, df_NL['daytimeTemp'], label='Daytime Mean Temperature', color='black')
ax2.plot(df_NL.index, df_NL['Tmean'] , label='Mean Temperature', color='grey')
ax2.plot(df_NL.index, df_NL['TMIN'], label='Minimum Temperature', color='g')
ax2.set_xticklabels('')
ax2.legend()
ax2.set_title('Dutch Weather Data')
# Second subplot
# Third subplot
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(df_IND.index, df_IND['IRRAD'], label='IRRAD', color='b')
ax3.xaxis.set_major_locator(mdates.MonthLocator())
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

ax3.set_ylabel('Irradiation (J/m²/day)', color='b')

# Create a second y-axis for the third subplot

ax3b = ax3.twinx()
ax3b.bar(df_IND.index, df_IND['RAIN'] * 10, label='RAIN', color='r')


ax4 = fig.add_subplot(gs[1, 1], sharey=ax3)
ax4.plot(df_NL.index, df_NL['IRRAD'], label='IRRAD', color='b')
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# ax4.set_ylabel('Irradiation', color='b')
ax4b = ax4.twinx()
ax4b.get_shared_y_axes().join(ax4b, ax3b)
ax4b.bar(df_NL.index, df_NL['RAIN'] * 10, label='RAIN', color='r')
ax4b.set_ylabel('Rainfall (mm)', color='r')
plt.savefig(f'../output/weather_data.png', dpi = 300)
plt.savefig(f'../output/weather_data.svg', dpi = 600)
plt.show()
# %%
# Plot the data
fig, axs = plt.subplots(2, 2, figsize=(7, 7), sharey=True)
axs = axs.flatten()
# Temperature
# axs[0].legend()

# axs[1].set_ylabel('Temperature')
axs[2].plot(df_NL.index, df_NL['IRRAD'], label='IRRAD')
axs[2].plot(df_NL.index, df_NL['RAIN'], label='RAIN')


plt.tight_layout()
plt.show()