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
# Filter the DataFrame

# Now you can filter the DataFrame
df_IND = df_IND[(df_IND.index >= datetime.strptime("2022-11-10", '%Y-%m-%d').date()) & (df_IND.index <= datetime.strptime("2023-02-28",'%Y-%m-%d').date())]

df_NL = df_NL[(df_NL.index >= datetime.strptime("2021-04-20", '%Y-%m-%d').date()) & (df_NL.index <= datetime.strptime("2021-09-30",'%Y-%m-%d').date())]

# %%

df_IND['Tmean'] = (df_IND['TMAX'] + df_IND['TMIN']) / 2
df_NL['Tmean'] = (df_NL['TMAX'] + df_NL['TMIN']) / 2
df_IND['daytimeTemp'] = ((df_IND['TMAX'] + df_IND['TMIN']) / 2 + df_IND['TMAX'])/2
df_NL['daytimeTemp'] = ((df_NL['TMAX'] + df_NL['TMIN']) / 2 + df_NL['TMAX'])/2
max_daytime_NL = df_NL[df_NL['daytimeTemp'] > 24]['daytimeTemp'].max()
df_NL[df_NL['daytimeTemp'] > 24].index[0] - datetime.strptime("2021-04-20", '%Y-%m-%d').date()
df_NL[df_NL['daytimeTemp'] == max_daytime_NL].index - datetime.strptime("2021-04-20", '%Y-%m-%d').date()
# %%
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
indicatation_text_x = 0
indicatation_text_y = 1.01
j_to_mj = 1000000
fig = plt.figure(figsize=(10, 10))

# Create a GridSpec for the whole figure
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.1, wspace=0.1)

# First subplot
ax1 = fig.add_subplot(gs[0, 1])
ax1.plot(df_IND.index, df_IND['TMAX'], label='Maximum Temperature', color='r')
ax1.plot(df_IND.index, df_IND['daytimeTemp'], label='Daytime Mean Temperature', color='black')
ax1.plot(df_IND.index, df_IND['Tmean'] , label='Mean Temperature', color='grey')
ax1.plot(df_IND.index, df_IND['TMIN'], label='Minimum Temperature', color='g')
ax1.xaxis.set_major_locator(mdates.MonthLocator())
ax1.set_ylabel('Temperature (°C)')
ax1.set_xticklabels('')

ax1.set_title('India')
ax1.text(indicatation_text_x, indicatation_text_y, 'b)', transform=ax1.transAxes, size=config.subplot_fs, weight='bold')

ax2 = fig.add_subplot(gs[0, 0], sharey=ax1)
ax2.plot(df_NL.index, df_NL['TMAX'], label='Maximum Temperature', color='r')
ax2.plot(df_NL.index, df_NL['daytimeTemp'], label='Daytime Mean Temperature', color='black')
ax2.plot(df_NL.index, df_NL['Tmean'] , label='Mean Temperature', color='grey')
ax2.plot(df_NL.index, df_NL['TMIN'], label='Minimum Temperature', color='g')
ax2.set_xticklabels('')
ax2.set_ylabel('Temperature (°C)')
ax2.legend()
ax2.set_title('The Netherlands')
ax2.text(indicatation_text_x, indicatation_text_y, 'a)', transform=ax2.transAxes, size=config.subplot_fs, weight='bold')


# Second subplot
# Third subplot
# ax3 = fig.add_subplot(gs[1, 1])
# ax3.plot(df_IND.index, df_IND['IRRAD']/j_to_mj, label='Irraditation', color='b')
# ax3.xaxis.set_major_locator(mdates.MonthLocator())
# ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
# # add day length 
# daylength_data.df_ind['daylength'].plot(label='Daylength', color='black', ax=ax3)
# # Create a second y-axis for the third subplot
# ax3b = ax3.twinx()
# ax3b.bar(df_IND.index, df_IND['RAIN'] * 10, label='Rain', color='r')
# ax3b.set_ylabel('Rainfall (mm)', color='r')

# ax3.text(indicatation_text_x, indicatation_text_y, 'd)', transform=ax3.transAxes, size=config.subplot_fs, weight='bold')
# ax3.legend()


# ax4 = fig.add_subplot(gs[1, 0], sharey=ax3)
# ax4.plot(df_NL.index, df_NL['IRRAD']/j_to_mj, label='IRRAD', color='b')
# daylength_data.df_nl['daylength'].plot(label='Daylength', color='black', ax=ax4)
# ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
# # ax4.set_ylabel('Irradiation', color='b')
# ax4b = ax4.twinx()
# ax4b.get_shared_y_axes().join(ax4b, ax3b)
# ax4b.bar(df_NL.index, df_NL['RAIN'] * 10, label='Rain', color='r')
# # Add letter indication
# ax4.text(indicatation_text_x, indicatation_text_y, 'c)', transform=ax4.transAxes, size=config.subplot_fs, weight='bold')
# ax4.set_ylabel('Irradiation (MJ/m²/day)', color='b')
# ax4b.legend()
# ax4b.set_yticklabels('')

plt.savefig(f'../output/weather_data.png', dpi = 300, bbox_inches='tight')
plt.savefig(f'../output/weather_data.svg', dpi = 600, bbox_inches='tight')
plt.show()
# %%
