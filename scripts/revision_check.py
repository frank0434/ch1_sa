# %%
import pandas as pd
import matplotlib.pyplot as plt
import config
from pcse.fileinput import ExcelWeatherDataProvider
from pcse.fileinput import NASAPowerWeatherDataProvider
from scipy import stats

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
# Now you can filter the DataFrame
df_IND = df_IND[(df_IND.index >= datetime.strptime("2022-11-10", '%Y-%m-%d').date()) & (df_IND.index <= datetime.strptime("2023-02-28",'%Y-%m-%d').date())]

df_NL = df_NL[(df_NL.index >= datetime.strptime("2021-04-20", '%Y-%m-%d').date()) & (df_NL.index <= datetime.strptime("2021-09-30",'%Y-%m-%d').date())]
df_IND['Tmean'] = (df_IND['TMAX'] + df_IND['TMIN']) / 2
df_NL['Tmean'] = (df_NL['TMAX'] + df_NL['TMIN']) / 2
df_IND['daytimeTemp'] = ((df_IND['TMAX'] + df_IND['TMIN']) / 2 + df_IND['TMAX'])/2
df_NL['daytimeTemp'] = ((df_NL['TMAX'] + df_NL['TMIN']) / 2 + df_NL['TMAX'])/2
max_daytime_NL = df_NL[df_NL['daytimeTemp'] > 24]['daytimeTemp'].max()
df_NL[df_NL['daytimeTemp'] > 24].index[0] - datetime.strptime("2021-04-20", '%Y-%m-%d').date()
df_NL[df_NL['daytimeTemp'] == max_daytime_NL].index - datetime.strptime("2021-04-20", '%Y-%m-%d').date()

# %%  REVIWER 1 COMMENT 2
Longterm_NL = pd.read_csv('../data_raw/NL91to23TemperatureAverage.csv')
Longterm_IND = pd.read_csv('../data_raw/indian91to23TemperatureAverage.csv')
Longterm_NL['Month'] = range(1, 13)
Longterm_IND['Month'] = range(1, 13)

# Filter Netherlands data for April (4) to October (10)
Longterm_NL_filtered = Longterm_NL[Longterm_NL['Month'].isin([4, 5, 6, 7, 8, 9, 10])]

# Filter India data for months 11, 12, 1, 2 and reorder them
Longterm_IND_filtered = Longterm_IND[Longterm_IND['Month'].isin([11, 12, 1, 2])]
# Sort to ensure the order is 11, 12, 1, 2
month_order = [11, 12, 1, 2]
Longterm_IND_filtered = Longterm_IND_filtered.set_index('Month').loc[month_order].reset_index()

# Calculate monthly statistics for Netherlands data
df_NL_monthly = df_NL.groupby(pd.to_datetime(df_NL.index).month).agg({
    'Tmean': ['mean', 'max', 'min'],
    'TMAX': ['mean', 'max', 'min'],
    'TMIN': ['mean', 'max', 'min']
}).round(2)

# Calculate monthly statistics for India data
df_IND_monthly = df_IND.groupby(pd.to_datetime(df_IND.index).month).agg({
    'Tmean': ['mean', 'max', 'min'],
    'TMAX': ['mean', 'max', 'min'],
    'TMIN': ['mean', 'max', 'min']
}).round(2)

# %% Plot the monthly temperature data for Netherlands and India
import matplotlib.pyplot as plt

# Define colors for easy modification
colors = {
    'tmean': 'blue',
    'tmax': 'red', 
    'tmin': 'green'
}

# Set the figure size
# Create separate subplots for Netherlands and India
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 6), sharey=True)

# Plot Netherlands data
ax1.plot(Longterm_NL_filtered['Month'], Longterm_NL_filtered['Average Mean Surface Air Temperature'],
    marker='o', label='Tmean (1991-2023)', color=colors['tmean'])
ax1.plot(Longterm_NL_filtered['Month'], Longterm_NL_filtered['Average Maximum Surface Air Temperature'],
    marker='o', label='TMAX (1991-2023)', color=colors['tmax'])
ax1.plot(Longterm_NL_filtered['Month'], Longterm_NL_filtered['Average Minimum Surface Air Temperature'],
    marker='o', label='TMIN (1991-2023)', color=colors['tmin'])

# Add 2021 data for Netherlands
months_nl = df_NL_monthly.index
ax1.plot(months_nl, df_NL_monthly[('Tmean', 'mean')], 
    marker='s', linestyle='--', label='Tmean (2021)', color=colors['tmean'], alpha=0.7)
ax1.plot(months_nl, df_NL_monthly[('TMAX', 'mean')], 
    marker='s', linestyle='--', label='TMAX (2021)', color=colors['tmax'], alpha=0.7)
ax1.plot(months_nl, df_NL_monthly[('TMIN', 'mean')], 
    marker='s', linestyle='--', label='TMIN (2021)', color=colors['tmin'], alpha=0.7)

ax1.set_xticks([4, 5, 6, 7, 8, 9, 10])
ax1.set_xticklabels(['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct'])
ax1.set_xlabel('Month')
ax1.set_ylabel('Temperature (°C)')
ax1.set_title('The Netherlands - Monthly Temperature')
ax1.legend()
ax1.grid(True)
ax1.set_ylim(0, 35)  # Set y-axis limit for better comparison

# Plot India data - reorder x-axis to show continuous timeline from Nov to Feb
india_months = [11, 12, 1, 2]
india_labels = ['Nov', 'Dec', 'Jan', 'Feb']
ax2.plot(range(len(india_months)), Longterm_IND_filtered['Average Mean Surface Air Temperature'],
    marker='o', label='Tmean (1991-2023)', color=colors['tmean'])
ax2.plot(range(len(india_months)), Longterm_IND_filtered['Average Maximum Surface Air Temperature'],
    marker='o', label='TMAX (1991-2023)', color=colors['tmax'])
ax2.plot(range(len(india_months)), Longterm_IND_filtered['Average Minimum Surface Air Temperature'],
    marker='o', label='TMIN (1991-2023)', color=colors['tmin'])

# Add 2022-2023 data for India
months_ind = df_IND_monthly.index
# Reorder to match Nov, Dec, Jan, Feb sequence
if 11 in months_ind and 12 in months_ind and 1 in months_ind and 2 in months_ind:
    reordered_months = [11, 12, 1, 2]
    reordered_data_tmean = [df_IND_monthly.loc[m, ('Tmean', 'mean')] for m in reordered_months]
    reordered_data_tmax = [df_IND_monthly.loc[m, ('TMAX', 'mean')] for m in reordered_months]
    reordered_data_tmin = [df_IND_monthly.loc[m, ('TMIN', 'mean')] for m in reordered_months]
    
    ax2.plot(range(len(india_months)), reordered_data_tmean, 
        marker='s', linestyle='--', label='Tmean (2022-23)', color=colors['tmean'], alpha=0.7)
    ax2.plot(range(len(india_months)), reordered_data_tmax, 
        marker='s', linestyle='--', label='TMAX (2022-23)', color=colors['tmax'], alpha=0.7)
    ax2.plot(range(len(india_months)), reordered_data_tmin, 
        marker='s', linestyle='--', label='TMIN (2022-23)', color=colors['tmin'], alpha=0.7)

ax2.set_xticks(range(len(india_months)))
ax2.set_xticklabels(india_labels)
ax2.set_xlabel('Month')
# ax2.set_ylabel('Temperature (°C)')
ax2.set_title('India - Monthly Temperature')
ax2.legend()
ax2.grid(True)
ax2.set_ylim(0, 35)  # Set y-axis limit for better comparison
plt.tight_layout()
plt.savefig('../manuscript/FigS1.2.png', dpi=600, bbox_inches='tight')
plt.show()

# %% 
# wdp = NASAPowerWeatherDataProvider(latitude=52, longitude=5)
# print(wdp)
wdp = NASAPowerWeatherDataProvider(51.54, 5.86)
wdp_IND = NASAPowerWeatherDataProvider(23.84, 73.13)  # Coordinates for India
print(wdp)
df_nl_NASA = pd.DataFrame(wdp.export())
# filter the data from 2021-01-01 to 2021-10-11
df_nl_NASA['DAY'] = pd.to_datetime(df_nl_NASA['DAY'])
df_nl_NASA_season1 = df_nl_NASA[(df_nl_NASA['DAY'] >= '2021-01-01') & 
                        (df_nl_NASA['DAY'] <= '2021-10-11')]
df_nl_NASA_Season2 = df_nl_NASA[(df_nl_NASA['DAY'] >= "2022-04-20") & 
                                (df_nl_NASA['DAY'] <= "2022-10-11")].copy()
df_IND_NASA = pd.DataFrame(wdp_IND.export())
# Filter the data for India from 2022-11-10 to 2023-02-28
df_IND_NASA['DAY'] = pd.to_datetime(df_IND_NASA['DAY'])
df_IND_NASA = df_IND_NASA[(df_IND_NASA['DAY'] >= '2023-11-10') & 
                          (df_IND_NASA['DAY'] <= '2024-02-28')].copy()
# Output Season2 data to Excel
# df_nl_NASA_Season2.to_excel('../data_raw/nl_NASA_Season2.xlsx', index=False)
# df_IND_NASA.to_excel('../data_raw/ind_NASA_Season2.xlsx', index=False)
# plot the data
# Filter NASA data to match the same date range as df_NL
# df_nl_NASA = df[(df.index >= datetime.strptime("2021-04-20", '%Y-%m-%d')) & 
                # (df.index <= datetime.strptime("2021-09-30",'%Y-%m-%d'))].copy()
 
# compare with the original data
df_nl_orig = pd.DataFrame(wdp_NL.export())
j_to_mj = 1000000
# %% plot the data in each column as x and y pairs
import matplotlib.pyplot as plt
plt.figure(figsize=(5, 5))
plt.scatter(df_nl_orig['IRRAD'], df_nl_NASA_season1['IRRAD'], alpha=0.6)
plt.xlabel('Weather Station measured solar radiation')
plt.ylabel('NASA\'s solar radiation')
plt.title('NASA vs Weather Station IRRAD Comparison in the Netherlands')

# Add 1:1 line
min_val = min(df_nl_orig['IRRAD'].min(), df_nl_NASA_season1['IRRAD'].min())
max_val = max(df_nl_orig['IRRAD'].max(), df_nl_NASA_season1['IRRAD'].max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 line')

# Calculate R² and linear regression equation
slope, intercept, r_value, p_value, std_err = stats.linregress(df_nl_orig['IRRAD'], df_nl_NASA_season1['IRRAD'])
r_squared = r_value**2

# Add R² and equation as text on the plot
plt.text(0.05, 0.95, f'R² = {r_squared:.2f}\ny = {slope:.2f}x + {intercept:.2f}', 
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.xlabel('Weather Station solar radiation (J m⁻² day⁻¹)')
plt.ylabel('NASA\'s solar radiation (J m⁻² day⁻¹)')
plt.legend()
# plt.grid(True, alpha=0.3)
plt.savefig(f'../manuscript/FigS1.1.png', dpi=600, bbox_inches='tight', pad_inches=0.1)

plt.show()

#%% plot the radiation data in the Netherlands and India


# %%
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
indicatation_text_x = 0
indicatation_text_y = 1.01

fig = plt.figure(figsize=(10, 10))

# Create a GridSpec for the whole figure
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.1, wspace=0.1, width_ratios=[6,4])

# First subplot
ax1 = fig.add_subplot(gs[0, 1])
ax1.plot(df_IND.index, df_IND['IRRAD'], label='Solar Radiation', color='blue')
ax1.xaxis.set_major_locator(mdates.MonthLocator())
ax1.set_ylabel('Solar Radiation (kJ m⁻² day⁻¹)')
# ax1.set_xticklabels('')

ax1.set_title('India')
ax1.text(indicatation_text_x, indicatation_text_y, 'b)', transform=ax1.transAxes, size=config.subplot_fs, weight='bold')

ax2 = fig.add_subplot(gs[0, 0], sharey=ax1)
ax2.plot(df_NL.index, df_NL['IRRAD'], label='Solar Radiation', color='orange')
# ax2.set_xticklabels('')
ax2.set_ylabel('Solar Radiation (kJ m⁻² day⁻¹)')
ax2.legend()
ax2.set_title('The Netherlands')
ax2.text(indicatation_text_x, indicatation_text_y, 'a)', transform=ax2.transAxes, size=config.subplot_fs, weight='bold')
plt.show()
# %% calculate the cummulative radiation
df_IND_cum_rad = df_IND['IRRAD'].sum() / j_to_mj
df_NL_cum_rad = df_NL['IRRAD'].sum() / j_to_mj
print(f'Cumulative radiation in The Netherlands: {df_NL_cum_rad:.2f} MJ m⁻², India: '
      f'{df_IND_cum_rad:.2f} MJ m⁻²')
# how many days are there in the Netherlands and India
days_NL = (df_NL.index[-1] - df_NL.index[0]).days + 1
days_IND = (df_IND.index[-1] - df_IND.index[0]).days + 1
# print the number of days
print(f'The Netherlands: {days_NL} days, India: {days_IND} days')
# calculate the average radiation per day
avg_rad_NL = df_NL_cum_rad / days_NL
avg_rad_IND = df_IND_cum_rad / days_IND
print(f'Average radiation in The Netherlands: {avg_rad_NL:.2f} MJ m⁻² day⁻¹, India: '
      f'{avg_rad_IND:.2f} MJ m⁻² day⁻¹')
# if the NL had only 107 days as in india, what would be cummulative radiation
cum_rad_NL_107_days = avg_rad_NL * 107
print(f'If the Netherlands had only 107 days, the cumulative radiation would be: ' 
        f'{cum_rad_NL_107_days:.2f} MJ m⁻²')
# %% what if the Netherlands had only 107 days as a growing season like India?
df_NL_107_days = df_NL[df_NL.index <= df_NL.index[0] + pd.Timedelta(days=106)]
df_NL_107_days_cum_rad = df_NL_107_days['IRRAD'].sum() / j_to_mj
print(f'Cumulative radiation in The Netherlands with 107 days: {df_NL_107_days_cum_rad:.2f} MJ m⁻²')

