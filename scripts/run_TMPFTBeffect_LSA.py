# %%
import numpy as np
import matplotlib.pyplot as plt
TEFFMX = 32  # Maximum effective temperature
TBASEM = 3  # Base temperature
TSUMEM = 170  # Effective temperature sum
Temp = np.linspace(0, 40)  # Daily temperatures
# Assuming drv.TEMP is a list of daily temperatures
# and p.TEFFMX, p.TBASEM, and p.TSUMEM are defined constants

def calculate_DVR_values(TEFFMX, TBASEM):
    DVR_values = []  # List to store DVR values

    for TEMP in Temp:
        DTSUME = max(0., min(TEFFMX - TBASEM, TEMP - TBASEM))
        DTSUM = 0.
        DVR = 0.1 * DTSUME / TSUMEM
        DVR_values.append(DVR)
    
    return DVR_values

# Calculate DVR_values for two different TEFFMX values
plt.figure(figsize=(10, 6))

# Iterate over the range from 18 to 32 with a step of 0.1
for TEFFMX in np.arange(18, 32.1, 1):
    # Calculate DVR_values for the current TEFFMX value
    DVR_values = calculate_DVR_values(TEFFMX, TBASEM)
    
    # Plot the DVR_values
    plt.plot(Temp, DVR_values, label=f'TEFFMX = {TEFFMX:.1f}')

plt.xlabel('Temperature')
plt.ylabel('DVR')
plt.title('Development rate over different temperatures for the phase of emergence.')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Move the legend outside the plot
plt.grid(True)
plt.show()
# %%
from scipy.interpolate import UnivariateSpline

# Generate some data
# Generate some data
Temp = np.linspace(0, 50, 100)
TBASEM = 10
TSUMEM = 20

# Calculate DVR_values for TEFFMX = 18 and TEFFMX = 32
TEFFMX1 = 18
TEFFMX2 = 32
DVR_values1 = calculate_DVR_values(TEFFMX1,TBASEM)
DVR_values2 = calculate_DVR_values(TEFFMX2,TBASEM)

# Fit a cubic spline to the data for TEFFMX = 18 and TEFFMX = 32
spline1 = UnivariateSpline(Temp, DVR_values1, k=3)
spline2 = UnivariateSpline(Temp, DVR_values2, k=3)

# Generate some x values for the fitted function
Temp_fit = np.linspace(0, 50, 1000)

# Calculate the y values for the fitted function
DVR_values_fit1 = spline1(Temp_fit)
DVR_values_fit2 = spline2(Temp_fit)

# Plot the original data and the fitted function
plt.figure(figsize=(10, 6))
plt.plot(Temp, DVR_values1, 'o', label='Original data (TEFFMX = 18)')
plt.plot(Temp_fit, DVR_values_fit1, label='Fitted function (TEFFMX = 18)')
plt.plot(Temp, DVR_values2, 'o', label='Original data (TEFFMX = 32)')
plt.plot(Temp_fit, DVR_values_fit2, label='Fitted function (TEFFMX = 32)')
plt.xlabel('Temperature')
plt.ylabel('DVR')
plt.title('Development rate over different temperatures for the phase of emergence.')
plt.legend()
plt.grid(True)
plt.show()
# %% # fix teffmx but varying tbasem 
# Fix TEFFMX
TEFFMX = 18

# Iterate over a range of TBASEM values
for TBASEM in np.arange(2, 10.1, 1):
    # Calculate DVR_values for the current TBASEM value
    DVR_values = calculate_DVR_values(TEFFMX, TBASEM)
    
    # Plot the DVR_values
    plt.plot(Temp, DVR_values, label=f'TBASEM = {TBASEM:.1f}')

plt.xlabel('Temperature')
plt.ylabel('DVR')
plt.title('Development rate over different temperatures for the phase of emergence.')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Move the legend outside the plot
plt.grid(True)
plt.show()

# Generate some data
Temp = np.linspace(0, 50, 100)
TSUMEM = 20

# Calculate DVR_values for TBASEM = 5 and TBASEM = 15
TBASEM1 = 2
TBASEM2 = 10
DVR_values1 = calculate_DVR_values(TEFFMX, TBASEM1)
DVR_values2 = calculate_DVR_values(TEFFMX, TBASEM2)

# Fit a cubic spline to the data for TBASEM = 5 and TBASEM = 15
spline1 = UnivariateSpline(Temp, DVR_values1, k=3)
spline2 = UnivariateSpline(Temp, DVR_values2, k=3)

# Generate some x values for the fitted function
Temp_fit = np.linspace(0, 50, 1000)

# Calculate the y values for the fitted function
DVR_values_fit1 = spline1(Temp_fit)
DVR_values_fit2 = spline2(Temp_fit)

# Plot the original data and the fitted function
plt.figure(figsize=(10, 6))
plt.plot(Temp, DVR_values1, 'o', label=f'Original data TBASEM = {TBASEM1}')
plt.plot(Temp_fit, DVR_values_fit1, label=f'Fitted function TBASEM = {TBASEM1}')
plt.plot(Temp, DVR_values2, 'o', label=f'Original data TBASEM = {TBASEM2}')
plt.plot(Temp_fit, DVR_values_fit2, label=f'Fitted function TBASEM = {TBASEM2}')
plt.xlabel('Temperature')
plt.ylabel('DVR')
plt.title('Development rate over different temperatures for the phase of emergence.')
plt.legend()
plt.grid(True)
plt.show()


# SPAN
# %%
TBASE = 3
FYSAGE = np.maximum(0., (Temp- TBASE)/(35. - TBASE))

plt.figure(figsize=(10, 6))
plt.plot(Temp, FYSAGE)
plt.xlabel('Temperature')
plt.ylabel('FYSAGE')
plt.title('Span of the development stage')
plt.grid(True)
plt.show()
# q10
# %%
def calculate_teff(Q10, temp):
    # Replace this with your actual function
    return Q10**((temp-25.)/10.)
# Define the temperature range
temp = np.linspace(0, 50, 500)

# Define the Q10 values to try
q10_values = [1.5, 2.0, 2.5, 3.0]

# Calculate and plot TEFF for each Q10 value
for Q10 in q10_values:
    teff = calculate_teff(Q10, temp)
    plt.plot(temp, teff, label=f'Q10={Q10}')

# Add a legend and show the plot
plt.legend()
plt.show()
# %% --------------------
# Temp effect on tmpftb
# ---------------------
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import config
from pcse.fileinput import ExcelWeatherDataProvider
# Define the data
temps = [0, 3, 10, 15, 24, 29, 36]
effeciency = [0.01, 0.01, 0.75, 1, 1, 0.75, 0.01]
tmpftb = pd.DataFrame({'Temperature': temps, 'TMPFTB': effeciency})

# Plot the data
plt.plot(tmpftb['Temperature'], tmpftb['TMPFTB'])
plt.show()

# load weather data
wdp = ExcelWeatherDataProvider('../data_raw/India2022_23.xlsx')
wdp_nl = ExcelWeatherDataProvider('../data_raw/350_weatherfile_2021.xlsx')
# %%
# Assuming weather is a DataFrame with the specified columns
weather = pd.DataFrame(wdp.export())
weather['Date'] = pd.to_datetime(weather['DAY'])
# weather= weather_nl[(weather_nl['Date'] >= pd.to_datetime("2021-04-20")) & (weather_nl['Date'] <= pd.to_datetime("2021-09-30"))]

weather['Temperature...C.'] = (((weather['TMIN'] + weather['TMAX']) / 2) + weather['TMAX'])/2
ws_temp = weather[['Date', 'Temperature...C.']]
weather_nl = pd.DataFrame(wdp_nl.export())

weather_nl['Date'] = pd.to_datetime(weather_nl['DAY'])
# Assuming df_NL is a DataFrame with a datetime index
weather_nl = weather_nl[(weather_nl['Date'] >= pd.to_datetime("2021-04-20")) & (weather_nl['Date'] <= pd.to_datetime("2021-09-30"))]
weather_nl['Temperature...C.'] = (((weather_nl['TMIN'] + weather_nl['TMAX']) / 2) + weather_nl['TMAX'])/2


# Interpolate
f = interpolate.interp1d(temps, effeciency, fill_value="extrapolate")

weather['TMPFTB'] = f(weather['Temperature...C.'])
weather_nl['TMPFTB'] = f(weather_nl['Temperature...C.'])
# Add DAP column
weather['DAP'] = np.arange(1, len(weather) + 1)
weather_nl['DAP'] = np.arange(1, len(weather_nl) + 1)
# %%
def plot_data(df, scenario):
    # Plot the data
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('DAP', fontsize=config.subplot_fs)
    ax1.set_ylabel('TMPFTB', color=color, fontsize=config.subplot_fs)
    ax1.plot(df['DAP'], df['TMPFTB'], color=color, linewidth=1)
    ax1.tick_params(axis='y', labelcolor=color)


    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    # we already handled the x-label with ax1
    ax2.set_ylabel('Temperature', color=color, fontsize=config.subplot_fs)  
    ax2.plot(df['DAP'], df['Temperature...C.'], color=color, linewidth=1)
    ax2.tick_params(axis='y', labelcolor=color)

    # Add horizontal lines
    ax2.axhline(15, color='green', linestyle='--')
    ax2.axhline(24, color='green', linestyle='--')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    ax1.set_ylim([0, 1.1])  # Set limits for first y-axis
    ax2.set_ylim(0, 35)  # Set limits for second y-axis
    plt.savefig(f'../output/{scenario}tmpftb.svg', dpi = 600, bbox_inches='tight')
    plt.savefig(f'../output/{scenario}tmpftb.png', dpi = 300, bbox_inches='tight')
    # plt.savefig(f'../output/weather_data.svg', dpi = 600, bbox_inches='tight')
    plt.show()

# Call the function with the dataframes
plot_data(weather, 'India')
plot_data(weather_nl, 'Netherlands')

# %%
# Calculate the number of days with temperature below 15 or above 24 in weather DataFrame
days_below_15_or_above_24_weather = weather[(weather['Temperature...C.'] < 15) | (weather['Temperature...C.'] > 24)].shape[0]

# Calculate the number of days with temperature below 15 or above 24 in weather_nl DataFrame
days_below_15_or_above_24_weather_nl = weather_nl[(weather_nl['Temperature...C.'] < 15) | (weather_nl['Temperature...C.'] > 24)].shape[0]

print(f"Number of days with temperature below 15 or above 24 in weather DataFrame: {days_below_15_or_above_24_weather}")
print(f"Number of days with temperature below 15 or above 24 in weather_nl DataFrame: {days_below_15_or_above_24_weather_nl}")

# Calculate the number of rows in the first half of the DataFrame
half_length = len(weather) // 2

# Slice the DataFrame to only include the first half of the rows
first_half_weather = weather.iloc[:half_length]

# Calculate the number of days with temperature below 15 or above 24 in the first half of the DataFrame
days_below_15_or_above_24_first_half_weather = first_half_weather[(first_half_weather['Temperature...C.'] < 15) | (first_half_weather['Temperature...C.'] > 24)].shape[0]
emergence = 8
print(f"Number of days with temperature below 15 or above 24 in the first half of weather DataFrame: {days_below_15_or_above_24_first_half_weather - emergence}")
first_half_weather_nl = weather_nl.iloc[:half_length]
days_below_15_or_above_24_first_half_weather_nl = first_half_weather_nl[(first_half_weather_nl['Temperature...C.'] < 15) | (first_half_weather_nl['Temperature...C.'] > 24)].shape[0]
emergence_nl = 24
print(f"Number of days with temperature below 15 or above 24 in the first half of weather_nl DataFrame: {days_below_15_or_above_24_first_half_weather_nl - emergence_nl}")
# %%
