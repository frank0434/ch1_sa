# This script is used to manually explore the WOFOST model and its parameters.
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
