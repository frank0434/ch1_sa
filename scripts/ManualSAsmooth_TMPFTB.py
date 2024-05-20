#%%

import numpy as np
import matplotlib.pyplot as plt
def combined_equation(Temp_daytime, tm1, t1, t2, te):
    return np.where(Temp_daytime < t1, 
                    1 * (1 + (t1 - Temp_daytime) / (t1 - tm1)) * (Temp_daytime / t1) ** (t1 / (t1 - tm1)),
                    np.where((Temp_daytime >= t1) & (Temp_daytime <= t2), 1, 
                             1 * ((te - Temp_daytime)/(te - t2))*((Temp_daytime + t1 - t2)/t1)**(t1/(te - t2))))
from scipy.optimize import curve_fit


AGFUN_TMPFTB = [0.0,0.01,3.0,0.01,10.0,0.75, 11, 0.8,  12, 0.85, 13, 0.9, 
                14, 0.95, 15.0,1.0,24.0,1.0,29.0,0.75,36.0,0.01]
x = AGFUN_TMPFTB[::2]  # Values in odd positions
y = AGFUN_TMPFTB[1::2]  # Values in even positions
plt.plot(x, y)
plt.xlabel('Temperature (°C)')
plt.ylabel('Reduction factor of \nMaximum leaf CO2 assimilation rate')

# Initial guess: [tm1, t1, t2, te, 1]
p0 = [0, 10, 20, 30]

# Fit the curve
popt, pcov = curve_fit(combined_equation, x, y, p0, method='dogbox', maxfev=5000)

# Generate y data from the combined_equation function with the optimized parameters
x_smooth = np.linspace(min(x), max(x), 100)
y_smooth = combined_equation(x_smooth, *popt)

# Plot the original data and the fitted curve
plt.plot(x, y, 'o', label='Data points sampled from AGFUN_TMPFTB', c = 'blue')
plt.plot(x_smooth, y_smooth, label='Fitted curve', c = 'red')
plt.xlabel('Mean Daytime Temperature (°C)')
plt.ylabel('Reduction factor of \nMaximum leaf CO2 assimilation rate')
plt.legend()

plt.savefig('../output/TMPFTB_curving_fitting.svg')
plt.show()
plt.close()

