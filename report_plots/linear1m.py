import numpy as np
import matplotlib.pyplot as plt
import uncertainties.unumpy as unp
from uncertainties import ufloat
import scienceplots

plt.style.use(['science'])

time = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
sigma_x = unp.uarray([0.503130, 0.547280, 0.580346, 0.672302, 0.759587, 0.796655, 0.868040, 0.929550, 1.002161, 1.080240], [
                     0.008598, 0.011707, 0.017054, 0.028192, 0.032641, 0.044864, 0.048306, 0.052652, 0.081813, 0.075654])
sigma_y = unp.uarray([0.635809, 0.687667, 0.767308, 0.888766, 0.893932, 0.999166, 1.025778, 0.949115, 1.043321, 1.190555], [
                     0.010357, 0.011707, 0.017054, 0.028192, 0.032641, 0.044864, 0.048306, 0.052652, 0.081813, 0.086413])


# calculate σ^2 = σ_x^2 + σ_y^2
sigma_squared = sigma_x**2 + sigma_y**2
coefficients = np.polyfit(time, unp.nominal_values(sigma_squared), 1, cov=True)
fit_slope = ufloat(coefficients[0][0], np.sqrt(coefficients[1][0, 0]))
fit_intercept = ufloat(coefficients[0][1], np.sqrt(coefficients[1][1, 1]))

# Calculate the propagated uncertainties for the entire time range
fit_line = fit_slope * time + fit_intercept
fit_band_upper = unp.nominal_values(
    fit_line) + np.sqrt(coefficients[1][0, 0]) * time + np.sqrt(coefficients[1][1, 1])
fit_band_lower = unp.nominal_values(
    fit_line) - np.sqrt(coefficients[1][0, 0]) * time - np.sqrt(coefficients[1][1, 1])


# create scatter plot with error bars and error propagation band
plt.errorbar(time, unp.nominal_values(sigma_squared), yerr=unp.std_devs(
    sigma_squared), fmt='.', color='b', label='Data with Error Bars')
plt.plot(time, unp.nominal_values(fit_line), color='r', label='Linear Fit')
plt.fill_between(time, fit_band_lower, fit_band_upper,
                 color='g', alpha=0.3, label='Error Propagation Band')
plt.xlim(0, 10.5)
plt.xlabel('Time (sec)')
plt.ylabel(r'$\sigma^2 (\times 10^{-12}m^2)$')
plt.title(r'$\sigma^2 (\times 10^{-12}m^2)$ vs Time ')
plt.legend(loc='upper left')
plt.grid(True)
plt.savefig('/Users/kostis/Desktop/sl1.png', dpi=900)
plt.show()

print(f'Fit: {fit_slope:.4u} * t + {fit_intercept:.4u}')
print(f'Fitted Line: y = {fit_slope:.4u} * t + {fit_intercept:.4u}')
