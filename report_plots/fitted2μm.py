import numpy as np
import matplotlib.pyplot as plt
import uncertainties.unumpy as unp
from uncertainties import ufloat
import scienceplots

plt.style.use(['science'])

time = np.array([1, 2, 3, 4, 5])


x_mean_1 = ufloat(-0.353937, 0.059846)
sigma_x_1 = ufloat(1.811392, 0.059846)
y_mean_1 = ufloat(-0.214056, 0.063665)
sigma_y_1 = ufloat(1.837906, 0.063665)


y_mean_2 = ufloat(-0.272469, 0.090336)
sigma_y_2 = ufloat(2.335956, 0.090336)
x_mean_2 = ufloat(-0.278456, 0.094011)
sigma_x_2 = ufloat(2.250005, 0.094011)


y_mean_3 = ufloat(-0.347571, 0.100487)
sigma_y_3 = ufloat(2.555779, 0.100487)
x_mean_3 = ufloat(-0.356089, 0.103635)
sigma_x_3 = ufloat(2.476218, 0.103636)


y_mean_4 = ufloat(-0.358002, 0.113240)
sigma_y_4 = ufloat(2.699782, 0.113240)
x_mean_4 = ufloat(-0.283161, 0.114822)
sigma_x_4 = ufloat(2.564491, 0.114822)


y_mean_5 = ufloat(-0.366347, 0.119686)
sigma_y_5 = ufloat(2.772138, 0.119686)
x_mean_5 = ufloat(-0.353736, 0.111959)
sigma_x_5 = ufloat(2.562899, 0.111959)


mean_values_y = np.array([y_mean_1, y_mean_2, y_mean_3, y_mean_4, y_mean_5])
sigma_values_y = np.array([sigma_y_1, sigma_y_2, sigma_y_3, sigma_y_4, sigma_y_5])
mean_values_x = np.array([x_mean_1, x_mean_2, x_mean_3, x_mean_4, x_mean_5])
sigma_values_x = np.array([sigma_x_1, sigma_x_2, sigma_x_3, sigma_x_4, sigma_x_5])


sigma_squared = (sigma_values_x**2 + sigma_values_y**2)/10


coefficients = np.polyfit(time, unp.nominal_values(sigma_squared), 1, cov=True)
fit_slope = ufloat(coefficients[0][0], np.sqrt(coefficients[1][0, 0]))
fit_intercept = ufloat(coefficients[0][1], np.sqrt(coefficients[1][1, 1]))

fit_line = fit_slope * time + fit_intercept
fit_band_upper = unp.nominal_values(fit_line) + np.sqrt(coefficients[1][0, 0]) * time + np.sqrt(coefficients[1][1, 1])
fit_band_lower = unp.nominal_values(fit_line) - np.sqrt(coefficients[1][0, 0]) * time - np.sqrt(coefficients[1][1, 1])

plt.errorbar(time, unp.nominal_values(sigma_squared), yerr=unp.std_devs(sigma_squared), fmt='o', color='b', label='Data with Error Bars')
plt.plot(time, unp.nominal_values(fit_line), color='r', label='Linear Fit')
plt.fill_between(time, fit_band_lower, fit_band_upper, color='g', alpha=0.3, label='Error Propagation Band')
plt.xlim(0, 5.5)
plt.xlabel('Time (sec)')
plt.ylabel(r'$\sigma^2 (\times 10^{-12}m^2)$')
plt.title(r'$\sigma^2 (\times 10^{-12}m^2)$ vs Time')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

print(f'Fitted Line: y = {fit_slope:.14u} * t + {fit_intercept:.4u}')
