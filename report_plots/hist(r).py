import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import scienceplots
plt.style.use(['science'])

data = """
particle id:  1 displacement:  50.04750388189237 frames tracked: 354
particle id:  2 displacement:  39.99518796839765 frames tracked: 215
particle id:  3 displacement:  33.83236142825319 frames tracked: 32
particle id:  4 displacement:  51.259102358735724 frames tracked: 284
particle id:  5 displacement:  19.520839017365468 frames tracked: 66
particle id:  6 displacement:  10.825545894192167 frames tracked: 14
particle id:  7 displacement:  45.850149739537365 frames tracked: 213
particle id:  8 displacement:  3.535201792133838 frames tracked: 5
particle id:  9 displacement:  49.59837840190636 frames tracked: 429
particle id:  10 displacement:  14.80438304912209 frames tracked: 32
particle id:  12 displacement:  15.499248801763283 frames tracked: 28
particle id:  13 displacement:  39.74239608252959 frames tracked: 32
particle id:  16 displacement:  2.6332479167818805 frames tracked: 3
particle id:  17 displacement:  7.02169993663932 frames tracked: 3
particle id:  18 displacement:  16.590814384326013 frames tracked: 253
particle id:  19 displacement:  2.0495848875298837 frames tracked: 3
particle id:  20 displacement:  16.430413345872726 frames tracked: 11
particle id:  21 displacement:  17.559310758020743 frames tracked: 4
particle id:  23 displacement:  11.202459076784528 frames tracked: 11
particle id:  24 displacement:  13.613926150208217 frames tracked: 147
particle id:  25 displacement:  35.371659036855505 frames tracked: 155
particle id:  26 displacement:  10.146604148444196 frames tracked: 285
particle id:  27 displacement:  14.656616541868448 frames tracked: 107
particle id:  28 displacement:  27.605168208794296 frames tracked: 97
particle id:  29 displacement:  6.3837620409330915 frames tracked: 8
particle id:  30 displacement:  24.156073847507177 frames tracked: 355
particle id:  33 displacement:  27.499833398010868 frames tracked: 16
particle id:  34 displacement:  3.1411586185251577 frames tracked: 3
particle id:  35 displacement:  92.95639716915286 frames tracked: 296
particle id:  37 displacement:  14.41693807104825 frames tracked: 57
particle id:  39 displacement:  5.751206398437785 frames tracked: 9
particle id:  40 displacement:  36.037194507918635 frames tracked: 299
particle id:  41 displacement:  3.101500351373304 frames tracked: 19
particle id:  42 displacement:  23.612924199524628 frames tracked: 113
particle id:  43 displacement:  31.877525866462985 frames tracked: 269
particle id:  44 displacement:  9.03265289197962 frames tracked: 17
particle id:  45 displacement:  20.67682553450354 frames tracked: 11
particle id:  46 displacement:  29.01556929884429 frames tracked: 247
particle id:  47 displacement:  26.35692614717777 frames tracked: 94
particle id:  48 displacement:  18.111420109035077 frames tracked: 76
particle id:  49 displacement:  15.829480353711324 frames tracked: 107
particle id:  50 displacement:  9.89060499617134 frames tracked: 3
particle id:  51 displacement:  12.390245017043489 frames tracked: 41
particle id:  54 displacement:  43.69788619348539 frames tracked: 194
particle id:  55 displacement:  10.79914342817423 frames tracked: 33
particle id:  56 displacement:  23.37276689102107 frames tracked: 23
particle id:  57 displacement:  44.59014335996181 frames tracked: 204
particle id:  58 displacement:  5.633815877066715 frames tracked: 15
particle id:  60 displacement:  41.33172531188655 frames tracked: 142
particle id:  61 displacement:  2.4281411578218894 frames tracked: 3
particle id:  63 displacement:  5.431351402617919 frames tracked: 3
particle id:  65 displacement:  11.719269418782256 frames tracked: 3
particle id:  66 displacement:  35.56593528528171 frames tracked: 7
particle id:  67 displacement:  13.49188568800506 frames tracked: 3
particle id:  68 displacement:  20.669659139885855 frames tracked: 29
particle id:  69 displacement:  48.97089743820026 frames tracked: 155
particle id:  70 displacement:  18.432334121652694 frames tracked: 23
particle id:  71 displacement:  7.099852415395441 frames tracked: 10
particle id:  73 displacement:  23.58398293505046 frames tracked: 89
particle id:  74 displacement:  13.85277985560858 frames tracked: 5
particle id:  75 displacement:  7.088501155646644 frames tracked: 4
particle id:  76 displacement:  12.731616647639415 frames tracked: 123
particle id:  79 displacement:  49.30898509723738 frames tracked: 125
particle id:  80 displacement:  8.854838045982147 frames tracked: 10
particle id:  81 displacement:  4.149301494634531 frames tracked: 3
particle id:  82 displacement:  41.99743164680387 frames tracked: 55
particle id:  83 displacement:  6.307406453926774 frames tracked: 45
particle id:  84 displacement:  16.181215935714253 frames tracked: 7
particle id:  85 displacement:  48.37953813361197 frames tracked: 68
particle id:  86 displacement:  29.22116662584622 frames tracked: 74
particle id:  87 displacement:  52.71914589295276 frames tracked: 72
particle id:  88 displacement:  3.97170169311629 frames tracked: 58
particle id:  89 displacement:  8.68559977277496 frames tracked: 52
particle id:  90 displacement:  7.561505311461776 frames tracked: 34
particle id:  91 displacement:  21.78161388878972 frames tracked: 32
particle id:  92 displacement:  7.573556905782643 frames tracked: 30
"""
lines = data.strip().split('\n')
displacements = []

for line in lines:
    parts = line.split()
    displacement = float(parts[4]) * 10**(-7)
    displacements.append(displacement)


hist, edges, _ = plt.hist(displacements, bins=16,
                          edgecolor='black', alpha=0.7, label='Data')


def full_gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))


A_guess = max(hist)
mu_guess = 0.0
sigma_guess = np.std(displacements)


params, covariance = curve_fit(
    full_gaussian, edges[:-1], hist, p0=[A_guess, mu_guess, sigma_guess])


x_fit = np.linspace(min(displacements), max(displacements), 10000)


y_fit = full_gaussian(x_fit, *params)


plt.plot(x_fit, y_fit, 'r-', label='Fitted Gaussian')


peak_x = x_fit[np.argmax(y_fit)]


plt.axvline(peak_x, color='green', linestyle='--', label='Peak')


plt.text(peak_x, max(y_fit), f'{peak_x:.4e}',
         rotation=0, ha='right', va='bottom')


plt.title('Histogram and Gaussian Fit of Displacement')
plt.xlabel(r'Displacement in xy axis ($\mu m$)')
plt.ylabel('Frequency')
plt.grid(True)
plt.legend()


plt.show()
