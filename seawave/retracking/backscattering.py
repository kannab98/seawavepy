import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

name = 'aver3'
df = pd.read_csv('seawave/retracking/%s.dat' % name, sep='\s+', header=None)

theta = np.deg2rad(df.iloc[:, 0])
sigma = df.iloc[:, 1]


def cross_section(theta, xx, F):
    # theta = Surface.angle_correction(theta)
    # Коэффициент Френеля
    # F = 0.8



    sigma = F**2/(2*np.cos(theta)**4 * np.sqrt(xx))
    sigma *= np.exp(- np.tan(theta)**2 /(2*xx))
    return sigma


def line(theta, xx, F): 
    return 10*np.log10(cross_section(theta, xx, F))


# cond = (np.deg2rad(4) < theta) & (theta < np.deg2rad(9))
# cond = (np.deg2rad(3) < np.abs(theta)) & (np.abs(theta) < np.deg2rad(9))
cond = (np.deg2rad(3) < theta) & (theta < np.deg2rad(9))

popt = curve_fit(line,
                xdata=theta.values[cond],
                ydata=sigma.values[cond],
                p0=[1, 1],
                bounds=( (0, 0), (np.inf, np.inf) )
)[0]


theta0 = np.rad2deg(theta)
plt.plot(theta0, sigma, label="experiment")
plt.plot(theta0[cond], sigma[cond], '.', label="choosen dots")
plt.plot(theta0, line(theta, *popt), label="approximation")
plt.legend()
plt.savefig(name)
