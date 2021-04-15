import numpy as np
from scipy.special import erf
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import sys
sys.path.append(".")

import pandas as pd
from seawave import config
from seawave.spectrum import spectrum
from seawave.retracking import retracking



df = pd.read_csv("tests/IMPULSAK.DAT", sep="\s+")
# print(df)


varslopesx = 0.012
varslopesy = 0.018
H = 50
delta = np.deg2rad(30)

slopes_coeffx = 1/(2*varslopesx*H**2) + 5.52/(delta**2*H**2)
# print(slopes_coeffx)
slopes_coeffy = 1/(2*varslopesy*H**2) + 5.52/(delta**2*H**2)
varelev = (2/4)**2
# # print(varelev)


t = df['TEK'].values
t = np.sort(np.append(t, 0))

t0 = 0
# t += t0
F = retracking.full_pulse(t, slopes_coeff=slopes_coeffx, sigma0=1, t0=t0, varelev=varelev, H=H, t_pulse=50e-6, c=1500)
popt = retracking.pulse(t, F)
print(popt[1]/50/1500, 50, delta)
# print(retracking.varslopes(popt[1]/50/1500, 50, delta))
popt = retracking.pulse(t, F)
# print(popt)



# # plt.plot(df['TEK'], df['IMP1_NEW']/df['IMP1_NEW'].max())
# # plt.plot(df['TEK'], df['IMP1']/df['IMP1'].max())
# # F = F/F.max()
# plt.plot(t, F/F.max())

# # print(t[np.where(F==1)]-4*50e-6)
# plt.savefig('kek')