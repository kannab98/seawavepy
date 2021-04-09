import sys
sys.path.append(".")
import math


import numpy as np
from seawave import config
from numba import cuda
import xarray as xr
from seawave.radar import radar
import seawave.surface as srf
from seawave.surface  import dataset, surface,wind
from seawave.spectrum import spectrum
from seawave.spectrum.module import dispersion
import matplotlib.pyplot as plt





config["Surface"]["LimitsOfModeling"] = [[-15, 15], [-15, 15]]
config['Surface']['GridSize'] = [1, 1]
config['Surface']['WaveNumberSize'] = 1024*2
config['Surface']['AzimuthSize'] = 128
config['Surface']['ThreadPerBlock'] = [8, 8, 8]


x = np.linspace(0,1,1)
y = np.linspace(0,1,1)
t = np.arange(0, 1, 1, dtype=np.float32)

ds = surface((x,y,t), config=config)
k, A = ds.spectrum()

A = np.abs(A)
k = ds.k.values

S = np.sum(A**2/2, axis=1)
S0 = spectrum(k)


print(np.trapz(S0, k), np.sum(A**2/2), np.sum(S))
print(k.max())
S = S/k*(87.13-0.07)

plt.loglog(k, S)
plt.loglog(k, S0)
# plt.xlim([0.01, 1000])
# plt.ylim([1e-30, 1e-1])
# # plt.loglog(k, S0)
print(S0.max()/S.max())


plt.savefig('kek6')