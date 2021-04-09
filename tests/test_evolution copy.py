import sys
sys.path.append(".")
import math


import numpy as np
from seawave import rc, kernel
from numba import cuda
import xarray as xr
from seawave.radar import radar
import seawave.surface as srf
from seawave.surface  import dataset, surface, rough_surface
from seawave.spectrum import spectrum
from seawave.spectrum.module import dispersion
import matplotlib.pyplot as plt
from scipy.signal import welch




rc.surface.x = [0, 1000]
rc.surface.gridSize = [1, 1]


x = np.linspace(*rc.surface.x, rc.surface.gridSize[0])
y = np.linspace(*rc.surface.y, rc.surface.gridSize[1])

fs = 7
t = np.arange(0, 1800, 1/fs, dtype=np.float32)

srf = rough_surface((x,y,t))
surface(srf)
z = srf.elevations.values.flatten()

f, Sxx = welch(z, fs=fs, nfft=rc.surface.kSize, scaling='spectrum')
plt.loglog(f, Sxx)



k = spectrum.k
S = spectrum(k)/dispersion.det(k)

f0 = dispersion.omega(k)/2/np.pi

plt.loglog(f0, S)


print(np.trapz(S, dispersion.omega(k)), np.var(z), np.trapz(spectrum(k), k))
print(np.trapz(Sxx, f))

plt.savefig('kek6')