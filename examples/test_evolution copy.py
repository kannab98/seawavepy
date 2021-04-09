import sys
sys.path.append(".")
import math


import numpy as np
from seawave import rc, kernel
from numba import cuda
import xarray as xr
from seawave.radar import radar
import seawave.surface as srf
from seawave.surface  import dataset, surface
from seawave.spectrum import spectrum
from seawave.spectrum.module import dispersion
import matplotlib.pyplot as plt
from scipy.signal import welch





srf = xr.open_dataset('database_new.nc')
# print(srf)
z = srf.elevations.values[:,64,64]

fs = 1/(srf.time.values[1]-srf.time.values[0])


f, Sxx = welch(z, fs=fs, nfft=rc.surface.kSize)

Sxx1 = np.fft.fft(z, n=2*f.size)/2/f.size/(2*np.pi)
Sxx1 = np.fft.fftshift(Sxx1)
Sxx1 = np.abs(Sxx1)**2
Sxx1 = 2*Sxx1[Sxx1.size//2:]


k = srf.k
S = np.trapz(srf.harmonics**2, srf.phi)*np.pi/2
S = S*S.size

S0 = spectrum(k.values.flatten())

f0 = srf.omega/2/np.pi
Sxx = Sxx/Sxx.size/2/np.pi/4

print(np.trapz(S0, k), np.trapz(S, k), np.mean(srf.VoE.values))

plt.loglog(f0, S)
plt.loglog(f0, S0)
plt.xlim([0.01, 1000])
# plt.loglog(k, S0)
print(S0.max()/S.max())


plt.savefig('kek6')