import sys
sys.path.append(".")
import math


import numpy as np
from seawave import rc, kernel, config
from numba import cuda
import xarray as xr
from seawave.radar import radar
from seawave.surface  import dataset, surface, wind, stat
from seawave.spectrum.module import dispersion
import matplotlib.pyplot as plt
from scipy.signal import welch



config["Surface"]["LimitsOfModeling"] = [[-15, 15], [-15, 15]]
config['Surface']['GridSize'] = [128, 128]
config['Surface']['ThreadPerBlock'] = [8, 8, 8]
config['Radar']['Position'] = [0, 0, 30]

fs = 2
x = np.linspace(*config["Surface"]["LimitsOfModeling"][0], config['Surface']['GridSize'][0], endpoint=False)
y = np.linspace(*config["Surface"]["LimitsOfModeling"][1], config['Surface']['GridSize'][1], endpoint=False)
t = np.arange(0, 900, 1/fs, dtype=np.float32)

ds = surface((x,y,t), config=config)



kernels = [wind, stat]
for kernel in kernels:
    kernel(ds)

# print(ds.harmonics.shape)
P = ds['pulse'].values
t = ds['time_relative'].values

z = ds['distance'].values[:,64, 64]
zr = ds['z_restored'].values

plt.figure()
plt.plot(t, P)
plt.savefig('kek7')

plt.figure()
plt.plot(ds['time'].values, z)
plt.plot(ds['time'].values, zr)
print(ds)
plt.savefig('kek8')
