import sys
sys.path.append(".")
import math


import numpy as np
from seawave import rc, kernel, config
from numba import cuda
import xarray as xr
from seawave.radar import radar
import seawave.surface as srf
from seawave.surface  import dataset, surface, wind, stat
from seawave.spectrum import spectrum
from seawave.spectrum.module import dispersion
import matplotlib.pyplot as plt
from scipy.signal import welch




config["Surface"]["LimitsOfModeling"][0] = [0, 1000]
config['Surface']['GridSize'] = [12, 11]
config['Surface']['ThreadPerBlock'] = [16, 16, 1]

x = np.linspace(*config["Surface"]["LimitsOfModeling"][0], config['Surface']['GridSize'][0])
y = np.linspace(*config["Surface"]["LimitsOfModeling"][1], config['Surface']['GridSize'][1])
fs = 2
t = np.arange(0, 1, 1/fs, dtype=np.float32)

ds = surface((x,y,t), config=config)

print(ds)
kernels = [wind, stat, radar, radar.power]
for kernel in kernels:
    kernel(ds)

print(ds)