import sys
sys.path.append(".")


import numpy as np
from seawave import rc, kernel, config
from numba import cuda
import xarray as xr
from seawave.radar import radar
from seawave.surface  import dataset, surface, wind, stat
from seawave.spectrum.module import dispersion
import matplotlib.pyplot as plt

config['Radar']['Position'] = [0, 0, 30]

ds = xr.open_dataset('database.nc')

kernels = [radar, radar.power]

for kernel in kernels:
    kernel(ds)

print(ds)