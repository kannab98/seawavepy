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
config['Dataset']['File'] = "dataset_test1.nc"
config['Surface']["Kernel"] = ["CWM-grid"]

ds = xr.open_dataset('dataset.nc')

kernels = [wind]

for kernel in kernels:
    kernel(ds)

print(ds)