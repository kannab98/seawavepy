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
import matplotlib.pyplot as plt



rc.surface.gridSize = [128, 128]
rc.surface.kSize = 1024
rc.surface.phiSize = 128



x = np.linspace(*rc.surface.x, rc.surface.gridSize[0])
y = np.linspace(*rc.surface.y, rc.surface.gridSize[1])
t = np.array([0, 1], dtype=np.float64)



srf = dataset.float_surface(x,y,t)
srf = surface(srf)
srf = radar(srf)
T = np.linspace(radar.find_tmin(srf), radar.find_tmax(srf), 256) 
P = radar.power(srf, T)
plt.plot(T,P)
plt.savefig("example_pulse.png")
