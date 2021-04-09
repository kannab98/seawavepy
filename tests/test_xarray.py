import sys
sys.path.append(".")
import math


import numpy as np
from seawave import rc, kernel
from numba import cuda
import xarray as xr
from seawave.radar import radar
import seawave.surface 
from seawave.surface  import dataset, surface



rc.surface.gridSize = [17, 17]
rc.surface.kSize = 1024
rc.surface.phiSize = 128



x = np.linspace(*rc.surface.x, rc.surface.gridSize[0])
y = np.linspace(*rc.surface.y, rc.surface.gridSize[1])
t = np.array([0, 1], dtype=np.float64)


# floatsrf = srf.float_surface((x,y,t))
roughsrf = seawave.surface.rough_surface((x,y,t))
srf = surface(roughsrf)
# srf = radar(srf)
print(srf)

# rs = rs.assign_coords({'k': (['kx','ky'], rs.harmonics)}).coords
# print(roughsrf)
# rs = rs.set_coords(['k'])
# print(vars(rs))
# print(type(rs))