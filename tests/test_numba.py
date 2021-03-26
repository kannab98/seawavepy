import sys
sys.path.append(".")
import math
import matplotlib.pyplot as plt


import numpy as np
from seawave import rc, kernel
from seawave.cuda import default
from seawave.cuda_new import default_test, empty
from numba import cuda
import numba
import xarray as xr
from seawave.radar import radar
from seawave.retracking import retracking
from seawave.surface  import surface
from seawave.spectrum import spectrum
from seawave.surface  import dataset



rc.surface.gridSize = [128, 128]
rc.surface.kSize = 512
rc.surface.phiSize = 128

rc.antenna.z = 30
rc.constants.lightSpeed = 1500 
rc.antenna.impulseDuration = 40e-6

def init(kernel, srf: xr.Dataset, host_constants):
    if kernel == empty:
        dataset.datasets.append(srf)
        return

    x = srf.coords["x"].values
    y = srf.coords["y"].values
    t = srf.coords["time"].values

    arr = np.stack([srf.elevations.values, *srf.velocities.values, *srf.slopes.values], axis=0)

    cuda_constants = tuple(cuda.to_device(host_constants[i]) for i in range(len(host_constants)))
    threadsperblock = (8, 8, 4)
    sizes = (x.size, y.size, t.size)
    blockspergrid = tuple( math.ceil(sizes[i] / threadsperblock[i]) for i in range(len(threadsperblock)))

    x0 = cuda.to_device(x)
    y0 = cuda.to_device(y)
    t0 = cuda.to_device(t)

    kernel[blockspergrid, threadsperblock](arr, x0, y0, t0, *cuda_constants)

    srf.elevations.values = arr[0]
    srf.coords['Z'].values = arr[0]
    srf.velocities.values = arr[1:4]
    srf.slopes.values = arr[4:7]

    # srf = radar(srf)

    dataset.datasets.append(srf)
    return srf


x = np.linspace(*rc.surface.x, rc.surface.gridSize[0])
y = np.linspace(*rc.surface.y, rc.surface.gridSize[1])
t = np.array([0], dtype=np.float64)



n = 10
srf = dataset.float_surface(x, y, t)

slopes = ['slopes x', 'slopes y', 'slopes z' ]
velocity = ['velocity x', 'velocity y', 'velocity z']
coords = ['elevations']



host_constants = surface.export()



fig, ax = plt.subplots(ncols=2)

ax[0].contourf(srf)

srf = dataset.float_surface(x,y,t)
init(default_test, srf, host_constants)
ax[1].contourf(srf['elevations'].values[:,:,0])
plt.savefig('kek3')