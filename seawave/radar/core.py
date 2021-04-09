from numba import cuda 
from .. import config
import math 
import xarray as xr
import numpy as np


def distance(srf: xr.Dataset) -> xr.Dataset:
    
    srf.coords["X"] += config['Radar']['Position'][0]
    srf.coords["Y"] += config['Radar']['Position'][1]
    srf.coords["Z"] = srf['elevations'] + config['Radar']['Position'][2]

    distance = srf.elevations.copy()

    threadsperblock = (8,8,8)
    sizes = srf['X'].shape
    blockspergrid = tuple( math.ceil(sizes[i] / threadsperblock[i])  for i in range(len(threadsperblock)))

    x = cuda.to_device(srf['X'].values)
    y = cuda.to_device(srf['Y'].values)
    z = cuda.to_device(srf['Z'].values)

    __distance__[blockspergrid, threadsperblock](x, y, z, distance.values)

    distance.attrs = dict(
            description="Array with surface fields data.",
    )


def normal(srf: xr.Dataset) -> xr.Dataset:
def angle_of_arrival(srf: xr.Dataset ) -> xr.Dataset:
    d = srf['distance'].values
    n = srf['normal'].values
    r = np.array([srf['X'], srf['Y'], srf['Z']])

    AoA = srf.elevations.copy()
    prod = np.einsum('ijkm, ijkm -> jkm', r/d, n)
    AoA.values = np.arccos(prod)
    AoA.attrs=dict(
            description="Angle of arrival",
        )

    srf['AoA'] = AoA

# @cuda.njit
# def __angle_of_arrival__():

#     return

@cuda.jit
def __distance__(x, y, z, distance):

    n, i, j = cuda.grid(3)

    if n > x.shape[0] or i > x.shape[1] or j > x.shape[2] :
        return

    distance[n,i,j] = math.sqrt(x[n,i,j]**2 + y[n,i,j]**2 + z[n,i,j]**2)