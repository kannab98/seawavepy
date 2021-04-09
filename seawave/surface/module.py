import numpy as np
from numpy import pi
from scipy import interpolate, integrate
import scipy as sp
import pandas as pd
from .. import rc, config
from .. import dataset as ds 
from . import dataset
from ..spectrum import spectrum
from numba import cuda, float32, guvectorize
# from .. import cuda
import math
from cmath import exp, phase
import xarray as xr

import logging
logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt
import os
import atexit


device = cuda.get_current_device()
g = config['Constants']['GravityAcceleration']

@cuda.jit(device=True)
def dispersion(k):
    k = abs(k)
    return math.sqrt(g*k + 74e-6*k**3)

@cuda.jit
def default(out, x, y, t, k, A):
    i, j, n = cuda.grid(3)

    surface = cuda.local.array(6, float32)
    surface = base(surface, x[i], y[j], t[n], k, A)
    for m in range(6):
        out[m, n, i, j] = surface[m]


@cuda.jit(device=True)
def base(surface, x, y, t, k, A):

    for n in range(k.shape[0]): 
        for m in range(k.shape[1]):
                kr = k[n,m].real*x + k[n,m].imag*y
                w = dispersion(k[n,m])
                e = A[n,m] * exp(1j*kr)  * exp(1j*w*t)

                # Высоты (z)
                surface[0] +=  +e.real

                # Орбитальные скорости Vz (dz/dt)
                surface[1] +=  -e.imag * w

                # Vh -- скорость частицы вдоль направления распространения ветра.
                # см. [shuleykin], гл. 3, пар. 5 Энергия волн.
                # Из ЗСЭ V_h^2 + V_z^2 = const

                # Орбитальные скорости Vx
                surface[2] += e.real * w * k[n,m].real/abs(k[n,m])
                # Орбитальные скорости Vy
                surface[3] += e.real * w * k[n,m].imag/abs(k[n,m])
                # Наклоны X (dz/dx)
                surface[4] += -e.imag * k[n,m].real
                # Наклоны Y (dz/dy)
                surface[5] += -e.imag * k[n,m].imag

    return surface


def dispatcher(func):
    """
    Декоратор обновляет необходимые переменные при изменении
    разгона или скорости ветра
    """
    def wrapper(*args, **kwargs):
        self = args[0]
        x = rc.surface.x
        y = rc.surface.y
        gridSize = rc.surface.gridSize
        N = rc.surface.kSize
        M = rc.surface.phiSize

        spectrum.__call__dispatcher__()
        if self._x.min() != x[0] or \
            self._x.max() != x[1] or \
            self._y.min() != y[0] or \
            self._y.max() != y[1] or \
            self._x.shape != gridSize:
            self.gridUpdate()
        
        if self.N != N or self.M != M or rc.surface.randomPhases or \
            isinstance(self.phi, type(None))  or isinstance(self.k, type(None)):
            self.N, self.M = N, M
            self.amplUpdate()

        

        return func(*args, **kwargs)
    return wrapper



class __surface__(object):

    def phasesUpdate(self):
        self.psi = np.random.uniform(0, 2*pi, size=(self.N, self.M))

    def amplUpdate(self):
        self.k = np.logspace(np.log10(spectrum.KT[0]), np.log10(spectrum.KT[-1]), self.N + 1)
        self.phi = np.linspace(-np.pi, np.pi,self.M + 1, endpoint=True)
        self.psi = np.random.uniform(0, 2*pi, size=(self.N, self.M))

    def gridUpdate(self):
        self._x = np.linspace(*rc.surface.x, rc.surface.gridSize[0])
        self._y = np.linspace(*rc.surface.y, rc.surface.gridSize[1])
        self._t = np.array([0, 1])

        # srf = dataset.float_surface(self._x, self._y, self._t)
        # self._n = np.frombuffer(srf.slopes.data).reshape(srf.slopes.data.shape)
        # self._v = np.frombuffer(srf.velocities.data).reshape(srf.velocities.data.shape)





    def __init__(self, **kwargs):

        self.N = rc.surface.kSize
        self.M = rc.surface.phiSize
        self.__psi__ = np.random.uniform(0, 2*pi, size=(self.N, self.M))

        self.k = None
        self.phi = None
        self.gridUpdate()


    
    @staticmethod
    def angle_correction(theta):
        # Поправка на угол падения с учётом кривизны Земли
        R = rc.constants.earthRadius
        z = rc.antenna.z
        theta =  np.arcsin( (R+z)/R * np.sin(theta) )
        return theta
    
    # def cwmCorrection(self, moments):
        



    @staticmethod
    def ampld(k, phi, **quadkwargs):
        M = phi.size - 1
        N = k.size - 1

        amplitude = np.zeros((N,M))
        S = spectrum(k)
        azdist = spectrum(k, phi)/S
        azdist = azdist.T
        for i in range(N):
            amplitude[i,:] = np.trapz(S[i:i+2], k[i:i+2])
            for j in range(M):
                amplitude[i,j] *= np.trapz(azdist[i][j:j+2], phi[j:j+2])
        return np.sqrt(2*amplitude)



    @dispatcher
    def export(self):


        k = self.k[None].T * np.exp(1j*self.phi[None])
        k = np.array(k[:-1, :-1])


        A0 = self.ampld(self.k, self.phi)

        if rc.surface.randomPhases:
            self.psi = np.random.uniform(0, 2*pi, size=(self.N, self.M))
        else:
            self.psi = self.__psi__


        return k, A0*np.exp(1j*self.psi)
    
    def __call__(self, srf, kernel=default):
        # host_constants = srf.spectrum()
        host_constants = self.export()

        srf = init(srf, host_constants, kernel = default)
        dataset.statistics(srf)

        
        exit_handler = lambda: srf.to_netcdf('database.nc')

        atexit.register(exit_handler)

        return srf


def init(srf: xr.Dataset, host_constants, kernel=default):

    x = srf.coords["x"].values
    y = srf.coords["y"].values
    t = srf.coords["time"].values

    z = srf.elevations.values
    s = np.transpose(srf.slopes.values, (1, 0, 2, 3))
    v = np.transpose(srf.velocities.values, (1, 0, 2, 3))

    arr = np.stack([z, *v, *s], axis=0)

    cuda_constants = tuple(cuda.to_device(host_constants[i]) for i in range(len(host_constants)))



    limit_per_dim = np.array([
        device.MAX_BLOCK_DIM_X,
        device.MAX_BLOCK_DIM_Y,
        device.MAX_BLOCK_DIM_Z,
    ])
    limit = int(device.MAX_THREADS_PER_BLOCK)




    threadsperblock = np.array([8, 8, 8], dtype=int)
    sizes = np.array([x.size, y.size, t.size], dtype=int)

    if sizes[0]==1 and sizes[1]==1:
        threadsperblock = np.array([1, 1, limit_per_dim[2]], dtype=int)

    elif sizes[1]==1 and sizes[2]==1:
        threadsperblock = np.array([limit_per_dim[0], 1, 1], dtype=int)

    elif sizes[0]==1 and sizes[2]==1:
        threadsperblock = np.array([1,limit_per_dim[0], 1], dtype=int)

    threadsperblock = tuple(threadsperblock)

    blockspergrid = tuple( math.ceil(sizes[i] / threadsperblock[i]) for i in range(len(threadsperblock)))

    x0 = cuda.to_device(x)
    y0 = cuda.to_device(y)
    t0 = cuda.to_device(t)

    kernel[blockspergrid, threadsperblock](arr, x0, y0, t0, *cuda_constants)

    srf.elevations.values = arr[0]
    srf.coords['Z'].values = arr[0]
    srf.velocities.values = np.transpose(arr[1:4], axes=(1, 0, 2, 3))
    srf.slopes.values = np.transpose(arr[4:7], axes=(1, 0, 2, 3))

    return srf


def __call__(self, srf, kernel=default):
    host_constants = srf.spectrum()
    # host_constants = self.export()

    srf = init(srf, host_constants, kernel = default)
    dataset.statistics(srf)

    
    # exit_handler = lambda: srf.to_netcdf('database.nc')

    atexit.register(exit_handler, srf)

    return srf


def exit_handler(srf):

    print('kek')
    for coord in ['X', 'Y']:
        print('kek')
        if np.allclose(srf[coord].values[0,:,:], srf[coord].values):
            srf[coord] = (["x", "y"], srf[coord].values[0,:,:]) 

    srf.to_netcdf('database.nc')




