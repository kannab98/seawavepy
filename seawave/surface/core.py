import numpy as np 
import xarray as xr
import numba
from numba import f8, cuda
from .dataset import elevations, slopes, velocities, harmonics, phases, statistics
from ..spectrum import spectrum
from .. import config, exit_handler

import atexit
import math
from cmath import exp, phase


import logging
logger = logging.getLogger(__name__)



class float_surface(xr.Dataset):
    __slots__ = ()


    def __init__(self, coords, config=config):

        _N: int = config['Surface']['WaveNumberSize']
        _M: int = config['Surface']['AzimuthSize']


        if isinstance(coords, tuple):
            x,y,t = coords
        elif isinstance(coords, xr.Dataset):
            x,y,t = [coords[proj] for proj in ['x','y','time'] ]

        z = elevations(x, y, t)

        super().__init__({
                          'elevations': z, 
                          'slopes': slopes(x, y, z.values, t), 
                          'velocities': velocities(x, y, z.values, t),
                          'harmonics': harmonics(_N, _M), 
                          'phases': phases(_N, _M)
                        })
    
class surface(float_surface):
    __slots__ = ()



    # psi: np.ndarray(shape=(_N, _M), dtype=np.float32)
    # k:   np.ndarray(shape=(_N, _M), dtype=np.float32)
    # A:   np.ndarray(shape=(_N, _M), dtype=np.float32)

    def __init__(self, data, **kwargs):
            super().__init__(coords=data, **kwargs)
            atexit.register(exit_handler, self)



    @staticmethod
    # @numba.njit(parallel=True)
    def ampld(k: np.ndarray, phi: np.ndarray, S:np.ndarray, azdist:np.ndarray) -> np.ndarray:
        M = phi.size - 1
        N = k.size - 1

        amplitude = np.zeros((N,M), dtype=np.float64)

        for i in range(N):
            amplitude[i,:] = np.trapz(S[i:i+2], k[i:i+2])
            for j in range(M):
                amplitude[i,j] *= np.trapz(azdist[i][j:j+2], phi[j:j+2])

        return np.sqrt(2*amplitude)

    def spectrum(self, phases=True):
        phi = np.linspace(-np.pi, np.pi, self.phi.size + 1, endpoint=True)
        self.__setitem__('phi', phi[:-1])
        KT = spectrum.bounds
        k = np.logspace(np.log10(KT[0]), np.log10(KT[-1]), self.k.size + 1)
        self.__setitem__('k', k[:-1])


        S = spectrum(k)
        azdist = spectrum(k, phi, kind='azdist')
        A = self.ampld(k, phi, S , azdist)

        k = k[None].T * np.exp(1j*phi[None])
        k = np.array(k[:-1, :-1])


        # if self.__getitem__("phases").all() == 0:
        if phases:
            psi = np.random.uniform(0, 2*np.pi, size=self.phases.shape)
            self.__setitem__('phases', ((['k','phi'], psi)) )

        self.__setitem__('harmonics', (['k','phi'], A) )


        return k, A*np.exp(1j*self.phases.values)




device = cuda.get_current_device()
g = config['Constants']['GravityAcceleration']

@cuda.jit(device=True)
def dispersion(k):
    k = abs(k)
    return math.sqrt(g*k + 74e-6*k**3)

@cuda.jit
def kernel(out, x, y, t, k, A, method):
    i, j, n = cuda.grid(3)
    k = cuda.const.array_like(k)
    t = cuda.const.array_like(t)
    A = cuda.const.array_like(A)


    if i > x.size or j > y.size or n > t.size:
        return

    surface = cuda.local.array(6, f8)
    # for m in range(6):
    #     surface[m] = 0

    surface = base(surface, x[i], y[j], t[n], k, A, method)
    for m in range(6):
        out[m, n, i, j] = surface[m]

@cuda.jit
def cwm_grid_kernel(x, y, t, k, A):
    i, j, n = cuda.grid(3)

    k = cuda.const.array_like(k)
    t = cuda.const.array_like(t)
    A = cuda.const.array_like(A)

    if i > x.size or j > y.size or n > t.size:
        return

    ind = (n,i,j)


    x0, y0 = cwm_grid(x[ind], y[ind], t[n], k, A)
    x[ind] = x0
    y[ind] = y0
    cuda.syncthreads()

# @cuda.jit
# def cwm_grid_kernel_experimental(x, y, t, k, A):
#     i, j, n = cuda.grid(3)

#     k = cuda.const.array_like(k)
#     t = cuda.const.array_like(t)
#     A = cuda.const.array_like(A)

#     if i > x.size or j > y.size or n > t.size:
#         return

#     ind = (n,i,j)


#     x0, y0 = cwm_grid_experimental(x[ind], y[ind], t[n], k, A)
#     x[ind] = x0
#     y[ind] = y0
#     cuda.syncthreads()


@cuda.jit(device=True)
def cwm_grid(x, y, t, k, A):

    for n in range(k.shape[0]): 
        for m in range(k.shape[1]):
            kr = k[n,m].real*x + k[n,m].imag*y
            w = dispersion(k[n,m])
            e = A[n,m] * exp(1j*kr)  * exp(1j*w*t)

            x -= e.imag * k[n,m].real/abs(k[n,m])
            y -= e.imag * k[n,m].imag/abs(k[n,m])

    return x, y

# @cuda.jit(device=True)
# def cwm_grid_experimental(x, y, t, k, A):

#     for n in range(k.shape[0]): 
#         for m in range(k.shape[1]):
#             kr = k[n,m].real*x + k[n,m].imag*y
#             w = dispersion(k[n,m])
#             e = A[n,m] * exp(1j*kr)  * exp(1j*w*t)

#             x -= e.imag * k[n,m].real/abs(k[n,m])
#             y -= e.imag * k[n,m].imag/abs(k[n,m])

#     return x, y

@cuda.jit(device=True)
def base(surface, x, y, t, k, A, method):

    slopes_cwm_x = 1
    slopes_cwm_y = 1


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

            # Поправка на наклоны заостренной поверхности
            if method:
                # Смещение dDx/dx = \sum A k cos(phi) cos(kr+wt)
                slopes_cwm_x += -e.real * (k[n,m].real * k[n,m].real)/abs(k[n,m])
                # Смещение dDy/dy = 1 - \sum A k sin(phi) cos(kr+wt)
                slopes_cwm_y += -e.real * (k[n,m].imag * k[n,m].imag)/abs(k[n,m])
                # Орбитальные скорости Vh dVh/dx * dx/dx0
                surface[4] *= 1 - e.real * (k[n,m].real * k[n,m].real)/abs(k[n,m])

    # Наклоны X dz/dx0 * dx0/dx = dz/dx0 * 1/(1 - Dx)
    surface[4] /= slopes_cwm_x
    # Наклоны Y dz/dy0 * dy0/dy = dz/dx0 * 1/(1 - Dy)
    surface[5] /= slopes_cwm_y

    return surface


def init(srf: xr.Dataset, host_constants, dtype):


    x = srf.coords["x"].values
    y = srf.coords["y"].values
    t = srf.coords["time"].values




    limit_per_dim = np.array([
        device.MAX_BLOCK_DIM_X,
        device.MAX_BLOCK_DIM_Y,
        device.MAX_BLOCK_DIM_Z,
    ])
    limit = int(device.MAX_THREADS_PER_BLOCK)


    threadsperblock = config['Surface']['ThreadPerBlock']
    sizes = np.array([x.size, y.size, t.size], dtype=int)

    blockspergrid = tuple( math.ceil(sizes[i] / threadsperblock[i])  for i in range(len(threadsperblock)))

    # stream = cuda.stream()
        

    # arr0 = cuda.to_device(arr[0:-1])



    cuda_constants = tuple(cuda.to_device(host_constants[i]) for i in range(len(host_constants)))
    if dtype == "default" or dtype == "cwm":
        if dtype == "default":
            method = False 
        elif dtype == "cwm":
            method = True
 
        z = srf.elevations.values
        s = srf.slopes.values
        v = srf.velocities.values


        arr = np.zeros((6, *z.shape))

        x0 = cuda.to_device(x)
        y0 = cuda.to_device(y)
        t0 = cuda.to_device(t)

        kernel[blockspergrid, threadsperblock](arr, x0, y0, t0, *cuda_constants, method)
        # arr0.copy_to_host(arr[0:-1])

        srf.elevations.values = arr[0]
        srf.coords['Z'].values = arr[0]
        srf.velocities.values = arr[1:4]
        srf.slopes.values[0:2] = arr[4:]

        # stream.synchronize()
    
    # varelev = spectrum.quad(0,0)
    # if np.abs(np.var(arr[0])- varelev)/varelev > 0.125:
    #     logger.error("Дисперсии высот не совпадают")

    # varslopes = spectrum.quad(2,0)
    # if np.abs(np.var(arr[4])  + np.var(arr[5])- varslopes)/varslopes > 0.125:
    #     logger.error("Полные дисперсии наклонов не совпадают")

    if dtype == "cwm-grid":

        X = srf.coords["X"].values
        Y = srf.coords["Y"].values

        if X.shape[0] != len(srf.coords["time"].values) and len(X.shape) < 3:
            X = np.repeat(X[None], t.size, axis=0)
            Y = np.repeat(Y[None], t.size, axis=0)

        x0 = cuda.to_device(X)
        y0 = cuda.to_device(Y)
        t0 = cuda.to_device(t)
        cuda_constants = tuple(cuda.to_device(host_constants[i]) for i in range(len(host_constants)))

        cwm_grid_kernel[blockspergrid, threadsperblock](x0, y0, t0, *cuda_constants)
        X = x0.copy_to_host()
        Y = y0.copy_to_host()
        # print(X, Y)

        srf["X"] = (["time", "x", "y"], X)
        srf["Y"] = (["time", "x", "y"], Y)

    if dtype == "cwm-grid-experimental":

        X = srf.coords["X"].values
        Y = srf.coords["Y"].values

        if X.shape[0] != len(srf.coords["time"].values) and len(X.shape) < 3:
            X = np.repeat(X[None], t.size, axis=0)
            Y = np.repeat(Y[None], t.size, axis=0)

        x0 = cuda.to_device(X)
        y0 = cuda.to_device(Y)
        t0 = cuda.to_device(t)
        cuda_constants = tuple(cuda.to_device(host_constants[i]) for i in range(len(host_constants)))

        cwm_grid_exp_kernel[blockspergrid, threadsperblock](x0, y0, t0, *cuda_constants)
        X = x0.copy_to_host()
        Y = y0.copy_to_host()
        # print(X, Y)

        srf["X"] = (["time", "x", "y"], X)
        srf["Y"] = (["time", "x", "y"], Y)

    return srf




    

def wind(ds: surface, dtype=config["Surface"]["Kernel"],**kwargs):

    if hasattr(ds, "spectrum") and callable(getattr(ds, "spectrum")) and ds["phases"].all() == 0:
        host_constants = ds.spectrum(**kwargs)
    else: 
    # elif ds.get("k") is not None:
        k = ds["k"].values
        phi = ds["phi"].values
        psi = ds["phases"].values
        A = ds["harmonics"].values

        k = k[None].T * np.exp(1j*phi)
        host_constants = (k, A*np.exp(1j*psi))
    return init(ds, host_constants, dtype)


def stat(ds: xr.Dataset):
    statistics(ds)
