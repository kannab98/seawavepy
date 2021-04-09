import numpy as np 
import xarray as xr
import numba
from numba import f8, cuda
from .dataset import elevations, slopes, velocities, harmonics, phases, statistics
from ..spectrum import spectrum
from .. import config

import atexit
import math
from cmath import exp, phase





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
            # exit_handler = lambda: self.to_netcdf('database.nc')
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

    def spectrum(self):
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


        psi = np.random.uniform(0, 2*np.pi, size=self.phases.shape)
        

        self.__setitem__('harmonics', (['k','phi'], A) )
        self.__setitem__('phases', ((['k','phi'], psi)) )

        return k, A*np.exp(1j*psi)




device = cuda.get_current_device()
g = config['Constants']['GravityAcceleration']

@cuda.jit(device=True)
def dispersion(k):
    k = abs(k)
    return math.sqrt(g*k + 74e-6*k**3)

@cuda.jit
def default(out, x, y, t, k, A):
    i, j, n = cuda.grid(3)

    if i > x.size or j > y.size or n > t.size:
        return

    surface = cuda.local.array(6, f8)
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

def init(srf: xr.Dataset, host_constants, kernel=default):

    x = srf.coords["x"].values
    y = srf.coords["y"].values
    t = srf.coords["time"].values

    z = srf.elevations.values
    s = srf.slopes.values
    v = srf.velocities.values

    arr = np.stack([z, *v, *s], axis=0)

    cuda_constants = tuple(cuda.to_device(host_constants[i]) for i in range(len(host_constants)))

    limit_per_dim = np.array([
        device.MAX_BLOCK_DIM_X,
        device.MAX_BLOCK_DIM_Y,
        device.MAX_BLOCK_DIM_Z,
    ])
    limit = int(device.MAX_THREADS_PER_BLOCK)




    # threadsperblock = np.array([8, 8, 8], dtype=int)
    threadsperblock = config['Surface']['ThreadPerBlock']
    sizes = np.array([x.size, y.size, t.size], dtype=int)

    # if sizes[0]==1 and sizes[1]==1:
    #     threadsperblock = np.array([1, 1, limit_per_dim[2]], dtype=int)

    # elif sizes[1]==1 and sizes[2]==1:
    #     threadsperblock = np.array([limit_per_dim[0], 1, 1], dtype=int)

    # elif sizes[0]==1 and sizes[2]==1:
    #     threadsperblock = np.array([1,limit_per_dim[0], 1], dtype=int)

    # threadsperblock = tuple(threadsperblock)
    blockspergrid = tuple( math.ceil(sizes[i] / threadsperblock[i])  for i in range(len(threadsperblock)))

    x0 = cuda.to_device(x)
    y0 = cuda.to_device(y)
    t0 = cuda.to_device(t)

    kernel[blockspergrid, threadsperblock](arr, x0, y0, t0, *cuda_constants)

    srf.elevations.values = arr[0]
    srf.coords['Z'].values = arr[0]
    srf.velocities.values = arr[1:4]
    srf.slopes.values = arr[4:7]

    return srf




    

def wind(ds: xr.Dataset):
    host_constants = ds.spectrum()
    return init(ds, host_constants, kernel = default)

def stat(ds: xr.Dataset):
    statistics(ds)

def exit_handler(srf):

    for coord in ['X', 'Y']:
        if np.allclose(srf[coord].values[0,:,:], srf[coord].values):
            srf[coord] = (["x", "y"], srf[coord].values[0,:,:]) 

    srf.to_netcdf(config['Dataset']['File'])