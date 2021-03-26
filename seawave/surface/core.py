import numpy as np 
import xarray as xr
import numba
from numba import f8
from .dataset import elevations, slopes, velocities
from ..spectrum import spectrum
from .. import rc

# import .functions
# def cross_section(theta: np.ndarray, cov: np.ndarray) -> np.ndarray: 
    # return .functions.cross_section(theta, cov)

# @numba.njit





class float_surface(xr.Dataset):
    __slots__ = ()
    _N: int = rc.surface.kSize
    _M: int = rc.surface.phiSize
    def __init__(self, coords, **kwargs):

        if isinstance(coords, tuple):
            x,y,t = coords
        elif isinstance(coords, xr.Dataset):
            x,y,t = [coords[proj] for proj in ['x','y','time'] ]

        z = elevations(x, y, t)
        n = slopes(x, y, z.values, t)
        v = velocities(x, y, z.values, t)

        h = xr.DataArray(data=np.zeros((self._N, self._M), dtype=np.complex128), 
                        dims=["kx", "ky"], 
                        coords=dict(
            wave_vector = (['kx','ky'], np.zeros((self._N, self._M))),
        ))
        super().__init__({'elevations': z, 'slopes': n, 'velocities':v, 'harmonics':h})
    
class rough_surface(float_surface):
    __slots__ = ()
    _N: int = rc.surface.kSize
    _M: int = rc.surface.phiSize
    __psi__ = np.random.uniform(0, 2*np.pi, size=(_N, _M))


    psi: np.ndarray(shape=(_N, _M), dtype=np.float32)
    k:   np.ndarray(shape=(_N, _M), dtype=np.float32)
    A:   np.ndarray(shape=(_N, _M), dtype=np.float32)

    def __init__(self, data):
            super().__init__(coords=data)

    @staticmethod
    @numba.njit(parallel=False)
    def ampld(k: np.ndarray, phi: np.ndarray, S:np.ndarray, azdist:np.ndarray) -> np.ndarray:
        M = phi.size - 1
        N = k.size - 1

        amplitude = np.zeros((N,M), dtype=numba.float32)

        for i in range(N):
            amplitude[i,:] = np.trapz(S[i:i+2], k[i:i+2])
            for j in range(M):
                amplitude[i,j] *= np.trapz(azdist[i][j:j+2], phi[j:j+2])

        return np.sqrt(2*amplitude)

    def spectrum(self):
        phi = np.linspace(-np.pi, np.pi, self._M + 1, endpoint=True)
        k = np.logspace(np.log10(spectrum.KT[0]), np.log10(spectrum.KT[-1]), self._N + 1)



        S = spectrum(k)
        azdist = spectrum(k, phi, kind='azdist')
        A = self.ampld(k, phi, S , azdist)

        k = k[None].T * np.exp(1j*phi[None])
        k = np.array(k[:-1, :-1])


        if rc.surface.randomPhases:
            psi = np.random.uniform(0, 2*np.pi, size=(self._N, self._M))
        else:
            psi = self.__psi__
        

        self.__setitem__('harmonics', (['kx','ky'], A*np.exp(1j*psi)) )
        self.__setitem__('wavevector', (['kx','ky'], k))

        return self.wave_vector, self.harmonics



