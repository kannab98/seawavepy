
import logging
import numpy as np
import matplotlib.pyplot as plt
import os

from ..surface import surface
from .. import rc, kernel, cuda, DATADIR

from numba import njit, prange
import scipy as sp
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)
class __radar__():


    def __init__(self, srf: xr.Dataset):

        self._R = np.array([srf['X'], srf['Y'], srf['Z']])
        self._Rabs = srf.distance.values

        self._gainWidth = None
        self._waveLength = None
        self._coordinates = np.array([None])

        self._gamma = 2*np.sin(rc.antenna.gainWidth/2)**2/np.log(2)

    
    @staticmethod
    def G(r, phi, xi, gamma):
        x = r[0]
        y = r[1]
        z = r[2]
        # Поворот системы координат на угол phi
        X = np.array(x * np.cos(phi) + y * np.sin(phi), dtype=float)
        Y = np.array(x * np.sin(phi) - y * np.cos(phi), dtype=float)
        Z = np.array(z, dtype=float)

        # Проекция вектора (X,Y,Z) на плоскость XY
        rho =  np.sqrt(np.power(X, 2) + np.power(Y, 2))
        rho[np.where(rho == 0)] += 1.49e-8

        # Полярный угол в плоскости XY
        psi = np.arccos(X/rho)


        if not isinstance(xi, np.ndarray):
            if isinstance(xi, list):
                xi = np.array(xi)
            else:
                xi = np.array([xi])

        xi = xi[np.newaxis]


        theta = (Z*np.cos(xi.T) + rho*np.sin(xi.T)*np.cos(psi))
        theta *= 1/np.sqrt(X**2 + Y**2 + Z**2)
        theta = np.arccos(theta)

        # Возвращает массив размерности (xi, r)
        return np.exp(-2/gamma * np.sin(theta)**2)


    def gain(self, theta, xi):
        self._G = self.G(self._R, 
                            theta,
                            xi, self._gamma)
        return self._G

    
    def power(self, t):
        c =  rc.constants.lightSpeed
        timp = rc.antenna.impulseDuration
        G = self.gain(srf['AoD'], rc.antenna.deviation)
        theta0 = self.localIncidence(xi=rc.antenna.deviation)
        Rabs = self._Rabs

        P = power_vec(t, Rabs, G, theta0, timp, c)

        return P

    
    def find_rho(self):
        sigmaxx = np.var(surface.data.sx)
        theta = lambda rho: np.arctan(rho/rc.antenna.z)
        Wxx = lambda x: 1/np.sqrt(2*np.pi*sigmaxx**2) * np.exp(-x**2/(2*sigmaxx**2))
        eps = np.deg2rad(1)


        probality = 1.49e-16 # вероятность того, что наклон примет значение theta
        F = lambda rho: sp.integrate.quad(Wxx, a=np.tan(theta(rho)-eps), b=np.tan(theta(rho)+eps))[0] - probality

        return sp.optimize.root_scalar(F, bracket=[0, rc.surface.x[1]]).root
    
    def find_tmax(self):
        tmax = np.sqrt(self.find_rho()**2 + (rc.antenna.z-surface.data.z.min())**2)/rc.constants.lightSpeed
        return 1.04*tmax


    def find_tmin(self):
        tmin = (rc.antenna.z - surface.data.z.max())/rc.constants.lightSpeed 
        return tmin*0.99

    def create_multiple_pulses(self, N, t=None, dump=False):
        P = 0

        data = np.zeros((256, N+1), dtype=float)

        for i in range(N):
            srf = kernel.simple_launch(cuda.default)
            surface.data = srf
            t = np.linspace(self.find_tmin(), self.find_tmax(), data.shape[0])
            Pn = self.power(t)
            data[:, i] = Pn
            P += Pn

        
        data[:, -1] = P/N
        data[:, -1] += 1e-1*data[:, -1].max()*np.random.rand(data[:, -1].size)

        if dump:
            with pd.ExcelWriter(os.path.join(DATADIR,'impulses.xlsx'), engine="xlsxwriter") as writer:
                for i in range(N):
                    df = pd.DataFrame({'t': t, 'P': data[:, i] })
                    df.to_excel(writer, sheet_name='impulse%d' % i)
                
                df = pd.DataFrame({'t': t, 'P': data[:, -1]})
                df.to_excel(writer, sheet_name='impulse_mean')

                dt = {}
                for Key, Value in vars(rc).items():
                    # if type(Value) ==  type(rc.surface):
                    dt.update({Key: {}})
                    for key, value in Value.__dict__.items():
                        if key[0] != "_":
                            dt[Key].update({key: value})

                df = pd.DataFrame(dt)
                df.to_excel(writer, sheet_name='config')
            
            if rc.dump.plot:
                fig, ax = plt.subplots()
                ax.plot(t, data[:, -1])
                fig.savefig(os.path.join(DATADIR, 'mean_pulse.png'))

        return data[:, -1], 2*t

    def crossSection(self, theta):
        # Эти штуки не зависят от ДН антенны, только от геометрии
        # ind = self.sort(self.localIncidence)
        # Rabs = self._Rabs[ind]
        # R = R[:,index]
        self.geometryUpdate()
        # gamma = 2*np.sin(np.deg2rad(15)/2)**2/np.log(2)

        # G = self.G(self._R, np.pi/2, theta, gamma)

        N = np.zeros_like(theta, dtype=float)
        for i, xi in enumerate(theta):
            theta0 = self.localIncidence(xi=xi)

            ind = self.sort(theta0, xi=xi)
            # N[i] = np.sum(G[i][ind])
            N[i] = ind[0].size


        # # Эти, разумеется, зависят


        # E0 = G**2/R**2*cos


        return N

    

@njit(parallel=True)
def power_vec(t, Rabs, G, theta0, timp, c):
    P = np.zeros(t.size)
    tau = Rabs / c
    for i in prange(t.size):
        mask = (0 <= t[i] - tau) & (t[i] - tau <= timp) & (theta0 <= np.deg2rad(1))
        E0 = (G[:,mask]/Rabs[mask])**2
        P[i] = np.sum(E0**2/2)
    return P