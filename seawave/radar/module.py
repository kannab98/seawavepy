
import logging
import numpy as np
import matplotlib.pyplot as plt
import os

from ..surface import surface
from .. import rc, kernel, cuda, DATADIR

from numba import njit, prange
import scipy as sp
import pandas as pd

logger = logging.getLogger(__name__)
class __radar__():

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

            if self._n.all() != surface.normal.all() or \
                self._coordinates.all() != self.sattelite_coordinates.all():
                self.geometryUpdate()
            


            return func(*args, **kwargs)

        return wrapper

    def geometryUpdate(self):

        if self._coordinates.all() != self.sattelite_coordinates.all():
            logger.info('Init radar with position %s' % str(self.sattelite_coordinates))
            self._coordinates = self.sattelite_coordinates

        if self._gainWidth != rc.antenna.gainWidth or self._waveLength != rc.antenna.waveLength:
            logger.info('Transmitter params: gain_width=%.1f, wave length=%s' % (rc.antenna.gainWidth, str(rc.antenna.waveLength)) )
            self._gainWidth = rc.antenna.gainWidth
            self._waveLength = rc.antenna.waveLength

        self._R[:] = self.position(surface.coordinates, self.sattelite_coordinates)
        # Смещение координаты x!
        offset = rc.antenna.z*np.arctan(np.deg2rad(rc.antenna.deviation))
        R0 = np.array([offset, 0, 0])[None].T*np.ones((1,self._R.shape[-1]))

        self._R += R0
        self._n[:] = surface.normal
        self._n[:] = self._n/self.abs(self._n)
        self._Rabs = self.abs(self._R)




    def __init__(self):

        labels = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        srf = np.zeros((len(labels), np.prod(rc.surface.gridSize)))
        self._data = pd.DataFrame(srf.T, columns=labels)


        self._R = np.array([
            np.frombuffer(self._data.x.values),
            np.frombuffer(self._data.y.values),
            np.frombuffer(self._data.z.values),
        ], dtype="object")

        self._n = np.array([
            np.frombuffer(self._data.nx.values),
            np.frombuffer(self._data.ny.values),
            np.frombuffer(self._data.nz.values),
        ], dtype="object")

        self._gainWidth = None
        self._waveLength = None
        self._coordinates = np.array([None])



        self._Rabs = None

        # self.geometryUpdate()

        self._gamma = 2*np.sin(rc.antenna.gainWidth/2)**2/np.log(2)
        # self._G = self.G(self._R, 
        #                     rc.antenna.polarAngle, 
        #                     rc.antenna.deviation, self._gamma)


        # self.tau = self._Rabs / rc.constants.lightSpeed
        # self.t0 = self._Rabs.min() / rc.constants.lightSpeed



    @staticmethod
    def abs(vec):
        vec = np.array(vec, dtype=float)
        return np.sqrt(np.sum(vec**2,axis=0))
        # return np.sqrt(np.diag(vec.T@vec))

    @staticmethod
    def position(r, r0):
        r0 = r0[np.newaxis]
        return np.array(r + ( np.ones((r.shape[1], 1) ) @ r0 ).T)


    
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


    @property
    def sattelite_coordinates(self):
        return np.array([rc.antenna.x, rc.antenna.y, rc.antenna.z])

    @sattelite_coordinates.setter
    def sattelite_coordinates(self, r):
        rc.antenna.x, rc.antenna.y, rc.antenna.z = r
        self._R = self.position(self.surface_coordinates, self.sattelite_coordinates)
        self._Rabs = self.abs(self._R)



    @property
    def incidence(self):
        z = np.array(self._R[-1], dtype=float)
        R = np.array(self._Rabs, dtype=float)
        return np.arccos(z/R)

    def localIncidence(self, xi=0):
        rc.antenna.deviation = xi
        # self.geometryUpdate()
        R = np.array(self._R/self._Rabs, dtype=float)
        n = np.array(self._n, dtype=float)




        # print(R.shape)


        # Матрица поворта вокруг X
        # Mx = self.rotatex(np.pi/2)
        # My = self.rotatey(np.pi/2)
        # taux = np.einsum('ij, jk -> ik', My, n)
        # tauy = np.einsum('ij, jk -> ik', Mx, n)

        theta0n = np.einsum('ij, ij -> j', R, n)
        # theta0tau = np.einsum('ij, ij -> j', R, taux)
        # theta0tauy = np.einsum('ij, ij -> j', R, tauy)


        # return np.sign(theta0tau)*np.arccos(theta0n)
        return np.arccos(theta0n)

    @staticmethod
    def rotatex(alpha):
        Mx = np.array([
            [1,             0,             0 ],
            [0, np.cos(alpha), -np.sin(alpha)],
            [0, np.sin(alpha), +np.cos(alpha)]
        ])
        return Mx

    @staticmethod
    def rotatey(alpha):
        My = np.array([
            [+np.cos(alpha), 0, np.sin(alpha)],
            [0,              1,            0 ],
            [-np.sin(alpha), 0, np.cos(alpha)]
        ])
        return My

    @staticmethod
    def sort(incidence, xi=0, err = 0.7):

        # if not isinstance(xi, np.ndarray):
        #     if isinstance(xi, list):
        #         xi = np.array(xi)
        #     else:
        #         xi = np.array([xi])

        # xi = xi[np.newaxis]

        # theta0 = np.abs(incidence - xi.T)
        
        # index = np.abs(theta0) < np.deg2rad(err)
        index = np.where(incidence < np.deg2rad(err))
        return index

    
    def gain(self, theta, xi):
        self._G = self.G(self._R, 
                            theta,
                            xi, self._gamma)
        return self._G

    
    
    @dispatcher
    def power(self, t):
        c =  rc.constants.lightSpeed
        timp = rc.antenna.impulseDuration
        G = self.gain(self.incidence, rc.antenna.deviation)
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

    @dispatcher
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