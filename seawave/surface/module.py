import numpy as np
from numpy import pi
from scipy import interpolate, integrate
import scipy as sp
import pandas as pd
from multiprocessing import Array, Process, Pool, cpu_count
from itertools import product

from .. import rc
from ..spectrum import spectrum

import logging
logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt
import os

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


    def amplUpdate(self):
        self.k = np.logspace(np.log10(spectrum.KT[0]), np.log10(spectrum.KT[-1]), self.N + 1)
        self.phi = np.linspace(-np.pi, np.pi,self.M + 1, endpoint=True)

        if rc.surface.randomPhases:
            self.psi = np.random.uniform(0, 2*pi, size=(self.N, self.M))
        else:
            self.psi = self.__psi__

    def gridUpdate(self):

        self._x = np.linspace(*rc.surface.x, rc.surface.gridSize[0])
        self._y = np.linspace(*rc.surface.y, rc.surface.gridSize[1])
        self._x, self._y = np.meshgrid(self._x, self._y)
        """
        self._z -- высоты морской поверхности
        self._zx -- наклоны X (dz/dx)
        self._zy -- наклоны Y (dz/dy)
        """
        self._z = np.zeros(rc.surface.gridSize)
        self._zx = np.zeros(rc.surface.gridSize)
        self._zy = np.zeros(rc.surface.gridSize)
        self._zz = np.zeros(rc.surface.gridSize)

        """
        self._vz -- 
        self._vx -- 
        self._vy -- 

        """

        self._vx = np.zeros(rc.surface.gridSize)
        self._vy = np.zeros(rc.surface.gridSize)
        self._vz = np.zeros(rc.surface.gridSize)



        # Ссылка на область памяти, где хранятся координаты точек поверхности
        self._r = np.array([
            np.frombuffer(self._x),
            np.frombuffer(self._y),
            np.frombuffer(self._z),
        ], dtype="object")

        self._n = np.array([
            np.frombuffer(self._zx),
            np.frombuffer(self._zy),
            np.frombuffer(self._zz),
        ], dtype="object")

        self._v = np.array([
            np.frombuffer(self._vx),
            np.frombuffer(self._vy),
            np.frombuffer(self._vz),
        ], dtype="object")

    @staticmethod
    def abs(vec):
        vec = np.array(vec, dtype=float)
        return np.diag(vec.T@vec)

    @staticmethod
    def position(r, r0):
        r0 = r0[np.newaxis]
        return np.array(r + (np.ones((r.shape[1], 1)) @ r0).T)

    def __init__(self, **kwargs):


        self._data = None
        self._R = None

        self._A = None
        self._F = None
        self._Psi = None

        self.N = rc.surface.kSize
        self.M = rc.surface.phiSize
        self.__psi__ = np.random.uniform(0, 2*pi, size=(self.N, self.M))

        self._x = np.linspace(*rc.surface.x, rc.surface.gridSize[0])
        self._y = np.linspace(*rc.surface.y, rc.surface.gridSize[1])
        self._x, self._y = np.meshgrid(self._x, self._y)

        # Генерация плоской поверхности
        labels = ['x', 'y', 'z', 'sx', 'sy', 'sz', 'vz', 'vx', 'vy']
        srf = np.zeros((len(labels), np.prod(rc.surface.gridSize)))

        self._data = pd.DataFrame(srf.T, columns=labels)
        self._data['x'] = self._x.flatten()
        self._data['y'] = self._y.flatten()
        self._data['sz'] = np.ones(np.prod(rc.surface.gridSize))

    
        self.k = None
        self.phi = None


    
    @staticmethod
    def angle_correction(theta):
        # Поправка на угол падения с учётом кривизны Земли
        R = rc.constants.earthRadius
        z = rc.antenna.z
        theta =  np.arcsin( (R+z)/R * np.sin(theta) )
        return theta
    
    # def cwmCorrection(self, moments):
        

    @staticmethod
    def cross_section(theta, cov): 
        # theta = Surface.angle_correction(theta)
        theta = theta[np.newaxis]
        # Коэффициент Френеля
        F = 0.8

        if len(cov.shape) <= 2:
            cov = np.array([cov])

        K = np.zeros(cov.shape[0])
        for i in range(K.size):
            K[i] = np.linalg.det(cov[i])

        sigma =  F**2/( 2*np.cos(theta.T)**4 * np.sqrt(K) )
        sigma *= np.exp( - np.tan(theta.T)**2 * cov[:, 1, 1]/(2*K))
        return sigma


    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, data):
        columns = data.columns
        self._data[columns] = data[columns]

        data_hs = 4*np.std(self._data.z)
        data_varsx = np.var(self._data.sx) 
        data_varsy = np.var(self._data.sy)

        logger.info('Create new surface with parameters: Hs=%.4f, sigmaxx=%.2E, sigmayy=%.2E' % (data_hs, data_varsx, data_varsy))

        hs = 4*np.sqrt(spectrum.quad(0, 0, epsabs=1.49e-4))
        var_slopes = spectrum.quad(2, 0, epsabs=1.49e-4)

        logger.info('Theoretical parameters: Hs=%.4f, sigmaxx+sigmayy=%.2E' % (hs, var_slopes))


        if (data_hs > 1.25*hs) or (data_hs < .75*hs):
            logger.warning('Surface parameters does not match with theory')


    @property
    @dispatcher
    def meshgrid(self):
        r = np.array(self._r[0:2], dtype=float)
        x = self.reshape(self.data.x)
        y = self.reshape(self.data.y)
        return x,y

    @meshgrid.setter
    def meshgrid(self, rho: tuple):
        self._data['x'] = rho[0]
        self._data['y'] = rho[1]

    @property
    def gridx(self):
        return self.reshape(self.data.x)

    @property
    def gridy(self):
        return self.reshape(self.data.y)

    @property
    def heights(self):
        return self.reshape(self.data.z)

    @heights.setter
    def heights(self, z):
        self._data['z'] = z

    @property
    def coordinates(self):
        return self._data[['x', 'y', 'z']].to_numpy().T

    @coordinates.setter
    def coordinates(self, r: tuple):
        for i, x in enumerate(['x', 'y', 'z']):
            self._data[x] = r[i]

    @property
    def normal(self):
        return self._data[['sx', 'sy', 'sz']].to_numpy().T


    @normal.setter
    def normal(self, n: tuple):
        for i, n in enumerate(['sx', 'sy', 'sz']):
            self._data[n] = n[i]

    @property
    def amplitudes(self):
        return self._A

    @property
    def angleDistribution(self):
        return self._F

    @property
    def phases(self):
        return self._Psi


    @amplitudes.setter
    def amplitudes(self, A):
        self._A = A

    @phases.setter
    def phases(self, Psi):
        self._Psi = Psi
    
    def reshape(self, arr):
        return np.array(arr, dtype=float).reshape(*rc.surface.gridSize)
    


    def ampld(self, k, phi, **quadkwargs):
        M = phi.size - 1
        N = k.size - 1

        amplitude = np.zeros((N,M))
        S = spectrum(k)
        azdist = spectrum(k, phi)/S
        azdist = azdist.T
        for i in range(N):
            amplitude[i,:] = np.trapz(S[i:i+2], k[i:i+2])
            for j in range(M):
                # amplitude[i,j] = np.trapz(np.trapz(S[i:i+2][j:j+2], k[i:i+2]), phi[j:j+2])
                amplitude[i,j] *= np.trapz(azdist[i][j:j+2], phi[j:j+2])
        return np.sqrt(2*amplitude)



    @dispatcher
    def export(self):


        srf = rc.surface
        # Aname = "A_%s_%s_%s_%s_%s.npy" % (srf.band, srf.kSize, srf.phiSize, rc.wind.speed, rc.wind.direction)
        # Apath = os.path.join( CACHE_FOLDER, Aname)

        k = self.k[None].T * np.exp(1j*self.phi[None])
        k = np.array(k[:-1, :-1])
        A0 = self.ampld(self.k, self.phi)

        return k, A0*np.exp(1j*self.psi)