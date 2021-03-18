import numpy as np
from numpy import pi
from scipy import interpolate, integrate
import scipy as sp
import pandas as pd
from multiprocessing import Array, Process, Pool, cpu_count
from itertools import product

from .. import rc, DATADIR
from .. import dataset as ds 
from . import dataset
from ..spectrum import spectrum
import numba as nb
from .. import cuda
import math
import xarray as xr

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

        srf = dataset.float_surface(self._x, self._y, self._t)
        self._n = np.frombuffer(srf.slopes.data).reshape(srf.slopes.data.shape)
        self._v = np.frombuffer(srf.velocities.data).reshape(srf.velocities.data.shape)




    @staticmethod
    def abs(vec):
        vec = np.array(vec, dtype=float)
        return np.diag(vec.T@vec)

    @staticmethod
    def position(r, r0):
        r0 = r0[np.newaxis]
        return np.array(r + (np.ones((r.shape[1], 1)) @ r0).T)

    def __call__(*args, **kwargs):
        return self.export()

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
    

class float_surface():
    def __call__(self, *args):
        dataset.float_surface()

    def __init__(self, ):
        pass 
