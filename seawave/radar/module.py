
import logging
import numpy as np
import matplotlib.pyplot as plt
import os

from ..surface import surface, dataset
from .. import config

from numba import njit, jit, prange, guvectorize, cuda
import scipy as sp
import pandas as pd
import xarray as xr
import math

logger = logging.getLogger(__name__)


c =  config["Constants"]["WaveSpeed"]
timp = config["Radar"]['ImpulseDuration']

class __radar__():

    def __call__(self, srf: xr.Dataset):
        srf = dataset.radar(srf)
        self._R = np.array([srf['X'], srf['Y'], srf['Z']])
        # self._Rabs = srf.distance.values
        self._n = self.normal(srf)

        self._gainWidth = None
        self._waveLength = None
        self._coordinates = np.array([None])
        return srf

    def __init__(self, ):
        self._gamma = 2*np.sin(config['Radar']['GainWidth']/2)**2/np.log(2)

    
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
    @staticmethod
    def normal(srf):

        n = srf.slopes.copy()
        n.values = srf['slopes'].values/np.linalg.norm(srf['slopes'].values, axis=0)
        n.attrs = dict(
            description="normal to surface"
        )

        srf['normal'] = n
        return n

    @staticmethod
    def angle_of_departure(srf):
        AoD = srf.elevations.copy()
        AoD.values = srf['Z']/srf['distance']
        AoD.attrs=dict( description = "Angle of departure")

        srf['AoD'] = AoD
        return AoD.values

    @staticmethod
    def angle_of_arrival(srf, xi=0):
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

    
    def power(self, srf, t=None):


        G = self.gain(srf['AoD'], config['Radar']['Direction'][1])

        # self.angle_of_arrival(srf)
        d = srf['distance'].values
        AoA = srf['AoA'].values
 


        if isinstance(t, type(None)):
            t = np.arange(self.find_tmin(srf), self.find_tmax(srf), timp/4) 
        

        # P = np.zeros((t.size, *d.shape), dtype=float)
        P = np.zeros((t.size, d.shape[0]), dtype=float)
        numba_power(t, d, G, AoA, P)

        srf['pulse'] = dataset.pulse(P, srf['time'], t)

        return P
    
    def image(self, srf):
        srf['power'] = srf.elevations.copy()
        G = self.gain(srf['AoD'], config['Radar']['Direction'][1])
        d = srf['distance']
        srf['power'] = G**2/d**4

    
    def find_rho(self, srf):
        z = config['Radar']['Position'][-1]
        sigmaxx = srf['VoS'].values[0,:].max()
        theta = lambda rho: np.arctan(rho/z)
        xmax = config["Surface"]["LimitsOfModeling"][0][1]
        
        eps = np.deg2rad(1)
        if sigmaxx != 0:
            Wxx = lambda x: 1/np.sqrt(2*np.pi*sigmaxx**2) * np.exp(-x**2/(2*sigmaxx**2))
            probality = 1.49e-16 # вероятность того, что наклон примет значение theta
            F = lambda rho: sp.integrate.quad(Wxx, a=np.tan(theta(rho)-eps), b=np.tan(theta(rho)+eps))[0] - probality
            try:
                root = sp.optimize.root_scalar(F, bracket=[0, xmax]).root
            except:
                root = xmax/2

            return root
        else: return xmax/2
    
    def find_tmax(self, srf):
        z = config['Radar']['Position'][-1]
        c = config['Constants']['WaveSpeed']
        tmax = np.sqrt(self.find_rho(srf)**2 + (z-srf['elevations'].min())**2)/c
        return 1.04*tmax


    def find_tmin(self, srf):
        z = config['Radar']['Position'][-1]
        c = config['Constants']['WaveSpeed']
        tmin = (z - srf['elevations'].max())/c
        return tmin*0.99






    

# @guvectorize(
#     ["void(float64[:], float64[:,:,:], float64[:,:,:], float64[:,:,:], float64[:,:,:,:])"],
#     "(n), (x,y,t), (x,y,t), (x,y,t) -> (n,x,y,t)", forceobj=True, target='parallel'
# )
# def numba_power(t, Rabs, G, theta0, result, ):
#     tau = Rabs / c
#     t = np.expand_dims(t, axis=(0,1,2) ).T

#     anglemask = ( np.abs(np.arccos(np.cos(theta0)) ) <= np.deg2rad(1) )
#     timemask = (0 <= t - tau) & (t - tau <= timp) 

#     np.power(G/Rabs, 2, where=anglemask & timemask, out=result)
#     return result


@guvectorize(
    ["void(float64[:], float64[:,:,:], float64[:,:,:], float64[:,:,:], float64[:,:])"],
    "(n), (t,x,y), (t,x,y), (t,x,y) -> (n,t)", target='parallel'
)
def numba_power(t, Rabs, G, theta0, result):

    for m in range(t.size):
        for n in range(Rabs.shape[0]):
            for i in range(Rabs.shape[1]):
                for j in range(Rabs.shape[2]):
                    ind = (n,i,j)
                    tau = Rabs[ind]/c
                    anglemask = ( np.abs(np.arccos(np.cos(theta0[ind])) ) <= np.deg2rad(1) )
                    timemask = (0 <= t[m] - tau) & (t[m] - tau <= timp) 
                    if timemask and anglemask:
                        result[m, n] += (G[ind]/Rabs[ind])**4/2

# @cuda.jit
# def cuda_power(t, Rabs, G, theta0, result):
#     n, m = cuda.grid(2)
#     if m > t.size or n > Rabs.shape[0]:
#         return

#     for i in range(Rabs.shape[1]):
#         for j in range(Rabs.shape[2]):
#             tau = Rabs[n, i, j]/c
#             timemask = (0 <= t[m] - tau) & (t[m] - tau <= timp)
#             if timemask:
#                 result[m, n] += (G[n, i, j]/Rabs[n, i, j])**4/2