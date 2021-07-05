import matplotlib.pyplot as plt
import numpy as np
from numba import vectorize, float64

from .decorators import dispatcher, ufunc
from .. import config
from . import dispersion
from .integrate import Integrate
from .twoscaledmodel import TwoScaledModel


import logging
logger = logging.getLogger(__name__)

g = config['Constants']['GravityAcceleration']

"""
Спектр ветрового волнения и зыби. Используется при построении морской поверхности. 
"""
class spectrum(Integrate, TwoScaledModel):

    def __init__(self, **kwargs):
        for Key, Value in kwargs.items():
            if isinstance(Value, dict):
                for key, value in Value.items():
                    config[Key][key] = value




        super(Integrate, self).__init__()

        # two-scaled model
        self.__tsm__ = False
        if config['Surface']['TwoScaledModel']:
            self.__tsm__= True
            super(TwoScaledModel, self).__init__()




        self._x = config['Surface']['NonDimWindFetch']
        self._U = config['Wind']['Speed']
        self._wavelength = config['Radar']['WaveLength']
        self._band = config["Radar"]["WaveLength"]
        self.k_m = None
        self.peak = None
        self.KT = np.array([1.49e-2, 2000])
        self._k = np.logspace( np.log10(self.KT.min()), np.log10(self.KT.max()), 10**3+1)

    @property
    @dispatcher
    def bounds(self):
        return self.KT

    @bounds.setter
    def bounds(self, value):
        pass

    @property 
    @dispatcher
    def k(self):
        return self._k

    @dispatcher
    def __call__(self, k=None, phi=None, kind="spec"):
        
        if np.array([k]).all() == None:
            k = self.k

        if not isinstance(k, np.ndarray):
            k = np.array([k])

        k = np.abs(k)
        limit = np.array([0, *self.limit_k, np.inf])

        spectrum1d = np.zeros(k.size, dtype=np.float64)

        for j in range(1, limit.size):
            self.piecewise_spectrum(j-1, k, where = (limit[j-1] <= k) & (k <= limit[j]), out=spectrum1d)
        
        if not isinstance(phi, type(None)):
            spectrum = self.azimuthal_distribution(k, phi, dtype='Wind')
            if kind == "spec":
                spectrum = spectrum1d * spectrum.T
        else: 
            spectrum = spectrum1d
        

        if not isinstance(phi, type(None)) and config['Swell']['Enable']:
            swell1d = self.swell_spectrum(k)
            swell = swell1d * self.azimuthal_distribution(k, phi, dtype="Swell").T
            spectrum = spectrum + swell.T

        return spectrum

    def peakUpdate(self):
        logger.info('Refresh old spectrum parameters')
        x = config['Surface']['NonDimWindFetch']
        U = config['Wind']['Speed']
        Udir = config['Wind']['Direction']

        logger.info('Start modeling with U=%.1f, Udir=%.1f, x=%.1f, lambda=%s' % (U, Udir, x, str(config["Radar"]["WaveLength"])))

        # коэффициент gamma (см. спектр JONSWAP)
        self._gamma = self.Gamma(x)
        # коэффициент alpha (см. спектр JONSWAP)
        self._alpha = self.Alpha(x)

        # координата пика спектра по волновому числу
        self.peak = (self.Omega(x) / U)**2  * g
        self.k_m = self.peak

        # координата пика спектра по частоте
        self.omega_m = self.Omega(x) * g / U
        # длина доминантной волны
        self.lambda_m = 2 * np.pi / self.k_m
        logger.info('Set peak\'s parameters: kappa=%.4f, omega=%.4f, lambda=%.4f' % (self.peak, self.omega_m, self.lambda_m))




        limit = np.zeros(5)
        limit[0] = 1.2 * self.omega_m
        limit[1] = ( 0.8*np.log(U) + 1 ) * self.omega_m
        limit[2] = 20.0
        limit[3] = 81.0
        limit[4] = 500.0

        __limit_k = np.array([dispersion.k(limit[i]) for i in range(limit.size)])
        self.limit_k = __limit_k[np.where(__limit_k <= 2000)]
        del __limit_k, limit

        # массив с границами моделируемого спектра.
        waveLength = config['Radar']["WaveLength"]
        if  waveLength != None and self.__tsm__:
            logger.info('Calculate bounds of modeling for radar wave lenghts: %s' %  str(waveLength) )
            self.KT = self.kEdges(waveLength)

        elif not self.__tsm__:
            self.KT = np.array([0, 2000])

        


        self._k = np.logspace( np.log10(self.peak/4), np.log10(self.KT.max()), 10**3+1)

        logger.info('Set bounds of modeling %s' % str(np.round(self.KT, 2)))



    @staticmethod
    def __az_exp_arg__(k, km):
        k[np.where(k/km < 0.4)] = km * 0.4
        b=(
            -0.28+0.65*np.exp(-0.75*np.log(k/km))
            +0.01*np.exp(-0.2+0.7*np.log10(k/km))
        )
        return b

    @staticmethod
    def __az_normalization__(B):
        return B/np.arctan(np.sinh(2*np.pi*B))

    def azimuthal_distribution(self, k, phi, dtype="Wind"):
        if not isinstance(k, np.ndarray):
            k = np.array([k])

        if not isinstance(phi, np.ndarray):
            phi = np.array([phi])

        # Функция углового распределения
        km = self.peak


        phi = np.angle(np.exp(1j*phi))
        phi -= np.deg2rad(config[dtype]["Direction"])
        phi = np.angle(np.exp(1j*phi))


        B0 = np.power(10, self.__az_exp_arg__(k, km))

        A0 = self.__az_normalization__(B0)

        phi = phi[np.newaxis]
        Phi = A0/np.cosh(2*B0*phi.T )
        return Phi.T




    def JONSWAP(self, k):
        return JONSWAP_vec(k, self.peak, self._alpha, self._gamma)

    # Безразмерный коэффициент Gamma
    @staticmethod
    def Gamma(x):
        if x >= 20170:
            return 1.0
        gamma = (
            +5.253660929
            + 0.000107622*x
            - 0.03778776*np.sqrt(x)
            - 162.9834653/np.sqrt(x)
            + 253251.456472*x**(-3/2)
        )
        return gamma

    # Безразмерный коэффициент Alpha
    @staticmethod
    def Alpha(x):
        if x >= 20170:
            return 0.0081
        else:
            alpha = (+0.0311937
                - 0.00232774 * np.log(x)
                - 8367.8678786/x**2
            )
        return alpha

    # Вычисление безразмерной частоты Omega по безразмерному разгону x
    @staticmethod
    def Omega(x):
        if x >= 20170:
            return 0.835

        omega_tilde = (0.61826357843576103
                        + 3.52883010586243843e-06*x
                        - 0.00197508032233982112*np.sqrt(x)
                        + 62.5540113059129759/np.sqrt(x)
                        - 290.214120684236224/x
                        )
        return omega_tilde



    @ufunc(3, 1)
    def piecewise_spectrum(self, n, k):
        power = [   
                    4, 
                    5, 
                    7.647*np.power(self._U, -0.237), 
                    0.0007*np.power(self._U, 2) - 0.0348*self._U + 3.2714,
                    5,
                ]

        if n == 0:
            return self.JONSWAP(k)

        else:
            omega0 = dispersion.omega(self.limit_k[n-1])
            beta0 = self.piecewise_spectrum(n-1, self.limit_k[n-1]) * \
                omega0**power[n-1]/dispersion.det(self.limit_k[n-1])
            
            omega0 = dispersion.omega(k)
            return beta0 * np.power(omega0, -power[n-1]) * dispersion.det(k)
    
    @ufunc(2,1)
    def swell_spectrum(self, k):

        omega_m = self.Omega(20170) * g/config['Swell']['Speed']
        W = np.power(omega_m/dispersion.omega(k), 5)

        sigma_sqr = 0.0081 * g**2 * np.exp(-0.05) / (6 * omega_m**4)

        spectrum = 6 * sigma_sqr * W / \
            dispersion.omega(k) * np.exp(-1.2 * W) * dispersion.det(k)
        return spectrum



args = [float64 for i in range(4)]
@vectorize([float64(*args)])
def JONSWAP_vec(k, km, alpha, gamma):
    if k == 0:
        return 0

    if k >= km:
        sigma = 0.09
    else:
        sigma = 0.07


    Sw = (alpha/2 *
            np.power(k, -3) * 
            np.exp(-1.25 * np.power(km/k, 2)) *
            np.power(gamma,
                    np.exp(- (np.sqrt(k/km) - 1)**2 / (2*sigma**2))
                )
            )
    return Sw
