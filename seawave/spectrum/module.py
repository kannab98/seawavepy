import logging

import matplotlib.pyplot as plt
import numpy as np
import math as m
from scipy import fft, interpolate, optimize, integrate
from scipy.special import erf
from numba import vectorize, float64

from ..__decorators__ import spectrum_dispatcher as dispatcher
from ..__decorators__ import ufunc
# from .integrate import quad as specquad
from .. import config



g = config['Constants']['GravityAcceleration']
logger = logging.getLogger(__name__)



class dispersion:
    # коэффициенты полинома при степенях k
    p = [74e-6, 0, g, 0]
    # f(k) -- полином вида:
    # p[0]*k**3 + p[1]*k**2 + p[2]*k + p[3]
    f = np.poly1d(p)
    # df(k) -- полином вида:
    # 3*p[0]*k**2 + 2*p[1]*k + p[2]
    df = np.poly1d(np.polyder(p))

    @staticmethod
    def omega(k):
        """
        Решение прямой задачи поиска частоты по известному волновому числу 
        из дисперсионного соотношения
        """
        k = np.abs(k)
        return np.sqrt( dispersion.f(k))
        
    @staticmethod
    def k(omega):
        """
        Решение обратной задачи поиска волнового числа по известной частоте
        из дисперсионного соотношения

        Поиск корней полинома третьей степени. 
        Возвращает сумму двух комплексно сопряженных корней
        """
        p = dispersion.p
        p[-1] = omega**2
        k = np.roots(p)
        return 2*np.real(k[0])

    @staticmethod
    def det(k):
        """
        Функция возвращает Якобиан при переходе от частоты к
        волновым числам по полному дисперсионному уравнению
        """
        return dispersion.df(k) / (2*dispersion.omega(k))







"""
Спектр ветрового волнения и зыби. Используется при построении морской поверхности. 
"""
class __spectrum__(object):

    def __init__(self):
        self._x = config['Surface']['NonDimWindFetch']
        self._U = config['Wind']['Speed']
        self._wavelength = config['Radar']['WaveLength']
        self._band = config["Radar"]["WaveLength"]
        self.k_m = None
        self.peak = None
        self.KT = np.array([1.49e-2, 2000])
        self.k = np.logspace( np.log10(self.KT.min()), np.log10(self.KT.max()), 10**3+1)

        # self.peakUpdate(True)

    @property
    @dispatcher()
    def bounds(self):
        return self.KT

    @dispatcher()
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



    def curv_criteria(self, band='Ku'):
        speckwargs = dict(radar_dispatcher=False)
        # Сейчас попробуем посчитать граничное волновое число фактически из экспериментальных данных
        # Из работы Панфиловой известно, что полная дисперсия наклонов в Ku-диапазоне задается формулой

        # Дисперсия наклонов из статьи
        if band == "Ku":
            # var = lambda U10: 0.0101 + 0.0022*np.sqrt(U10)
            var = lambda U10: 0.0101 + 0.0022*U10
            radarWaveLength = 0.022

        elif band == "Ka":
            var = lambda U10: 0.0101 + 0.0034*U10
            radarWaveLength = 0.008

        # Необходимо найти верхний предел интегрирования, чтобы выполнялось равенство
        # интеграла по спектру и экспериментальной формулы для дисперсии
        # Интеграл по спектру наклонов


        epsabs = 1.49e-6
        Func = lambda k_bound: self.quad(2, 0, 0, k_bound,  speckwargs=speckwargs, epsabs=epsabs, ) - var(self._U)
        # Поиск граничного числа 
        # (Ищу ноль функции \integral S(k) k^2 dk = var(U10) )
        opt = optimize.root_scalar(Func, bracket=[0, 2000]).root

        # Значение кривизны 
        curv0 = self.quad(4,0,0,opt, epsabs=epsabs)

        # Критерий выбора волнового числа
        eps = np.power(radarWaveLength/(2*np.pi) * np.sqrt(curv0), 1/3)

        return eps
    
    def __find_k_bound(self, radarWaveLength,  **kwargs):
        speckwargs = dict(radar_dispatcher=False)
        eps = self.curv_criteria()
        Func = lambda k_bound: np.power( radarWaveLength/(2*np.pi) * np.sqrt(self.quad(4,0,0, k_bound, speckwargs=speckwargs, epsabs=1.49e-6, )), 1/3 ) - eps
        # root = optimize.root_scalar(Func, bracket=[self.KT[0], self.KT[-1]]).root
        root = optimize.root_scalar(Func, bracket=[0, 2000]).root
        return root


    def kEdges(self, band, ):

        """
        Границы различных электромагнитных диапазонов согласно спецификации IEEE
        
        Band        Freq, GHz            WaveLength, cm         BoundaryWaveNumber, 
        Ka          26-40                0.75 - 1.13            2000 
        Ku          12-18                1.6  - 2.5             80 
        X           8-12                 2.5  - 3.75            40
        C           4-8                  3.75 - 7.5             10

        """
        bands = {"C":1, "X":2, "Ku":3, "Ka":4}


        # if self.k_m == None:
        #     self.__call__dispatcher__(radar_dispatcher=False)

        k_m = self.k_m

        if isinstance(band, str):
            bands_edges = [

                lambda k_m: k_m/4,

                lambda k_m: (
                    2.74 - 2.26*k_m + 15.498*np.sqrt(k_m) + 1.7/np.sqrt(k_m) -
                    0.00099*np.log(k_m)/k_m**2
                ),


                lambda k_m: (
                    25.82 + 25.43*k_m - 16.43*k_m*np.log(k_m) + 1.983/np.sqrt(k_m)
                    + 0.0996/k_m**1.5
                ),


                lambda k_m: (
                    + 68.126886 + 72.806451 * k_m  
                    + 12.93215 * np.power(k_m, 2) * np.log(k_m) 
                    - 0.39611989*np.log(k_m)/k_m 
                    - 0.42195393/k_m
                ),

                lambda k_m: (
                    #   24833.0 * np.power(k_m, 2) - 2624.9*k_m + 570.9
                    2000
                )

            ]
            edges = np.array([ bands_edges[i](k_m) for i in range(bands[band]+1)])
        else:

            edges = np.zeros(len(band)+1)
            edges[0] = k_m/4
            for i in range(1, len(edges)):
                if band[i-1] == 0:
                    edges[i] = 2000
                else:
                    edges[i] = self.__find_k_bound(band[i-1], )

        # edges = np.array([ bands_edges[i](k_m) for i in range(bands[band]+1)])
        return edges
    



    def peakUpdate(self, radar_dispatcher=True):
        logger.info('Refresh old spectrum parameters')
        x = config['Surface']['NonDimWindFetch']
        U = config['Wind']['Speed']
        Udir = config['Wind']['Direction']

        logger.info('Start modeling with U=%.1f, Udir=%.1f, x=%.1f,' % (U, Udir, x))

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
        if  waveLength != None and radar_dispatcher:
            logger.info('Calculate bounds of modeling for radar wave lenghts: %s' %  str(waveLength) )

            self.KT = self.kEdges(waveLength)

        elif radar_dispatcher == False:
            self.KT = np.array([0, np.inf])

        else: 
            self.KT = self.kEdges(rc.surface.band)
        


        self.k = np.logspace( np.log10(self.peak/4), np.log10(self.KT.max()), 10**3+1)

        logger.info('Set bounds of modeling %s' % str(np.round(self.KT, 2)))



    def plot(self, stype="ryabkova"):
        S = self.get_spectrum(stype)
        edges = np.log10(self.KT)
        k = np.logspace(edges[0], edges[1], 1000)
        return plt.loglog(k, S(k))

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
    
    def quad(self, a,b, k0=None, k1=None,  speckwargs=dict(), **quadkwargs):
    # def quad(*args, **kwargs):
        if k0==None:
            k0 = self.bounds[0]

        if k1==None:
            k1 = self.bounds[-1]

        S = lambda k: self.__call__(k, **speckwargs) * k**a * dispersion.omega(k)**b
        var = integrate.quad(S, k0, k1, **quadkwargs)[0]

        return var
        # return specquad(*args, **kwargs)


    def dblquad(self, a, b, c, k0=None, k1=None, phi0=None, phi1=None,  speckwargs=dict(), **quadkwargs):
        limit = np.array([self.KT[0], *self.bounds, self.KT[-1]])

        if k0==None:
            k0 = self.KT[0]

        if k1==None:
            k1 = self.KT[-1]

        if phi0==None:
            phi0 = -np.pi
        
        if phi1==None:
            phi1 = np.pi
        

        S = lambda phi, k:  self.__call__(k, phi, **speckwargs) *  k**(a+b-c) * np.cos(phi)**a * np.sin(phi)**b
        var = integrate.dblquad( S,
                a=k0, b=k1,
                gfun=lambda phi: phi0, 
                hfun=lambda phi: phi1, **quadkwargs)
        
        return var[0]

    def cov(self):

        cov = np.zeros((2, 2))
        cov[0, 0] = self.dblquad(2, 0, 0)
        cov[1, 1] = self.dblquad(0, 2, 0)
        cov[1, 0] = self.dblquad(1, 1, 0)
        cov[0, 1] = cov[1, 0]

        return cov

    def correlate(self, rho):


    # def quad(self, a,b, k0=None, k1=None):
        S = lambda k, rho: self.get_spectrum()(k) *  np.cos(k*rho)
        limit = np.array([self.KT[0], *self.limit_k, self.KT[-1]])
        # k0 = np.logspace( np.log10(self.KT[0]), np.log10(self.KT[-1]), 2**10 + 1)
        k0 = np.linspace( self.KT[0], self.KT[-1], 2**11 + 1)
        k0[0] = self.KT[0]
        k0[-1] = self.KT[-1]

        integral=np.zeros(len(rho))
        for i in range(len(rho)):
            # integral[i] = integrate.quad(S, limit[0], limit[-1],args=(rho[i],))[0]
            # integral[i] = integrate.romb(S(k0, rho[i]), np.diff(k0[:2]))
            integral[i] =integrate.trapz(S(k0, rho[i]), k0)

        return integral

    def fftstep(self, x):
        return np.pi/x


    def fftfreq(self, xmax):
        step = self.fftstep(xmax)
        k = np.arange(-self.KT[-1], self.KT[-1], step)
        d = np.diff(k[0:2])
        return np.linspace(-np.pi/d, +np.pi/d, k.size)

    def fftcorrelate(self, xmax, a=0, b=0, c=1):

        xkorr = 2*np.pi/self.KT[0]

        x = xmax
        # if xmax <= xkorr:
        #     x = 2*xkorr

        step = self.fftstep(x)
        k = np.arange(-self.KT[-1], self.KT[-1], step)
        D = k.max()

        S = lambda k: k**a * dispersion.omega(k)**b * self.ryabkova(k)**c

        S = fft.fftshift(S(k))
        K = fft.ifft(S) * D

        ind = int(np.ceil(S.size/2))
        # K = K[:ind]
        K = fft.fftshift(K)
        return K

    # def spectrum_cwm(self, xmax):
    #     sigma = np.zeros(2)
    #     sigma[0] = self.quad(0,0)
    #     sigma[1] = self.quad(1,0)

    #     step = self.fftstep(xmax)
    #     k = np.arange(-self.KT[-1], self.KT[-1], step)
    #     f = self.fftcorrelate(xmax)
    #     S = np.zeros((2, k.size), dtype=np.complex64)
    #     S[0,:] = 2*(sigma[0] - f)
    #     S[1,:] = 2*(sigma[1] - self.fftcorrelate(xmax, a=1))

    #     dC = np.zeros((2, k.size), dtype=np.complex64)
    #     dC[0] = 1j*k*f
    #     dC[1] = -(k)**2*f

    #     spec = fft.fft(
    #         + np.exp( -(k*sigma[0])**2 ) 
    #             * (sigma[0] - sigma[1]**2)  
    #         - np.exp(-k**2/2*S[0]) 
    #             * (   
    #                 + 1/2 * S[0] * (1 - 2j*k*dC[0] - dC[1] - (k*dC[0])**2)
    #                 - 1/4 * S[1]**2
    #               )

    #     )
    #     return fft.fftshift(spec)

    def pdf_heights(self, z, dtype="default"):
        sigma0 = self.quad(0,0)
        if dtype == 'default':
            return 1/np.sqrt(2*np.pi*sigma0) * np.exp(-1/2*z**2/sigma0)

        if dtype == 'cwm':
            sigma0 = self.quad(0,0)
            sigma1 = self.quad(1,0)
            return self.pdf_heights(z, 'default') * (1 - sigma1/sigma0*z)

    def pdf_slopes(self, z, dtype="default"):

        # sigma = self.quad(2,0)
        sigma = 0.0
        if dtype == 'default':
            return 1/np.sqrt(2*np.pi*sigma) * np.exp(-1/2*z**2/sigma)

        if dtype == 'cwm':
            return (
                + np.exp(-1/(2*sigma))/(np.pi*(1+z**2)**2)
                + 1/np.sqrt(2*np.pi*sigma)
                * (sigma*(1+z**2)+1)/(1+z**2)**(5/2)
                * erf(1/np.sqrt(2*sigma*(1+z**2)))
                * np.exp(-1/(2*sigma)*(z**2/(1+z**2)))
            )








    @staticmethod
    def __wind__(x, z):
        f = lambda x, z: x/0.4 * np.log(z/(0.684/x + 428e-7* x**2 - 443e-4))
        return f(x,z)



    def  find_friction(self):

        # Finds zeroes, X - friction velocity, Z - height, Karman's constant = 0.4
        z = 10
        f = lambda x: x/0.4 * np.log(z/(0.684/x + 428e-7* x**2 - 443e-4))
        root = optimize.ridder( f, 1e-8, 80)
        return root

        
    def kontrast(self, k, beta0, sigma_w, sigma_m, e):

        g = 981
        R = 1
        nu = 0.01

        def gamma(k, e, omega):

            Rw = R * np.power(omega, 2)

            x1 = 2*nu*k**2/omega 
            x2 = e * np.power(k, 3) * np.sqrt(x1) / Rw
            x3 = 1 * x2 * e * np.power(k, 3) / (2 * x1 * Rw)
            x4 = 2 * x2
            x5 = 2 * x2 * e * np.power(k, 3) / (1 * np.sqrt(x1) * Rw)

            return 2*nu*k**2*(x1 - x2 + x3)/(x1 - x4+ x5)
        
        omega_w = np.sqrt(g*k + sigma_w/R * np.power(k, 3))
        omega_m = np.sqrt(g*k + sigma_m/R * np.power(k, 3))



        G0 = gamma(k, 0, omega_w)
        G1 = gamma(k, e, omega_m)


        uftr = self.find_friction()
        beta = beta0*np.power(k*uftr, 2)/omega_w

        # kontrast = 0.1 * np.ones(k.size)
        # slick_kontrast = 0

        # flag = True
        # for i in range(k.size):
        #     if (beta[i] > 2*G0[i]) & (beta[i] > 2*G1[i]):
        #         kontrast[i] = (beta[i] - 2*G1[i])/(beta[i] - 2*G0[i])

        #         print(kontrast[i])
        #     elif (beta[i] < 2*G0[i]) & (beta[i] < 2*G1[i]):
        #         kontrast[i] = (2*G0[i] - beta[i]) / (2*G1[i] - beta[i])
        #         print(kontrast[i])
        #     else:
        #         flag = False
            
        #     if flag:
        #         slick_kontrast = kontrast[i]
        #     else:
        #         kontrast[i] *= slick_kontrast
        #         print(kontrast[i])



        # ind = np.where( (beta > 2*G0) & (beta > 2*G1) )[0]
        # if len(ind) != 0:
        #     kontrast[ind] = \
        #         (beta - 2*G1[ind])/(beta - 2*G0[ind])

        # ind = np.where( (beta < 2*G0) & (beta < 2*G1) )[0]
        # print(ind)
        # if len(ind) != 0:
        #     kontrast[ind] = \
        #         (beta - 2*G0[ind])/(beta - 2*G1[ind])


        # return kontrast
        return np.abs( (beta - np.minimum(G0, G1) ) / (beta - np.maximum(G0, G1)) )
        
    
    def with_slick(self, k):
        beta0 = rc.slick.beta0
        sigma_w = rc.slick.sigmaWater
        sigma_m = rc.slick.sigmaOil
        sigma = rc.slick.sigma

        return self.kontrast(k/100, beta0, sigma_w, sigma_m, sigma)






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
