import numpy as np
import re
import os
import pandas as pd

from scipy.optimize import curve_fit
from scipy.special import erf


from  .. import config

import logging
logger = logging.getLogger(__name__)

def get_files(file, **kwargs):
    """
    Рекурсивный поиск данных по регулярному выражению 
    """
    path, file = os.path.split(file)

    path = os.path.abspath(path)
    rx = re.compile(file)


    _files_ = []
    for root, dirs, files in os.walk(path, **kwargs):
        for file in files:
            tmpfile = os.path.join(root,file)
            _files_ += rx.findall(tmpfile)
    
    for file in _files_:
        logger.info("Found file: %s" % file)

    return _files_

class pulse(object):
    def __init__(self, config, **kwargs):
        # Скорость света/звука
        self.c = config['Constants']['WaveSpeed']
        self.tau = config["Radar"]["ImpulseDuration"]
        self.delta = np.deg2rad(config["Radar"]["GainWidth"])

        if 'file' in kwargs:
            df = pd.read_csv(kwargs['file'], sep="\s+", comment="#")
            self.time = df.iloc[:,0].values
            self.power = df.iloc[:,1].values
        elif 't' in kwargs and 'P' in kwargs:
            self.time = kwargs["t"]
            self.power = kwargs["P"]

        self.curve_fit()

    def curve_fit(self):
        pass

class brown(pulse):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

    @staticmethod
    def pulse(t, A, alpha, tau, sigma_l, T):
        """
        Точная аппроксимация формулы Брауна.
        В отличии от Брауна не привязяна к абсолютному времени. 
        См. отчет по Ростову за 2020 год

        """
        return A * np.exp( -alpha * (t-tau) ) * (1 + erf( (t-tau)/sigma_l ) ) + T

    @property
    def height(self):
        """
        Вычисление высоты от антенны до поверхности воды
        """
        # Скорость света/звука [м/с]
        tau = self.popt[2] 
        c = self.c
        return tau*c/2

    @property
    def swh(self):

        """
        Вычисление высоты значительного волнения
        """
        # Скорость света/звука [м/с]
        c = self.c
        
        # Длительность импульса [с]
        sigma_l = self.popt[3]
        T = config["Radar"]["ImpulseDuration"]
        theta = np.deg2rad(config["Radar"]["GainWidth"])
        sigma_p = 0.425 * T 
        sigma_c = sigma_l/np.sqrt(2)
        sigma_s = np.sqrt((sigma_c**2 - sigma_p**2))*c/2
        factor = np.sqrt(0.425/(2*np.sin(theta/2)**2/np.log(2)))
        # return 4*sigma_s * factor
        return 4*sigma_s


    def curve_fit(self, **kwargs):

        t = self.time
        power = self.power

        p0 = [power.max()/2, 1, (t.max() + t.min())/2, (t[-1]-t[0])/t.size, 0]
        self.popt, self.pcov = curve_fit(self.pulse, 
                            xdata=t,
                            ydata=power,
                            p0=p0,
                            **kwargs
                        )
        

class karaev(pulse):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

    @staticmethod
    def slopes_coeff(varslopes, H, delta):
        # Вычисление коэффициента Ax через дисперсию наклонов, высоту и ширину ДН
        return 1/(2*varslopes*H**2) + 5.52/(delta**2*H**2)

    @property
    def varslopes(self):
        # Вычисление дисперсии наклонов через Ах, высоту и ширину ДН
        H = self.height
        slopes_coeff = self.popt[1]
        delta = self.delta
        invvarslopes = 2*(slopes_coeff*H**2 - 5.52/delta**2)
        return 1/invvarslopes
        # return 1/ ( 2 * ( slopes_coeff * H**2 - 5.52/delta**2)  )

    @property
    def height(self):
        return self.popt[2]

    @property
    def varelev(self):
        return self.popt[0]

    @property
    def swh(self):
        return 4*np.sqrt(self.popt[0])

    
    def curve_fit(self):
        t = self.time
        power = self.power
        Hmin = t.min()*self.c
        Hmax = t.max()*self.c
        H0 = t[np.argmax(power)]*self.c
        sigma0max = np.max(power)

        self.popt, self.pcov = curve_fit(self.pulse, 
                    xdata=t,
                    ydata=power,
                    p0=[0.002, 0, H0, sigma0max/2],
                    bounds = ( 
                                (0.002, 0, Hmin, 0),
                                (5.5, np.inf, Hmax, np.inf)
                    )
                )

    def pulse(self, t, varelev, slopes_coeff, H, sigma0):


        c = self.c
        t_pulse = self.tau

        t = t.copy()
        t -= H/c

        F1 =  np.exp(-slopes_coeff*H*c*t + 2*varelev*slopes_coeff**2*H**2) * \
            (1 - erf( slopes_coeff*H*np.sqrt(2*varelev) + (t_pulse - t)*c/(2*np.sqrt(2*varelev))) )

        # F2 = erf((t_pulse - t)*c/(2*np.sqrt(2*varelev))) +  erf(t*c/(2*np.sqrt(2*varelev)))

        # F3 = np.exp(-slopes_coeff*H*c*t + 2*varelev*slopes_coeff**2*H**2) * \
        #     (
        #         erf( slopes_coeff*H*np.sqrt(2*varelev) + (t_pulse - t)*c/(2*np.sqrt(2*varelev)))
        #         -
        #         erf( slopes_coeff*H*np.sqrt(2*varelev) - t*c/(2*np.sqrt(2*varelev)))
        #     )
        
        F2, F3 = 0, 0

        return sigma0/2 * (F1 + F2 + F3)




    
    

