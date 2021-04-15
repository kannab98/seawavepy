import matplotlib.pyplot as plt
import numpy as np
import re
import os
import pandas as pd

from numpy import pi
from scipy.optimize import curve_fit
from scipy.special import erf
from pandas import read_csv

import numpy

from  .. import config#, spectrum, surface

import logging
logger = logging.getLogger(__name__)
class __retracking__():
    """
    Самый простой способ использовать этот класс, после вызова его конструктора
    это использовать метод класса from_file. 
    Перед этим имеет смысл указать параметры конкретной системы радиолокации.
    Для класса пока что нужны только два параметра:
        1. Скорость света (звука)
        2. Длительность импульса

    Задать их можно с помощью  объекта rc:
    >>> from modeling import rc
    >>> rc.constants.lightSpeed = 1500 # м/с
    >>> rc.antenna.impulseDuration = 40e-6 # с

    Или же изменить файл rc.json и положить его в рабочую директорию.

    Пример простого использования:
    # Импорт модуля
    >>> from modeling import rc
    >>> from modeling.retracking import Retracking 
    >>> retracking = Retracking()
    # Ретрекинг для всех файлов, заканчивающихся на .txt в директории impulses
    >>> df0, df = retracking.from_file(path.join("impulses", ".*.txt"))

    

    """
    def __init__(self, **kwargs):
        # Скорость света/звука
        self.c = config['Constants']['WaveSpeed']
        # Длительность импульса в секундах
        self.T = config['Radar']['ImpulseDuration']


    def from_file(self, file, config):
        """
        Поиск импульсов в файлах по регулярному выражению. 

        Вычисление для всех найденных коэффициентов 
        аппроксимации формулы ICE. 
        
        Оценка SWH и высоты до поверхности.

        Экспорт данных из найденных файлов в output.xlsx в лист raw

        Эспорт обработанных данных в output.xlsx в лист brown

        """
        
        path, file = os.path.split(file)

        path = os.path.abspath(path)
        rx = re.compile(file)


        _files_ = []
        for root, dirs, files in os.walk(path):
            for file in files:
                tmpfile = os.path.join(root,file)
                _files_ += rx.findall(tmpfile)


        # print(_files_)
        columns = pd.MultiIndex.from_product([ _files_, ["t", "P"] ], names=["file", "data"])
        df0 = pd.DataFrame(columns=columns)

        df = pd.DataFrame(columns=["SWH", "H", "VarSlopes", "Amplitude", "Alpha", "Epoch", "Sigma", "Noise"], index=_files_)

        for i, f in enumerate(_files_):
            sr = pd.read_csv(os.path.join(path, f), sep="\s+", comment="#")
            df0[f, "t"] = sr.iloc[:, 0]
            df0[f, "P"] = sr.iloc[:, 1]

            popt = self.pulse(sr.iloc[:, 0].values, sr.iloc[:, 1].values)

            df.iloc[i][3:] = popt
            df.iloc[i][0] = self.swh(df.iloc[i]["Sigma"])
            df.iloc[i][1] = self.height(df.iloc[i]["Epoch"])

            H = df.iloc[i]['H']

            slopes_coeff = self.slopes_coeff(df.iloc[i]['Alpha'], H, self.c)
            print(slopes_coeff, H, np.deg2rad(config['Radar']['GainWidth']))
            df.iloc[i][2] = self.varslopes(slopes_coeff, H, np.deg2rad(config['Radar']['GainWidth']))

        excel_name = os.path.join(config['Dataset']['RetrackingFileName'])

        df.to_excel(excel_name, sheet_name='brown')

        with pd.ExcelWriter(excel_name, mode='a', engine='openpyxl') as writer:  
            print(excel_name)
            df0.to_excel(writer, sheet_name='raw')

        return df0, df
        

    @staticmethod
    def leading_edge(t, pulse, dtype="needed"):
        """
        Аппроксимация экспонентой заднего фронта импульса. 
        dtype = "full" -- возвращает все коэффициенты аппроксимации
        dtype = "needed" -- возвращает коэффициенты аппроксимации,
                            необходимые для формулы Брауна

        """
        # Оценили положение максимума импульса
        n = np.argmax(pulse)
        # Обрезали импульс начиная с положения максимума
        pulse = np.log(pulse[n:])
        t = t[n:]
        line = lambda t,alpha,b: -alpha*t + b   
        # Аппроксимация
        popt = curve_fit(line, 
                            xdata=t,
                            ydata=pulse,
                            p0=[1e6,0],
                        )[0]

        if dtype == "full":
            return popt
        elif dtype == "needed":
            return popt[0]

    @staticmethod 
    def trailing_edge(t, pulse):
        """
        Аппроксимация функией ошибок переднего фронта импульса. 

        """

        # Оценили амплитуду импульса
        A0 = (max(pulse) - min(pulse))/2

        # Оценили положение максимума импульса
        n = np.argmax(pulse)

        # Обрезали импульс по максимум

        pulse = pulse[0:n]
        t = t[0:n]


        func = lambda t, A, tau, sigma_l, b:   A * (1 + erf( (t-tau)/sigma_l )) + b

        # Аппроксимация
        popt = curve_fit(func, 
                            xdata=t,
                            ydata=pulse,
                            p0=[A0, (t.max() + t.min())/2, (t[-1]-t[0])/t.size, 0])[0]

                            
        return popt


    
    @staticmethod
    def ice(t, A,alpha, tau, sigma_l,T):
        """
        Точная аппроксимация формулы Брауна.
        В отличии от Брауна не привязяна к абсолютному времени. 
        См. отчет по Ростову за 2020 год

        """
        return A * np.exp( -alpha * (t-tau) ) * (1 + erf( (t-tau)/sigma_l ) ) + T

    def pulse(self, t, pulse, **kwargs):

        logger.debug('Start retracking')
        try:
            alpha = self.leading_edge(t, pulse, dtype="needed")
        except:
            alpha = 1
        
        try:
            A, tau, sigma_l, b = self.trailing_edge(t, pulse)
        except:
            A = 1
            tau = 1
            sigma_l = 1
            b = 0

        p0 = [A, alpha, tau, sigma_l, b]
        popt = curve_fit(self.ice, 
                            xdata=t,
                            ydata=pulse,
                            p0=p0,
                            **kwargs
                        )[0]
        logger.info('Restore pulse parameters: h=%.4f, Hs=%.4f' % (self.height(popt[2]), self.swh(popt[3]) ))
        return popt 

    @staticmethod
    def varslopes(slopes_coeff, H, delta):
        return np.abs(1/ ( 2 * ( slopes_coeff * H**2 - 5.52/delta**2)  ))

    @staticmethod
    def slopes_coeff(alpha, H, c):
        return alpha/(H*c)


    def full_pulse(self, t, varelev, slopes_coeff, H, sigma0, t0, t_pulse=None, c=None):

        # Программа Караева считает только F1
        if t_pulse == None:
            t_pulse = self.T

        if c == None:
            c = self.c

        # t += H/c
# (np.exp(slopes_coeff*H*c*t_pulse) - 1)
        F1 =  np.exp(-slopes_coeff*H*c*(t-t0) + 2*varelev*slopes_coeff**2*H**2) * \
            (1 - erf( slopes_coeff*H*np.sqrt(2*varelev) + (t_pulse - t + t0)*c/(2*np.sqrt(2*varelev))) )

        # F2 = erf((t_pulse - t)*c/(2*np.sqrt(2*varelev))) +  erf(t*c/(2*np.sqrt(2*varelev)))

        # F3 = np.exp(-slopes_coeff*H*c*t + 2*varelev*slopes_coeff**2*H**2) * \
        #     (
        #         erf( slopes_coeff*H*np.sqrt(2*varelev) + (t_pulse - t)*c/(2*np.sqrt(2*varelev)))
        #         -
        #         erf( slopes_coeff*H*np.sqrt(2*varelev) - t*c/(2*np.sqrt(2*varelev)))
        #     )

        return sigma0/2 * ( F1 )



    
    def swh(self, sigma_l):

        """
        Вычисление высоты значительного волнения
        """
        # Скорость света/звука [м/с]
        c = self.c
        # Длительность импульса [с]
        T = self.T

        sigma_p = 0.425 * T
        sigma_c = sigma_l/np.sqrt(2)
        sigma_s = np.sqrt((sigma_c**2 - sigma_p**2))*c/2
        return 4*sigma_s
    

    def height(self, tau):
        """
        Вычисление высоты от антенны до поверхности воды
        """

        # Скорость света/звука [м/с]
        c = self.c
        return tau*c/2

    @staticmethod
    def emb(swh, U10, dtype = "Rostov"):
        """
        Поправка на состояние морской поверхности (ЭМ-смещение)
        """
        if dtype ==  "Rostov":
            emb = swh * (- 0.019 + 0.0027 * swh - 0.0037 * U10 + 0.00014 * U10**2)
            return emb

        elif dtype == "Chelton":
            coeff = np.array([0.0029, -0.0038, 0.000155 ])
            emb = [coeff[i]*U10**i for i in range(coeff.size)]
            EMB = 0
            for i in range(coeff.size):
                EMB += emb[i]
            return  -abs(EMB)


        elif dtype == "Ray":
            coeff = np.array([0.00666,  0.0015])
            emb = [coeff[i]*U10**i for i in range(coeff.size)]
            EMB = 0
            for i in range(coeff.size):
                EMB += emb[i]
            return  -abs(EMB)
        
        return None
    

