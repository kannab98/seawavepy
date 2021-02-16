import matplotlib.pyplot as plt
import numpy as np
import re
import os
import pandas as pd

from numpy import pi
from scipy.optimize import curve_fit
from scipy.special import erf
from pandas import read_csv
from modeling import rc #, spectrum, surface
import numpy


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
        self.c = rc.constants.lightSpeed
        # Длительность импульса в секундах
        self.T = rc.antenna.impulseDuration


    def from_file(self, file):
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
                _files_ += rx.findall(file)

        columns = pd.MultiIndex.from_product([ _files_, ["t", "P"] ], names=["file", "data"])
        df0 = pd.DataFrame(columns=columns)

        df = pd.DataFrame(columns=["SWH", "H", "Amplitude", "Alpha", "Epoch", "Sigma", "Noise"], index=_files_)

        for i, f in enumerate(_files_):
            sr = pd.read_csv(os.path.join(path, f), sep="\s+", comment="#")
            df0[f, "t"] = sr.iloc[:, 0]
            df0[f, "P"] = sr.iloc[:, 1]

            popt = self.pulse(sr.iloc[:, 0].values, sr.iloc[:, 1].values)

            df.iloc[i][2:] = popt
            df.iloc[i][0] = self.swh(df.iloc[i]["Sigma"])
            df.iloc[i][1] = self.height(df.iloc[i]["Epoch"])

        excel_name = "output.xlsx"

        df.to_excel(excel_name, sheet_name='brown')

        with pd.ExcelWriter(excel_name, mode='a') as writer:  
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
    def ice(t, A,alpha,tau,sigma_l,T):
        """
        Точная аппроксимация формулы Брауна.
        В отличии от Брауна не привязяна к абсолютному времени. 
        См. отчет по Ростову за 2020 год

        """
        return A * np.exp( -alpha * (t-tau) ) * (1 + erf( (t-tau)/sigma_l ) ) + T

    def pulse(self, t, pulse, func=None):
        alpha = self.leading_edge(t, pulse, dtype="needed")
        A, tau, sigma_l, b = self.trailing_edge(t, pulse)

        popt = curve_fit(self.ice, 
                            xdata=t,
                            ydata=pulse,
                            p0=[A, alpha, tau, sigma_l, b],
                            # bounds = [0, np.inf]
                        )[0]
        return popt 

    # def karaev(self, t, pulse):
    #     h = rc.antenna.z
    #     A = lambda var: 1/(2*var*h**2) + 5.52/(rc.antenna.beamWidth)
    #     F1 = 


    
    def swh(self, sigma_l):

        """
        Вычисление высоты значительного волнения
        """
        # Скорость света/звука [м/с]
        c = rc.constants.lightSpeed
        # Длительность импульса [с]
        T = rc.antenna.impulseDuration

        sigma_p = 0.425 * T
        sigma_c = sigma_l/np.sqrt(2)
        sigma_s = np.sqrt((sigma_c**2 - sigma_p**2))*c/2
        return 4*sigma_s

    def height(self, tau):
        """
        Вычисление высоты от антенны до поверхности воды
        """

        # Скорость света/звука [м/с]
        c = rc.constants.lightSpeed
        return tau*c/2

    def emb(self, swh, U10, dtype = "Rostov"):
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
    

