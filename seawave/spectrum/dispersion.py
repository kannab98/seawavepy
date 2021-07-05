from .. import config
import numpy as np
from scipy.optimize import fsolve

import logging
logger = logging.getLogger(__name__)

g = config['Constants']['GravityAcceleration']
T = 74e-3  # Newton per meter -- surface tension coefficient
rho = 1e3  # kg/m^3 -- density of water
a = T/rho


__bibtex__ = {
    "label" : "lopatuhin-and-rozhkov",
    "title" : "Ветровое волнение в Мировом океане",
    "author" : "Лопатухин, Л.И. and Рожков В.А.",
    "year" : "1985",
    "publisher" : "Л.: Гидрометеоиздат",
    "pages": "256"
}


def f(k, H):
    # omega2
    return (g*k + a*k**3)*np.tanh(k*H)

def df(k, H):
    # domega2/dk
    # https://www.wolframalpha.com/input/?i=+d%2Fdk+%28g*k+%2B+a*k%5E3%29*tanh%28H*k%29
    if np.isinf(H):
        return g + 3*a*k**2
    else:
        return (
            (g + 3*a*k**2) * np.tanh(k*H) + H*k * (a*k**2 + g) * 1/np.sinh(k*H)**2
        )

def solver(omega, omega0, H): 
    return f(omega, H) - omega0**2

def omega(k):
    """
    Решение прямой задачи поиска частоты по известному волновому числу 
    из дисперсионного соотношения
    """
    k = np.abs(k)
    H = config['Surface']['CoastHeight']
    return np.sqrt(f(k, H))

def k(omega):
    """
    Решение обратной задачи поиска волнового числа по известной частоте
    из дисперсионного соотношения

    """

    H = config['Surface']['CoastHeight']
    sol = fsolve(solver, x0=2000, args=(omega, H))
    return sol

def det(k):
    """
    Функция возвращает Якобиан при переходе от частоты к
    волновым числам по полному дисперсионному уравнению
    """
    H = config['Surface']['CoastHeight']
    return df(k, H) / (2*omega(k))
