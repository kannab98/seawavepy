from .. import config 
import numpy as np 

g = config['Constants']['GravityAcceleration']
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