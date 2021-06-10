from . import rc, config
import numpy as np
# from .spectrum import spectrum

def surface_dispatcher(func):
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

def spectrum_dispatcher():
    def decorator(func):
        """
        Декоратор обновляет необходимые переменные при изменении
        разгона или скорости ветра
        """
        def wrapper(*args, dispatcher=True, radar_dispatcher=True, **kwargs):
            # self = spectrum
            self = args[0]
            if dispatcher:
                x = config['Surface']['NonDimWindFetch']
                U = config['Wind']['Speed']
                band = config['Surface']['Band']
                waveLength = config['Radar']['WaveLength']

                if self._x != x or self._U != U or self._band != band or self.peak == None or \
                (self._wavelength != waveLength and radar_dispatcher):
                    self.peakUpdate(radar_dispatcher)

                self._x, self._U = x, U
                self._wavelength = config["Radar"]["WaveLength"]



            return func(*args, **kwargs)
        return wrapper
    return decorator

def ufunc(nin, nout):
    def decorator(func):
        def wrapper(*args, **kwargs):
            ufunc = np.frompyfunc(func, nin, nout)
            return ufunc(*args, **kwargs)
        return wrapper
    return decorator