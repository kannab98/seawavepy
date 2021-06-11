from . import config
import numpy as np
# from .spectrum import spectrum


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
                waveLength = config['Radar']['WaveLength']

                if self._x != x or self._U != U or self.peak == None or \
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