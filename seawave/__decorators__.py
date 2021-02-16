from . import rc
import numpy as np
# from . import spectrum

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
                x = rc.surface.nonDimWindFetch
                U = rc.wind.speed
                band = rc.surface.band
                waveLength = rc.antenna.waveLength

                if self._x != x or self._U != U or self._band != band or self.peak == None or \
                (self._wavelength != waveLength and radar_dispatcher):
                    self.peakUpdate(radar_dispatcher)

                self._x, self._U = x, U
                self._wavelength = rc.antenna.waveLength

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