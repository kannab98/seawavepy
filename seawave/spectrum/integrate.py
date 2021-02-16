# from . import spectrum
from .module import dispersion
from scipy import integrate

def quad(self, a, b, k0=None, k1=None, **quadkwargs):
    self = args[0]
    if k0==None:
        k0 = self.KT[0]

    if k1==None:
        k1 = self.KT[-1]

    S = lambda k: self.__call__(k) * k**a * dispersion.omega(k)**b
    var = integrate.quad(S, k0, k1, **quadkwargs)[0]
    return var


def dblquad(a, b, c, k0=None, k1=None, phi0=None, phi1=None, **quadkwargs):
    self = args[0]
    limit = np.array([self.KT[0], *self.limit_k, self.KT[-1]])

    if k0==None:
        k0 = self.KT[0]

    if k1==None:
        k1 = self.KT[-1]

    if phi0==None:
        phi0 = -np.pi
    
    if phi1==None:
        phi1 = np.pi
    

    S = lambda phi, k:  self.__call__(k, phi) *  k**(a+b-c) * np.cos(phi)**a * np.sin(phi)**b
    var = integrate.dblquad( S,
            a=k0, b=k1,
            gfun=lambda phi: phi0, 
            hfun=lambda phi: phi1, **quadkwargs)
    
    return var[0]