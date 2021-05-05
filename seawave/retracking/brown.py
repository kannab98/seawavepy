from .. import config
import numpy as np
from ..spectrum import spectrum
from scipy.special import erf

class brown():

    def __init__(self, config):

        theta = np.deg2rad(config['Radar']['GainWidth'])
        self.Gamma = self.gamma(theta)

    def t(self):
        T = config['Radar']['ImpulseDuration']
        return np.linspace(-10*T, 100*T, 1000)

    @staticmethod
    def H(h):
        R = config['Constants']['EarthRadius']
        # return h * ( 1 + h/R )
        return h
    
    @staticmethod
    def A(gamma, A0=1.):
        xi = np.deg2rad(config['Radar']['Direction'][1])
        return A0*np.exp(-4/gamma * np.sin(xi)**2 )

    @staticmethod
    def u(t, alpha, sigma_c, cwm_mean = 0, cwm_var = 0):
        c = config['Constants']['WaveSpeed']
        return (t - alpha * sigma_c**2 - cwm_mean/c) / (np.sqrt(2) * sigma_c)

    @staticmethod
    def v(t, alpha, sigma_c, cwm_mean = 0, cwm_var = 0):
        c = config['Constants']['WaveSpeed']
        return alpha * (t - alpha/2 * sigma_c**2 - cwm_mean/c)

    @staticmethod
    def alpha(beta,delta):
        xi = np.deg2rad(config['Radar']['Direction'][1])
        c = config['Constants']['WaveSpeed']
        return delta - beta**2/4

    def delta(self, gamma):
        h = np.abs(config['Radar']['Position'][2])

        c = config['Constants']['WaveSpeed']
        xi = np.deg2rad(config['Radar']['Direction'][1])
        return 4/gamma * c/self.H(h) * np.cos(2 * xi)
    
    @staticmethod
    def gamma(theta):
        return 2*np.sin(theta/2)**2/np.log(2)

    def beta(self, gamma):
        c = config['Constants']['WaveSpeed']
        xi = np.deg2rad(config['Radar']['Direction'][1])
        h = np.abs(config['Radar']['Position'][2])
        return 4/gamma * np.sqrt( c/self.H(h) ) * np.sin( 2*xi )

    @staticmethod
    def sigma_c(sigma_s):
        c = config['Constants']['WaveSpeed']
        T = config['Radar']['ImpulseDuration']
        sigma_p = 0.425 * T 
        return np.sqrt(sigma_p**2 + (2*sigma_s/c)**2 )

    def pulse(self, t, dim = 1, cwm=False):

        self.dim = dim
        print(config["Radar"]["GainWidth"])
        gamma = self.Gamma
        delta = self.delta(gamma)
        beta  = self.beta(gamma)

        if dim == 1:
            alpha = self.alpha(beta, delta)
        else:
            alpha = self.alpha(beta/np.sqrt(2), delta)


        spec = spectrum
        sigma_s = np.sqrt(spec.quad(0, 0))
        sigma_c = self.sigma_c(sigma_s)

        cwm_mean = 0

        if cwm == True:

            s = np.sign(config["Radar"]["Position"][2])
            cwm_mean = s*spec.quad(1, 0)
            sigma_s = spec.quad(0, 0) - cwm_mean**2
            sigma_c = self.sigma_c(sigma_s)


        u = self.u(t, alpha, sigma_c, cwm_mean=cwm_mean)
        v = self.v(t, alpha, sigma_c, cwm_mean=cwm_mean)

        A = self.A(gamma)
        pulse = A * np.exp(-v) * ( 1 + erf(u) )

        if self.dim == 2:
            alpha = gamma
            u = self.u(t, alpha, sigma_c)
            v = self.v(t, alpha, sigma_c)
            pulse -= A/2 * np.exp(-v) * ( 1 + erf(u) )

        return pulse

