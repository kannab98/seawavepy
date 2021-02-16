

class brown():

    def __init__(self):

        theta = np.deg2rad(rc.antenna.gainWidth)
        self.Gamma = self.gamma(theta)

    def t(self):
        T = rc.antenna.impulseDuration
        return np.linspace(-10*T, 25*T, 1000)

    @staticmethod
    def H(h):
        R = rc.constants.earthRadius
        return h * ( 1 + h/R )
    
    @staticmethod
    def A(gamma, A0=1.):
        xi = np.deg2rad(rc.antenna.deviation)
        return A0*np.exp(-4/gamma * np.sin(xi)**2 )

    @staticmethod
    def u(t, alpha, sigma_c, cwm_mean = 0, cwm_var = 0):
        c = rc.constants.lightSpeed
        return (t - alpha * sigma_c**2 - cwm_mean/c) / (np.sqrt(2) * sigma_c)

    @staticmethod
    def v(t, alpha, sigma_c, cwm_mean = 0, cwm_var = 0):
        c = rc.constants.lightSpeed
        return alpha * (t - alpha/2 * sigma_c**2 - cwm_mean/c)

    @staticmethod
    def alpha(beta,delta):
        return delta - beta**2/4

    def delta(self, gamma):
        c = rc.constants.lightSpeed
        xi = np.deg2rad(rc.antenna.deviation)
        h = rc.antenna.z
        return 4/gamma * c/self.H(h) * np.cos(2 * xi)
    
    @staticmethod
    def gamma(theta):
        return 2*np.sin(theta/2)**2/np.log(2)

    def beta(self, gamma):
        c = rc.constants.lightSpeed
        xi = np.deg2rad(rc.antenna.deviation)
        h = rc.antenna.z
        return 4/gamma * np.sqrt( c/self.H(h) ) * np.sin( 2*xi )

    @staticmethod
    def sigma_c(sigma_s):
        T = rc.antenna.impulseDuration
        c = rc.constants.lightSpeed
        sigma_p = 0.425 * T 
        return np.sqrt(sigma_p**2 + (2*sigma_s/c)**2 )

    def pulse(self, t, dim = 1, cwm=False):

        self.dim = dim
        gamma = self.Gamma
        delta = self.delta(gamma)
        beta  = self.beta(gamma)

        if dim == 1:
            alpha = self.alpha(beta, delta)
        else:
            alpha = self.alpha(beta/np.sqrt(2), delta)


        spec = spectrum
        surf = surface
        sigma_s = spec.quad(0, 0)
        sigma_c = self.sigma_c(sigma_s)

        cwm_mean = 0

        if cwm == True:

            cwm_mean = spec.quad(1)
            sigma_s = spec.quad(0) - cwm_mean
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

