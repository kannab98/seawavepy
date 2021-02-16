import numpy as np
from . import surface

from . import rc
import matplotlib.pyplot as plt


class experiment():

    def dispatcher(func):
        """
        Декоратор обновляет необходимые переменные при изменении
        разгона или скорости ветра
        """
        def wrapper(*args):
            self = args[0]
            x = rc.surface.x
            y = rc.surface.y
            gridSize = rc.surface.gridSize
            N = rc.surface.kSize
            M = rc.surface.phiSize

            if self._n.all() != surface.normal.all() or \
               self._r.all() != surface.coordinates.all():
                self.geometryUpdate()
            


            return func(*args)

        return wrapper

    def geometryUpdate(self):
        self._r = surface.coordinates
        self._R = self.position(surface.coordinates, self.sattelite_coordinates)

        offset = rc.antenna.z*np.arctan(np.deg2rad(rc.antenna.deviation))
        R0 = np.array([offset, 0, 0])[None].T*np.ones((1,self._R.shape[-1]))
        # print(self._R[:,0])
        self._R += R0
        # print(self._R[:,0])
        self._n = surface.normal
        self._nabs = self.abs(self._n)
        self._Rabs = self.abs(self._R)




    def __init__(self):

        self._r = None
        self._R = None
        self._n = None
        self._nabs = None
        self._Rabs = None

        self.geometryUpdate()

        self._gamma = 2*np.sin(rc.antenna.gainWidth/2)**2/np.log(2)
        # self._G = self.G(self._R, 
        #                     rc.antenna.polarAngle, 
        #                     rc.antenna.deviation, self._gamma)


        # self.tau = self._Rabs / rc.constants.lightSpeed
        # self.t0 = self._Rabs.min() / rc.constants.lightSpeed



    @staticmethod
    def abs(vec):
        vec = np.array(vec, dtype=float)
        return np.sqrt(np.sum(vec**2,axis=0))
        # return np.sqrt(np.diag(vec.T@vec))

    @staticmethod
    def position(r, r0):
        r0 = r0[np.newaxis]
        return np.array(r + ( np.ones((r.shape[1], 1) ) @ r0 ).T)


    
    @staticmethod
    def G(r, phi, xi, gamma):
        x = r[0]
        y = r[1]
        z = r[2]
        # Поворот системы координат на угол phi
        X = np.array(x * np.cos(phi) + y * np.sin(phi), dtype=float)
        Y = np.array(x * np.sin(phi) - y * np.cos(phi), dtype=float)
        Z = np.array(z, dtype=float)

        # Проекция вектора (X,Y,Z) на плоскость XY
        rho =  np.sqrt(np.power(X, 2) + np.power(Y, 2))
        rho[np.where(rho == 0)] += 1.49e-8

        # Полярный угол в плоскости XY
        psi = np.arccos(X/rho)


        if not isinstance(xi, np.ndarray):
            if isinstance(xi, list):
                xi = np.array(xi)
            else:
                xi = np.array([xi])

        xi = xi[np.newaxis]


        theta = (Z*np.cos(xi.T) + rho*np.sin(xi.T)*np.cos(psi))
        theta *= 1/np.sqrt(X**2 + Y**2 + Z**2)
        theta = np.arccos(theta)

        # Возвращает массив размерности (xi, r)
        return np.exp(-2/gamma * np.sin(theta)**2)


    @property
    def sattelite_coordinates(self):
        return np.array([rc.antenna.x, rc.antenna.y, rc.antenna.z])

    @sattelite_coordinates.setter
    def sattelite_coordinates(self, r):
        rc.antenna.x, rc.antenna.y, rc.antenna.z = r
        self._R = self.position(self.surface_coordinates, self.sattelite_coordinates)
        self._Rabs = self.abs(self._R)



    @property
    def incidence(self):
        z = np.array(self._R[-1], dtype=float)
        R = np.array(self._Rabs, dtype=float)
        return np.arccos(z/R)

    def localIncidence(self, xi=0):
        rc.antenna.deviation = xi
        self.geometryUpdate()
        R = np.array(self._R/self._Rabs, dtype=float)
        n = np.array(self._n/self._nabs, dtype=float)




        # print(R.shape)


        # Матрица поворта вокруг X
        # Mx = self.rotatex(np.pi/2)
        My = self.rotatey(np.pi/2)
        # taux = np.einsum('ij, jk -> ik', My, n)
        # tauy = np.einsum('ij, jk -> ik', Mx, n)

        theta0n = np.einsum('ij, ij -> j', R, n)
        # theta0tau = np.einsum('ij, ij -> j', R, taux)
        # theta0tauy = np.einsum('ij, ij -> j', R, tauy)


        # return np.sign(theta0tau)*np.arccos(theta0n)
        return np.arccos(theta0n)

    @staticmethod
    def rotatex(alpha):
        Mx = np.array([
            [1,             0,             0 ],
            [0, np.cos(alpha), -np.sin(alpha)],
            [0, np.sin(alpha), +np.cos(alpha)]
        ])
        return Mx

    @staticmethod
    def rotatey(alpha):
        My = np.array([
            [+np.cos(alpha), 0, np.sin(alpha)],
            [0,              1,            0 ],
            [-np.sin(alpha), 0, np.cos(alpha)]
        ])
        return My

    @staticmethod
    def sort(incidence, xi=0, err = 0.7):

        # if not isinstance(xi, np.ndarray):
        #     if isinstance(xi, list):
        #         xi = np.array(xi)
        #     else:
        #         xi = np.array([xi])

        # xi = xi[np.newaxis]

        # theta0 = np.abs(incidence - xi.T)
        
        # index = np.abs(theta0) < np.deg2rad(err)
        index = np.where(incidence < np.deg2rad(err))
        return index

    
    def gain(self, theta, xi):
        self._G = self.G(self._R, 
                            theta,
                            xi, self._gamma)
        return self._G

    
    # @property
    # def x(self):
    #     size = self.surface.gridSize
    #     return self._R[0].reshape((size, size))

    # @property
    # def y(self):
    #     size = self.surface.gridSize
    #     return self._R[1].reshape((size, size))
    # @property
    # def z(self):
    #     size = self.surface.gridSize
    #     return self._R[2].reshape((size, size))
    
    def power(self, t):
        tau = self._Rabs / self.constants.c
        timp = self.sattelite.impulseDuration
        index = [ i for i in range(tau.size) if 0 <= t - tau[i] <= timp ]

        theta =self.incidence[index]
        Rabs = self._Rabs[index]
        R = self._R[:,index]
        theta0 = self.localIncidence[index]

        index = self.sort(theta0)
        theta = theta[index]
        Rabs = Rabs[index]
        R = R[:,index]


        G = self.gain

        E0 = (G/Rabs)**2
        P = np.sum(E0**2/2)
        return P

    def crossSection(self, theta):
        # Эти штуки не зависят от ДН антенны, только от геометрии
        # ind = self.sort(self.localIncidence)
        # Rabs = self._Rabs[ind]
        # R = R[:,index]
        self.geometryUpdate()
        # gamma = 2*np.sin(np.deg2rad(15)/2)**2/np.log(2)

        # G = self.G(self._R, np.pi/2, theta, gamma)

        N = np.zeros_like(theta, dtype=float)
        for i, xi in enumerate(theta):
            theta0 = self.localIncidence(xi=xi)

            ind = self.sort(theta0, xi=xi)
            # N[i] = np.sum(G[i][ind])
            N[i] = ind[0].size


        # # Эти, разумеется, зависят


        # E0 = G**2/R**2*cos


        return N

    

    





# kernels = [kernel_default]
# labels = ["default", "cwm"] 

# surface = Surface()

# ex = Experiment(surface)
# z = ex.sattelite_coordinates[-1]

# U = surface.windSpeed
# g = ex.constants.g



# surface.spectrum.nonDimWindFetch = 20170
# surface.nonDimWindFetch = 20170
# xi = np.arctan(5000/z)
# Xi = np.deg2rad(np.linspace(-17, 17, 49))

# # rc = surface._rc
# # z = rc["antenna"]["z"]
# # R = rc["constants"]["earthRadius"]



# plt.plot(np.rad2deg(Xi), np.rad2deg(f(Xi)))
# plt.show()
# sigma0 = np.zeros(Xi.size)

# fig, ax = plt.subplots()
# X = U**2 * 20170 / g 

# for i, xi in enumerate(Xi):
#     arr, X0, Y0 = srf.run_kernels(kernels, surface)
#     sigma0[i] = surface.crossSection(xi)


# sigma0 = surface.crossSection(Xi)
# plt.plot(Xi, sigma0/sigma0.max())

# kernels = [srf.kernel_default]





# wind = np.linspace(3,15, Xsize)
# fetch = np.linspace(5000, 20170, Xsize)
# direction = np.linspace(-np.pi/2, np.pi/2, 180)
# Xsize = direction.size
# sigma0 = np.zeros((Xi.size, Xsize))

# Xb, Yb, Zb = ex.surface_coordinates
# for i, xi in enumerate(Xi):
#     X = Xb
#     Y = Yb + z*np.tan(xi)
#     for j in range(Xsize):
#         X, Y, Z = ex.surface_coordinates
#         # surface.nonDimWindFetch= fetch[j]
#         # surface.windSpeed = wind[j]
#         surface.direction[0] = direction[j]
#         ex.surface_coordinates = (X,Y,Z)
#         arr, X0, Y0 = srf.run_kernels(kernels, surface)
#         sigma0[i][j] = surface.crossSection(xi)
#         X += 5000




# y = np.array([z*np.tan(xi) for xi in Xi])
# x = np.array([5000*i for i in range(Xsize)])
# x, y = np.meshgrid(x,y)

# import pandas as pd
# df = pd.DataFrame({'x': x.flatten(), 'y': y.flatten(), 'sigma': sigma0.flatten()})

# df.to_csv('direction2.tsv' , sep='\t', float_format='%.2f')

# plt.contourf(sigma0, levels=100)
# # plt.savefig("track_fetch")

# plt.savefig("direction2" )

# X1, Y1 = np.meshgrid(X1, Y1)
# plt.contourf(X1, Y1, sigma0.T, levels=100)

# plt.imshow(sigma0)

# sigma0 = surface.crossSection(Xi)
# plt.plot(Xi, sigma0.max())

# plt.savefig("sigma0")
    # for i in range(len(kernels)):
    #     surface.plot(X0[i], Y0[i], arr[i][0], label="default%s" % (U))


# plt.figure()
# for i in range(t.size):
    # p[i] = ex.power(t=t[i])

# plt.plot(t,p)
# plt.show()

# rc.pickle()
# pulse = Pulse(rc)

# rc = pulse.rc
# rc.surface.gridSize = 252
# print(rc.surface.gridSize) 


# import matplotlib.pyplot as plt


# G = np.zeros(pulse.gain.shape)
# pulse.polarAngle = 90
# x = pulse.x
# y = pulse.y



# for i in range(8):
#     for xi in np.arange(-10,10,3):
#         r0 = pulse.sattelite_position
#         pulse.sattelite_position = np.array([r0[0]+5000,r0[1], r0[2]])
#         pulse.deviation = xi 
#         G  += pulse.gain

# plt.contourf(x,y,G)
# plt.savefig("kek")
    

    

    
