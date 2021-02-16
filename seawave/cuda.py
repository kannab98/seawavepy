from numba import cuda, float32
import math
from cmath import exp, phase

from . import rc
g = rc.constants.gravityAcceleration

__bibtex = {
    "label" : "shuleykin",
    "title" : "Физика моря",
    "author" : "В.В. Шулейкин",
    "year" : "1968",
    "publisher" : "М. Наука"
}

@cuda.jit(device=True)
def dispersion(k):
    k = abs(k)
    return math.sqrt(g*k + 74e-6*k**3)

@cuda.jit(device=True)
def identity(args: tuple):
    for i in range(len(args)):
        for n in range(args[i].shape[0]):
            for m in range(args[i].shape[1]):
                if n == m:
                    args[i][n, m] = 1
                else:
                    args[i][n, m] = 1
    return args

@cuda.jit(device=True)
def dot(A, B, C):
    """Perform square matrix multiplication of C = A * B
    """

    # if A.shape[1] != B.shape[0]:
    #     print(A.shape[1], B.shape[0])
    #     return 0
    

    # print(C.shape[0], C.shape[1])
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            C[i,j] = 0
            for k in range(A.shape[1]):
                C[i,j] += A[i,k] * B[k,j]

    return C



@cuda.jit(device=True)
def inverse(M, I):

    # forward trace
    n = M.shape[0]

    for k in range(n):
        # 1) Swap k-row with one of the underlying if m[k, k] = 0
        # 2) Make diagonal element equals to 1
        if M[k, k] != 1:
            tmp =  M[k, k]
            for j in range(n):
                I[k, j] *= 1 / tmp
                M[k, j] *= 1 / tmp
        # 3) Make all underlying elements in column equal to zero
        for row in range(k + 1, n):
            tmp = M[row, k]

            for j in range(n):
                I[row, j] -= I[k, j] * tmp
            for j in range(n):
                M[row, j] -= M[k, j] * tmp


    for k in range(n - 1, 0, -1):
        for row in range(k - 1, -1, -1):
            if M[row, k]:
                # 1) Make all overlying elements equal to zero in the former identity matrix
                tmp = M[row, k]
                for j in range(n):
                    I[row, j] -= I[k, j] * M[row, k]
                for j in range(n):
                    M[row, j] -= M[k, j] * M[row, k]

    return I

@cuda.jit(device=True)
def newton(x0, y0, k, A, ):
    epsabs = 1.49e-2
    limit = 5
    X0 = x0
    Y0 = y0

    jac = cuda.local.array((2, 2), float32)
    I = cuda.local.array((2, 2), float32)
    F = cuda.local.array((1, 2), float32)
    D = cuda.local.array((1, 2), float32)

    for iterable in range(limit):

        jac, I = identity((jac, I))

        Fx = x0
        Fy = y0

        Fx, Fy, jac = cwm_grid(Fx, Fy, jac, k, A)
        jacinv = inverse(jac, I)

        x = x0
        y = y0

        F[0] = Fx - X0
        F[1] = Fy - Y0

        D = dot(F, jacinv, D)

        x0 -= D[0, 0]
        y0 -= D[0, 1]

        if abs(x - x0) < epsabs:
            break

    return x0, y0

@cuda.jit(device=True)
def cwm_grid(x: float, y: float, jac, k, A):

    for n in range(k.shape[0]): 
        for m in range(k.shape[1]):
            kr = k[n,m].real*x + k[n,m].imag*y
            e = A[n,m] * exp(1j*kr) 

            jac[0, 0] -=  e.real * (k[n,m].real * k[n,m].real)/abs(k[n,m])
            jac[0, 1] -=  e.real * (k[n,m].real * k[n,m].imag)/abs(k[n,m])
            jac[1, 1] -=  e.real * (k[n,m].imag * k[n,m].imag)/abs(k[n,m])

            x -= e.imag * k[n,m].real/abs(k[n,m])
            y -= e.imag * k[n,m].imag/abs(k[n,m])
    
    jac[1, 0] = jac[0, 1]

    return x, y, jac


@cuda.jit
def linearized_cwm_grid(x0, y0, k, A):

    i = cuda.grid(1)
    if i >= x0.shape[0]:
        return
    
    x0[i], y0[i] = newton(x0[i], y0[i], k, A)


@cuda.jit(device=True)
def base(surface, x, y, k, A, method):
    for n in range(k.shape[0]): 
        for m in range(k.shape[1]):
                kr = k[n,m].real*x + k[n,m].imag*y
                w = dispersion(k[n,m])
                e = A[n,m] * exp(1j*kr)  # * exp(1j*w*t)

                # Высоты (z)
                surface[0] +=  +e.real
                # Наклоны X (dz/dx)
                surface[1] +=  -e.imag * k[n,m].real
                # Наклоны Y (dz/dy)
                surface[2] +=  -e.imag * k[n,m].imag
                # Орбитальные скорости Vz (dz/dt)
                surface[3] +=  -e.imag * w

                # Vh -- скорость частицы вдоль направления распространения ветра.
                # см. [shuleykin], гл. 3, пар. 5 Энергия волн.
                # Из ЗСЭ V_h^2 + V_z^2 = const

                # Орбитальные скорости Vx
                surface[4] += e.real * w * k[n,m].real/abs(k[n,m])
                # Орбитальные скорости Vy
                surface[5] += e.real * w * k[n,m].imag/abs(k[n,m])





                # # Поправка на наклоны заостренной поверхности
                # if method == "cwm":
                    # Наклоны X dz/dx * dx/dx0
                    # surface[1] *= 1 - e.real * (k[n,m].real * k[n,m].real)/abs(k[n,m])
                    # Наклоны Y dz/dy * dy/dy0 
                    # surface[2] *= 1 - e.real * (k[n,m].imag * k[n,m].imag)/abs(k[n,m])
                #     # Орбитальные скорости Vh dVh/dx * dx/dx0
                #     surface[4] *= 1 - e.real * (k[n,m].real * k[n,m].real)/abs(k[n,m])

    return surface

@cuda.jit
def cwm(ans, x, y, k, A):
    i = cuda.grid(1)

    if i >= x.shape[0]:
        return

    surface = cuda.local.array(6, float32)
    surface = base(surface, x[i], y[i], k, A, 'cwm')
    for j in range(6):
        ans[j, i] = surface[j]


@cuda.jit
def default(ans, x, y, k, A):
    i = cuda.grid(1)

    if i >= x.shape[0]:
        return

    surface = cuda.local.array(6, float32)
    surface = base(surface, x[i], y[i], k, A, 'default')
    for j in range(6):
        ans[j, i] = surface[j]


@cuda.jit
def check_cwm_grid(x, y, k, A):
    i = cuda.grid(1)

    if i >= x.shape[0]:
        return

    for n in range(k.shape[0]): 
        for m in range(k.shape[1]):
            kr = k[n,m].real*x[i] + k[n,m].imag*y[i]
            e = A[n,m] * exp(1j*kr) 
            x[i] -= e.imag * k[n,m].real/abs(k[n,m])
            y[i] -= e.imag * k[n,m].imag/abs(k[n,m])

