import sys

sys.path.append(".")
import numba
from numba import cuda
import numpy as np

from seawave import rc
from seawave.surface import surface
import matplotlib.pyplot as plt 


rc.surface.gridSize = [128, 128]
rc.surface.kSize = 1024
rc.surface.phiSize = 128

x = np.linspace(*rc.surface.x, rc.surface.gridSize[0])
y = np.linspace(*rc.surface.y, rc.surface.gridSize[1])
t = np.arange(0, .5, 0.5, dtype=np.float32)
k, A = surface.export()
from cmath import exp


srf = np.zeros((x.size, y.size, t.size) , dtype=np.complex128)

import time
start = time.time()
default_test(srf, x, y, t, k, A)
end = time.time()
print(end-start)

from seawave.surface.module import init
from seawave.surface.dataset import float_surface

start = time.time()
srf = float_surface(x,y,t)
srf = init(srf, (k,A))
end = time.time()

print(end-start)
start = time.time()
srf = float_surface(x,y,t)
srf = init(srf, (k,A))
end = time.time()
print(end-start)






