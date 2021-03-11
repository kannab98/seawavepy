import sys
sys.path.append(".")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from seawave.spectrum import spectrum
from seawave import rc

waveLength = [0.20,  0.1,  0.05, 0.022]
rc.antenna.waveLength = waveLength

U10 = np.linspace(3, 15, 2)
kb = np.zeros((U10.size, len(waveLength), ))

for i in range(U10.size):
    rc.wind.speed = U10[i]
    kb[i] = spectrum.kEdges(waveLength,radar_dispatcher=False)[1:]

plt.figure()
plt.plot(U10, kb)


