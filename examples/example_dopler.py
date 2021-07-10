import sys

sys.path.append(".")
from seawave.spectrum import dopler
from seawave import config

import matplotlib.pyplot as plt

import numpy as np 
import pandas as pd


# Ширина ДН, градусы
config['Radar']['GainWidth'] = [15, 15]



omega = np.linspace(-100, 100, 1000)*np.pi

print('Смещение, Ширина: ', dopler.shift, dopler.width)

S = dopler(omega)
plt.plot(omega, S)
plt.savefig("dopler_spectrum.png")


f = omega/2/np.pi
# Экспорт в файл.
df = pd.DataFrame({"f": f, "S": S})
df.to_excel("dopler_spectrum.xlsx")


