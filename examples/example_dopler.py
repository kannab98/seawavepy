import sys

sys.path.append(".")
from seawave.spectrum import dopler, spectrum
from seawave import config

import matplotlib.pyplot as plt

import numpy as np 
import pandas as pd


# Ширина ДН, градусы
config['Radar']['GainWidth'] = [15, 15]

phi = np.linspace(-90, 90, 100) 

Sw = np.zeros(phi.size)

for i, az in enumerate(phi):
	config["Wind"]["Direction"] = az 
	Sw[i] = dopler.width


plt.figure()
plt.plot(phi, Sw)
plt.savefig("kek")
print(Sw)

df = pd.DataFrame({"phi": phi, "S_width": Sw})
df.to_excel("dopler_spectrum.xlsx")


