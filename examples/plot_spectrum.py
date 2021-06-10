import matplotlib.pyplot as plt 
import sys
sys.path.append(".")
from seawave import config
from seawave.spectrum import spectrum

config["Wind"]["Speed"] = 12
config["Surface"]["NonDimWindFetch"] = 20170
config["Radar"]["WaveLength"] = [0.008]

plt.figure()

# spectrum.peakUpdate()
k = spectrum.k 
S = spectrum(k)

plt.loglog(k, S)
plt.savefig("spectrum.png")
