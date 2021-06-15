import numpy as np
from numpy import random
from scipy.special import erf
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.style import use
use(["science", "ieee", "vibrant"])


import sys
sys.path.append(".")
from seawave.retracking import retracking_new as rtr
from seawave import config


config["Radar"]["GainWidth"] = 15
config["Radar"]["ImpulseDuration"] = 60e-6

files = rtr.get_files("tests/impulses/.*.txt")

pulses = []
for file in files:
	pulses.append(rtr.karaev(file=file, config=config))
rtr.to_xlsx(pulses)

pulses = []
for file in files:
	pulses.append(rtr.brown(file=file, config=config))
rtr.to_xlsx(pulses)


# # popt_ice = pb.popt

# plt.figure()
# t = pk.time

# popt = np.array(pk.popt, copy=True)
# # popt[0] *= 0.9
# # popt[2] += 0.02
# # varslopes = 0.6*pk.varslopes
# # popt[1] = pk.slopes_coeff(varslopes, pk.height, pk.delta)


# plt.figure()
# plt.plot(pk.time, pk.power, label="experiment", linestyle="-")
# plt.plot(pk.time, pk.pulse(pk.time, *pk.popt), label="karaev", linestyle="--")
# # plt.plot(pk.time, pk.pulse(pk.time, *popt), label="karaev manual", linestyle="-")
# plt.legend()
# plt.xlim([0.0325, 0.038])

# plt.savefig("lel")

# plt.figure()
# P = pk.power - pk.popt[-1]

# popt = np.array(pk.popt, copy=True)
# P0 = pk.pulse(pk.time, *pk.popt) - pk.popt[-1]
# plt.plot(pk.time, P/P.max(), label="experiment", linestyle="-")
# plt.plot(pk.time, P0/P0.max(), label="karaev", linestyle="--")
# # plt.plot(pk.time, pk.pulse(pk.time, *popt), label="karaev manual", linestyle="-")
# plt.legend()
# plt.xlim([0.0325, 0.038])
# plt.savefig("lel1")

# plt.figure()



