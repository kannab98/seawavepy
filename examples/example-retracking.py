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
from seawave.retracking import spectrum as specrtr
from seawave import config


config["Radar"]["GainWidth"] = 15
config["Radar"]["ImpulseDuration"] = 60e-6

# files = rtr.get_files("tests/impulses/.*.txt")

# pulses = []
# for file in files:
# 	pulses.append(rtr.karaev(file=file, config=config))
# 	pulses.append(rtr.brown(file=file, config=config))
# rtr.to_xlsx(pulses)

files = rtr.get_files("tests/impulses/.*Spectrum.*.txt")
specrtr.to_xlsx(files)
print(files)
