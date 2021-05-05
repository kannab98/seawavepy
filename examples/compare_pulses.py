import sys
sys.path.append(".")
import numpy as np
import matplotlib.pyplot as plt

from seawave import config
from seawave.retracking import brown, retracking
from seawave.spectrum import spectrum




for H in [30, 100]:
    config["Wind"]["Speed"] = 10
    config["Radar"]["Position"][2] = H
    config["Radar"]["ImpulseDuration"] = 60e-6
    config["Radar"]["GainWidth"] = 15
    config["Constants"]["WaveSpeed"] = 1500

    varslopesx = spectrum.quad(2,0)
    varslopesy = 0.018

    delta = np.deg2rad(15)

    slopes_coeffx = 1/(2*varslopesx*H**2) + 5.52/(delta**2*H**2)


    varelev = (1.36/4)**2

    t0 = 0
    # t += t0
    t = np.linspace(-3e-3, 1e-2, 256)
    F = retracking.full_pulse(t, slopes_coeff=slopes_coeffx, sigma0=1, t0=t0, varelev=varelev, H=H, t_pulse=60e-6, c=1500)
    plt.plot(t, F/F.max())
    b = brown.brown(config)
    P = b.pulse(t)
    plt.plot(t, P/P.max())


plt.savefig("kek")
