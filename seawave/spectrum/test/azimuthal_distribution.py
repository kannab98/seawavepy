from .. import spectrum
from ... import rc
import matplotlib.pyplot as plt
import numpy as np

def main():
    phi = np.linspace(-np.pi, np.pi, 256)
    spectrum(0)
    F = spectrum.azimuthal_distribution(spectrum.peak, phi)

    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    ax.plot(phi, F.T)
    return fig, ax

