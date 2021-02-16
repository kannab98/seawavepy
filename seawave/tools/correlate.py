from .. import rc, surface, spectrum, kernel 
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import scipy

def test():
    x = np.linspace(0, 4*np.pi, 256)
    y = np.cos(x)

    K0 = scipy.signal.fftconvolve(y, y, mode="full")

    Y = scipy.fft.fft(y)
    X = scipy.fft.fftfreq(x.size, x[1] - x[0])

    X = scipy.fft.fftshift(X)
    Y = scipy.fft.fftshift(Y)


    fig, ax = plt.subplots(nrows=1, ncols=2)

    ax[0].plot(K0)
    ax[1].plot(X,Y)

    plt.show()

def correlate(x, y):
    K = scipy.signal.fftconvolve(y, y, mode="full")

    Y = scipy.fft.fft(y)
    X = scipy.fft.fftfreq(x.size, x[1] - x[0])

    X = scipy.fft.fftshift(X)
    Y = scipy.fft.fftshift(Y)

    fig, ax = plt.subplots(nrows=1, ncols=2)


    ax[0].plot(K)
    ax[1].plot(X,Y)
    k0 = np.logspace( np.log10(spectrum.KT[0]), np.log10(spectrum.KT[-1]), 10**5)
    ax[1].plot(k0, S(k0))

    plt.show()
