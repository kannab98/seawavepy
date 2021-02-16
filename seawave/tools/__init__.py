from .. import rc, surface
from .. import kernel

import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

band = ["C", "X", "Ku", "Ka"]

def to_ftp():
    srf = kernel.simple_launch(kernel.cwm)
    print(np.mean(srf[0]), np.var(srf[0]))
    print(np.var(srf[1]), np.var(srf[2]))

    srf0 = kernel.simple_launch(kernel.default)
    print(np.mean(srf0[0]), np.var(srf0[0]))
    print(np.var(srf0[1]), np.var(srf0[2]))


    srf0[1:] = np.rad2deg(np.arctan(srf0[1:]))
    srf[1:] = np.rad2deg(np.arctan(srf[1:]))

    x, y = surface.meshgrid
    z = [srf, srf0]

    df = pd.DataFrame({
                       "x": x.flatten(), 
                       "y": y.flatten(), 
                       "z_cwm": z[0][0].flatten(),
                       "sigmaxx_cwm": z[0][1].flatten(),
                       "sigmayy_cwm": z[0][2].flatten(),
                       "z": z[1][0].flatten(),
                       "sigmaxx": z[1][1].flatten(),
                       "sigmayy": z[1][2].flatten() 
                    })
    df.to_csv("surfaces/surface.tsv", sep="\t", index=False, float_format="%.6E")
    plt.show()

def plotSurface():
    srf = kernel.simple_launch(kernel.cwm)
    print(np.mean(srf[0]), np.var(srf[0]))
    print(np.var(srf[1]), np.var(srf[2]))

    srf0 = kernel.simple_launch(kernel.default)
    print(np.mean(srf0[0]), np.var(srf0[0]))
    print(np.var(srf0[1]), np.var(srf0[2]))


    srf0[1:] = np.rad2deg(np.arctan(srf0[1:]))
    srf[1:] = np.rad2deg(np.arctan(srf[1:]))


    X, Y = surface.meshgrid
    x = X 
    y = Y
    z = [srf, srf0]
    
    labels=["заостренная", "обычная"]
    barlabel = ["$Z$, м", "$arctan(\\frac {\\partial Z}{\\partial X}), град$", "$arctan(\\frac {\\partial Z}{\\partial Y}), град$"]
    name = ["heights", "sigmaxx", "sigmayy"]
    typesrf = ["cwm", "linear"]

    for j, t in enumerate(typesrf):
        for i, n in enumerate(name):
            plt.figure()
            plt.contourf( x, y, z[j][i] , levels=100, vmin=z[1][i].min(), vmax=z[1][i].max())
            bar = plt.colorbar()
            bar.set_label(barlabel[i])
            plt.xlabel("$X$, м")
            plt.ylabel("$Y$, м")
            plt.savefig("surfaces/%s_%s" % (n,t))
            plt.close()

    for i, yy in enumerate(y[0,:3]):
        plt.figure()
        plt.title("$Y=%.2f$, м" % yy)
        for j, t in enumerate(typesrf):
            plt.plot(x[i,:], z[j][0, i, :], label=labels[j])
            plt.xlabel("$X$, м")
            plt.ylabel(barlabel[0])
            plt.legend()

        plt.savefig("surfaces/srf%d" % i)
        plt.close()

    for i, yy in enumerate(y[0,:3]):
        plt.figure()
        plt.title("$Y=%.2f$, м" % yy)
        for j, t in enumerate(typesrf):
            plt.plot(x[i,:], z[j][1, i, :], label=labels[j])
            plt.xlabel("$X$, м")
            plt.ylabel(barlabel[1])
            plt.legend()

        plt.savefig("surfaces/sigmaxx_%d" % i)
        plt.close()

    for i, yy in enumerate(y[0,:3]):
        plt.figure()
        plt.title("$Y=%.2f$, м" % yy)
        for j, t in enumerate(typesrf):
            plt.plot(x[i,:], z[j][2, i, :], label=labels[j])
            plt.xlabel("$X$, м")
            plt.ylabel(barlabel[2])
            plt.legend()

        plt.savefig("surfaces/sigmayy_%d" % i)
        plt.close()



    df = pd.DataFrame({"x": x.flatten(), 
                       "y": y.flatten(), 
                       "z_cwm": z[0][0].flatten(),
                       "sigmaxx_cwm": z[0][1].flatten(),
                       "sigmayy_cwm": z[0][2].flatten(),
                       "z": z[1][0].flatten(),
                       "sigmaxx": z[1][1].flatten(),
                       "sigmayy": z[1][2].flatten() 
                    })
    df.to_csv("surfaces/surface.tsv", sep="\t", index=False, float_format="%.6E")
    plt.show()
