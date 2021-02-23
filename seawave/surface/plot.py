
def plot(self, x, y, surf, label = "default"):

    surf = self.reshape(surf)
    x = self.reshape(x)
    y = self.reshape(y)

    fig, ax = plt.subplots()
    mappable = ax.contourf(x, y,surf, levels=100)
    ax.set_xlabel("$x,$ м")
    ax.set_ylabel("$y,$ м")
    bar = fig.colorbar(mappable=mappable, ax=ax)
    bar.set_label("высота, м")

    ax.set_title("$U_{10} = %.0f $ м/с" % (spectrum.U10) )
    ax.text(0.05,0.95,
        '\n'.join((
                '$\\sigma^2_s=%.5f$' % (np.std(surf)**2/2),
                '$\\sigma^2_{0s}=%.5f$' % (spectrum.sigma_sqr),
                '$\\langle z \\rangle = %.5f$' % (np.mean(surf)),
        )),
        verticalalignment='top',transform=ax.transAxes,)

    fig.savefig("%s" % (label))

