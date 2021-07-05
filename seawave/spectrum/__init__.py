from .. import config



__all__ = ["spectrum", "omega", "k"]

__bibtex__ = r"""@article{ruabkova:2019,
    author ={Ryabkova, M. and Karaev, V. and Guo, J. and Titchenko, Yu.},
    title = {A Review of Wave Spectrum Models as Applied to the Problem of Radar Probing of the Sea Surface},
    year = 2019,
    journal = {Journal of Geophysical Research: Oceans},
    pages = {7104--7134}
}"""

from .core import spectrum
from .dispersion import *

spectrum = spectrum()
