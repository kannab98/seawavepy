from .. import config



__all__ = ["spectrum", "omega", "k", "dopler"]


from .core import *
from .dispersion import *
spectrum = spectrum()


from .dopler import Dopler
dopler = Dopler()


