import os
PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))

from .version import __version__
from .neural_network   import *
from .characterization import *
from .data_shuffle     import *
