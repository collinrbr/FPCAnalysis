from . import array_ops as ao
from . import analysis as an
from . import data_gkeyll as dg
from . import data_h5 as dh5
from . import data_netcdf4 as dnc
from . import data_tristan as dt
from . import fpc as fpc
from . import frametransform as ft
from . import metadata as md
from . import plume as pl
from . import supp as sp
from . import wavemode as wv

from .plot import debug as pdg
from .plot import fourier as pfr
from .plot import plotplume as ppl
from .plot import table as ptb
from .plot import resultsmanager as prm
from .plot import velspace as pvl
from .plot import twod as ptd



def version():
	"""Return the version of the package."""
	import pkg_resources
	try:
		import pkg_resources
		__version__ = pkg_resources.get_distribution(__name__).version
	except:
		print("Error, could not find version (likely because we are using a development version (created by pip install -e .)! Returning hard coded value of version...")
		__version__ = '1.0.0'
	return __version__
