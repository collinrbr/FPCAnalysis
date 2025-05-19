from . import array_ops as ao
from . import analysis as anl
from . import data_gkeyll as dg
from . import data_dhybridr as ddhr
from . import data_netcdf4 as dnc
from . import data_tristan as dtr
from . import fpc as fpc
from . import frametransform as ft
from . import metadata as md
from . import plume as pl
from . import supp as sp
from . import wavemode as wv
from . import flux as flx

from .plot import debug as pltdebug
from .plot import fourier as pltfr
from .plot import plotplume as ppl
from .plot import table as ptb
from .plot import resultsmanager as prm
from .plot import velspace as pltvv
from .plot import twod as plt2d
from .plot import oned as plt1d
from .plot import fluxes as pltflx

def version():
	"""Return the version of the package."""
	import pkg_resources
	try:
		import pkg_resources
		__version__ = pkg_resources.get_distribution(__name__).version
	except:
		print("Error, could not find version... Returning hard coded value of version...")
		__version__ = '1.0.1'
	return __version__
