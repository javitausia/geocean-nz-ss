"""
This init file contains the basic information of sscode
"""

__version__     = '0.1.0'
__author__      = 'GeoOcean (UC)'
__contact__     = 'tausiaj@unican.es'
__url__         = 'https://github.com/javitausia/geocean-nz-ss/sscode'
__description__ = 'Collection of customized tools for statistical storm surge data analysis'
__keywords__    = 'ocean data statistical analysis storm-surge'

# custom libs
from . import config
from . import utils
from . import data
from . import pca

# basic
import os, glob, sys

# arrays
import numpy as np
import pandas as pd
import xarray as xr

# plotting
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

