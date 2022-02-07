"""
This init file contains the basic information of sscode
"""

__version__     = '0.1.0'
__author__      = 'GeoOcean (UC) and MetOcean'
__contact__     = 'tausiaj@unican.es and s.delaux@metocean.co.nz'
__url__         = 'https://github.com/javitausia/geocean-nz-ss/sscode'
__description__ = 'Collection of customized tools for statistical storm surge data analysis'
__keywords__    = 'ocean data statistical analysis storm-surge reconstruction'

# custom libs
from . import config
from . import utils
from . import data
from . import pca
from . import experiment
from . import linear, knn, xgboost

# basic
import os, glob, sys

# arrays
import numpy as np
import pandas as pd
import xarray as xr

# plotting
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

