# arrays
import numpy as np

# plotting
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# custom
from ..config import default_location, default_region_reduced


def plot_ccrs_nz(axes, # all axes to plot the map
                 plot_location: tuple = (False,default_location),
                 plot_coastline: tuple = (True,'50m'), # ['110m','50m','10m']
                 plot_land: bool = True,
                 plot_region: tuple = (True,default_region_reduced),
                 plot_labels: bool = (True,5,5)):
    """
    Plot the New Zealand basic map with cartopy

    Args:
        axes (matplotlib.Axes): a matplotlib axes where NZ wants
        to be plotted
    """

    for ax in axes:
        if plot_location[0]:
            ax.scatter(*plot_location[1],s=100,c='red',
                       transform=ccrs.PlateCarree())
        if plot_coastline[0]:
            ax.coastlines(resolution=plot_coastline[1],linewidth=2)
        if plot_land:
            ax.add_feature(cfeature.LAND,zorder=10)
        if plot_region[0]:
            ax.set_extent(plot_region[1])
        if plot_labels:
            gl = ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,
                              linewidth=2,color='gray',linestyle='--')
            xlabels = np.arange(plot_region[1][0],plot_region[1][1],plot_labels[1])
            xlabels = np.where(xlabels<180,xlabels,xlabels-360)
            ylabels = np.arange(plot_region[1][3],plot_region[1][2],plot_labels[2])
            gl.xlocator = mticker.FixedLocator(list(xlabels))
            gl.ylocator = mticker.FixedLocator(list(ylabels))  
            gl.xlabels_top = False
            gl.ylabels_right = False

