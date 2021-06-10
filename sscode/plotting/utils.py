# arrays
import numpy as np
from math import sqrt

# plotting
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import colors, cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# custom
from ..config import default_location, default_region_reduced


def plot_ccrs_nz(axes, # all axes to plot the map
                 plot_location: tuple = (False,default_location),
                 plot_coastline: tuple = (True,'50m',2), # ['110m','50m','10m']
                 plot_land: bool = True,
                 plot_region: tuple = (True,default_region_reduced),
                 plot_labels: bool = (True,10,10)):
    """
    Plot the New Zealand basic map with cartopy

    Args:
        axes (matplotlib.Axes): a matplotlib axes where NZ wants
        to be plotted
    """

    # check size of region to plot
    if plot_region[0]:
        dist = plot_region[1][1]-plot_region[1][0]
        default_dist = default_region_reduced[1]-default_region_reduced[0]
        plot_region = plot_region if dist>default_dist else (True,default_region_reduced)

    # plot map in all the axes
    for ax in axes:
        if plot_location[0]:
            ax.scatter(*plot_location[1],s=50,c='red',
                       zorder=20,transform=ccrs.PlateCarree())
        if plot_coastline[0]:
            ax.coastlines(resolution=plot_coastline[1],
                          linewidth=plot_coastline[2],
                          zorder=12)
        if plot_land:
            ax.add_feature(cfeature.LAND,zorder=10)
        if plot_region[0]:
            ax.set_extent(plot_region[1])
        if plot_labels[0]:
            gl = ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,
                              linewidth=2,color='gray',linestyle='--')
            xlabels = np.arange(plot_region[1][0],plot_region[1][1],plot_labels[1])
            xlabels = np.where(xlabels<180,xlabels,xlabels-360)
            ylabels = np.arange(plot_region[1][3],plot_region[1][2],plot_labels[2])
            gl.xlocator = mticker.FixedLocator(list(xlabels))
            gl.ylocator = mticker.FixedLocator(list(ylabels))  
            gl.xlabels_top = False
            gl.ylabels_right = False
            
            
def get_n_colors(cmap,n_colors):
    """
    Summary
    """
    
    # get the cmap from attrs
    if type(cmap)==str:
        cmap = cm.get_cmap(cmap)
    elif type(cmap)==list:
        cmap = colors.LinearSegmentedColormap.from_list(
            'new_cmap',cmap
        )
    # get the step
    step = 1.0/n_colors
    # save colors to use
    colors_to_use = []
    for icol in range(n_colors):
        colors_to_use.append(cmap(step*icol))
        
    return cmap, colors_to_use


def colors_dwt(num_clusters):

    # 42 DWT colors
    l_colors_dwt = [
        (1.0000, 0.1344, 0.0021),
        (1.0000, 0.2669, 0.0022),
        (1.0000, 0.5317, 0.0024),
        (1.0000, 0.6641, 0.0025),
        (1.0000, 0.9287, 0.0028),
        (0.9430, 1.0000, 0.0029),
        (0.6785, 1.0000, 0.0031),
        (0.5463, 1.0000, 0.0032),
        (0.2821, 1.0000, 0.0035),
        (0.1500, 1.0000, 0.0036),
        (0.0038, 1.0000, 0.1217),
        (0.0039, 1.0000, 0.2539),
        (0.0039, 1.0000, 0.4901),
        (0.0039, 1.0000, 0.6082),
        (0.0039, 1.0000, 0.8444),
        (0.0039, 1.0000, 0.9625),
        (0.0039, 0.8052, 1.0000),
        (0.0039, 0.6872, 1.0000),
        (0.0040, 0.4510, 1.0000),
        (0.0040, 0.3329, 1.0000),
        (0.0040, 0.0967, 1.0000),
        (0.1474, 0.0040, 1.0000),
        (0.2655, 0.0040, 1.0000),
        (0.5017, 0.0040, 1.0000),
        (0.6198, 0.0040, 1.0000),
        (0.7965, 0.0040, 1.0000),
        (0.8848, 0.0040, 1.0000),
        (1.0000, 0.0040, 0.9424),
        (1.0000, 0.0040, 0.8541),
        (1.0000, 0.0040, 0.6774),
        (1.0000, 0.0040, 0.5890),
        (1.0000, 0.0040, 0.4124),
        (1.0000, 0.0040, 0.3240),
        (1.0000, 0.0040, 0.1473),
        (0.9190, 0.1564, 0.2476),
        (0.7529, 0.3782, 0.4051),
        (0.6699, 0.4477, 0.4584),
        (0.5200, 0.5200, 0.5200),
        (0.4595, 0.4595, 0.4595),
        (0.4100, 0.4100, 0.4100),
        (0.3706, 0.3706, 0.3706),
        (0.2000, 0.2000, 0.2000),
        (     0, 0, 0),
    ]

    # get first N colors 
    np_colors_base = np.array(l_colors_dwt)
    np_colors_rgb = np_colors_base[:num_clusters]

    return np_colors_rgb


def custom_cmap(numcolors, map1, m1ini, m1end, map2, m2ini, m2end):
    '''
    Generate custom colormap
    Example: Red-Orange-Yellow-Green-Blue -- map1='YlOrRd' map2='YlGnBu_r'
    mXini, mXend:   colormap range of colors
    numcolors:      number of colors (100-continuous, 15-discretization)
    '''

    # color maps
    cmap1 = plt.get_cmap(map1, numcolors)
    cmap2 = plt.get_cmap(map2, numcolors)

    # custom color ranges
    cmap1v = colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(
                n=cmap1.name, a=m1ini, b=m1end),
            cmap1(np.linspace(m1ini,m1end,100))
    )
    cmap2v = colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(
                n=cmap2.name, a=m2ini, b=m2end),
            cmap2(np.linspace(m2ini,m2end,100))
    )

    top = cm.get_cmap(cmap1v, 128)
    bottom = cm.get_cmap(cmap2v, 128)

    newcolors = np.vstack((
        bottom(np.linspace(0,1,128)),
        top(np.linspace(0,1,128))
    ))
    newcmp = colors.ListedColormap(newcolors, name='OrangeBlue')

    return newcmp

