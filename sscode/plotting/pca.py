# arrays
import numpy as np

# plotting
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cartopy.crs as ccrs

# custom
from .config import _figsize, _figsize_width, _figsize_height, _fontsize_title
from ..config import default_location, default_region_reduced
from .utils import plot_ccrs_nz

pca_EOFs_ttls = ['SLP in t', 'GRADIENT in t', 'SLP in t-1', 'GRADIENT in t-1']


def plot_pcs(pca_data, n_plot: int = 3,
             region: tuple = default_region_reduced):
    """
    Plot the EOFs/PCs for the n_plot first components

    Args:
        pca_data (xarray.Dataset): This is the data from PCA_DynamicPred()
        n_plot (int, optional): Number of PCs/EOFs to plot. 
            - Defaults to 3.
        region (tuple, optional): Region to plot the data in.
            - Defaults to default_region_reduced
    """

    # date axis locator
    yloc1 = mdates.YearLocator(1)
    yfmt = mdates.DateFormatter('%Y')

    # calculate number of different EOFs in data
    n_features = len(pca_data.n_features.values)
    n_lons = len(pca_data.n_lon.values)
    n_lats = len(pca_data.n_lat.values)
    n_EOFs = int(n_features/(n_lons*n_lats))
    print('\n plotting {} components with {} EOFs + PC... \n'\
        .format(n_plot,n_EOFs))
    print('\n being the EOFs the slp, the gradient, in steps t, t-1 \n')

    # variance of the PCs
    variance = pca_data.variance.values
    
    # plot all the components / EOFs / PCs
    for i_comp in range(n_plot):

        # plot EOFs
        fig, axes = plt.subplots(ncols=n_EOFs,
            figsize=(_figsize_width*n_EOFs*1.5,_figsize_height*1.5),
            subplot_kw={'projection':ccrs.PlateCarree(
                central_longitude=default_location[0]
            )}
        )
        # different EOFs
        for ix,ax in enumerate(axes): # num_axes = n_EOFs
            ax.pcolormesh(
                pca_data.pcs_lon.values,
                pca_data.pcs_lat.values,
                pca_data.EOFs.isel(n_components=i_comp)\
                    .values[ix*(n_lons*n_lats):(ix+1)*(n_lons*n_lats)]\
                    .reshape(n_lons,n_lats) / np.sqrt(variance[i_comp]),
                transform=ccrs.PlateCarree(),
                cmap='RdBu_r',clim=(-1,1)
            )
            ax.set_title(pca_EOFs_ttls[ix])
        # plot nz map
        plot_ccrs_nz(axes,plot_land=False)
        # plot variance in title
        fig.suptitle('EOF {}: {:.1%} of explained variance'.format(
            i_comp, (variance/np.sum(variance))[i_comp]
        ), fontsize=_fontsize_title)

        # plot the PCs
        fig, ax = plt.subplots(figsize=_figsize)
        (pca_data.PCs.isel(n_components=i_comp)/\
            np.sqrt(variance[i_comp])).plot(ax=ax,c='k')
        fig.suptitle('PC {}'.format(i_comp), fontsize=_fontsize_title)
        # configure axis
        ax.set_xlim(pca_data.time.values[0], pca_data.time.values[-1])
        ax.xaxis.set_major_locator(yloc1)
        ax.xaxis.set_major_formatter(yfmt)
        ax.grid(True,which='both',axis='x',linestyle='--',color='grey')
        ax.tick_params(axis='both',which='major',labelsize=8)

