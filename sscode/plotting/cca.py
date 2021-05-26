# arrays
import numpy as np

# plotting
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cartopy.crs as ccrs

# custom
from .config import _figsize, _figsize_width, _figsize_height, _fontsize_title
from ..config import default_location, default_region_reduced, default_region
from .utils import plot_ccrs_nz


def plot_ccs(cca_data, # this is all the CCA returned data
             pcs = None, scalers = None,
             n_plot: int = 3,
             region: tuple = default_region):
    """
    Plot the EOFs/PCs for the n_plot first components

    Args:
        cca_data (xarray.Dataset): This is the data from CCA_Analysis()
        n_plot (int, optional): Number of PCs/EOFs to plot. 
            - Defaults to 3.
        region (tuple, optional): Region to plot the data in.
            - Defaults to default_region
    """

    # date axis locator
    yloc1 = mdates.YearLocator(1)
    yfmt = mdates.DateFormatter('%Y')

    # calculate number of different EOFs in data
    n_features = len(cca_data.n_features.values)
    n_targets = len(cca_data.n_targets.values)
    n_components = len(cca_data.n_components.values)
    n_lons_pres, n_lons_ss = len(pcs[0].n_lon.values), len(pcs[1].n_lon.values)
    n_lats_pres, n_lats_ss = len(pcs[0].n_lat.values), len(pcs[1].n_lat.values)
    print('\n plotting {} components with slp and ss EOFs... \n'\
        .format(n_plot))
    print('\n being the EOFs the reconstructed cannonical loadings \n')

    # plot all the components / EOFs / CCs
    for i_comp in range(n_plot):

        # plot EOFs
        fig, axes = plt.subplots(ncols=2,
            figsize=(_figsize_width*4.4,_figsize_height*1.4),
            subplot_kw={'projection':ccrs.PlateCarree(
                central_longitude=default_location[0]
            )}
        )
        # calculate the transformed EOFs
        recon_eofs = [
            np.matmul(
                cca_data.x_loadings[:,i_comp].values.reshape(1,-1),
                pcs[0].EOFs[:n_features,:].values
            ).reshape(n_lats_pres,n_lons_pres),
            np.matmul(
                cca_data.y_loadings[:,i_comp].values.reshape(1,-1),
                pcs[1].EOFs[:n_targets,:].values
            ).reshape(n_lats_ss,n_lons_ss)
        ] # TODO: check standarization!!
        recon_eofs_ttls = ['SLP - EOF recon', 'SS - EOF recon']
        # different EOFs plotting
        for ix,ax in enumerate(axes): # num_axes = n_EOFs
            eofp = ax.pcolormesh(
                pcs[ix].pcs_lon.values,
                pcs[ix].pcs_lat.values,
                recon_eofs[ix],
                transform=ccrs.PlateCarree(),
                cmap='RdBu_r',clim=(-1,1)
            )
            ax.set_title(recon_eofs_ttls[ix])
            pos_ax = ax.get_position()
            pos_colbar = fig.add_axes([
                pos_ax.x0 + pos_ax.width + 0.01, 
                pos_ax.y0, 0.02, pos_ax.height
            ])
            fig.colorbar(eofp,cax=pos_colbar)
        # plot nz map
        plot_ccrs_nz(axes,plot_land=False,plot_region=(True,region),
                     plot_labels=(True,10,10))

        # TODO: plot variance in title
        # fig.suptitle('EOF {}: {:.1%} of explained variance'.format(
        #     i_comp+1, (variance/np.sum(variance))[i_comp]
        # ), fontsize=_fontsize_title)

        fig.suptitle('EOF reconstructions...!!',
                     fontsize=_fontsize_title)

        # plot the PCs
        fig, ax = plt.subplots(figsize=_figsize)
        cca_data.x_scores.isel(n_components=i_comp).plot(
            ax=ax,c='k',alpha=0.8,label='Pressure {} - CC'.format(i_comp+1)
        )
        cca_data.y_scores.isel(n_components=i_comp).plot(
            ax=ax,c='grey',linestyle='--',label='StormSurge {} - CC'.format(i_comp+1)
        )
        fig.suptitle(
            'Correlation between CCs is: {}'.format(
                round(np.corrcoef(
                    cca_data.x_scores.isel(n_components=i_comp).values,
                    cca_data.y_scores.isel(n_components=i_comp).values
                )[0,1],2) # round correlation
            ), 
            fontsize=_fontsize_title)
        # configure axis
        ax.set_xlim(cca_data.time.values[0],cca_data.time.values[-1])
        ax.xaxis.set_major_locator(yloc1)
        ax.xaxis.set_major_formatter(yfmt)
        ax.grid(True,which='both',axis='x',linestyle='--',color='grey')
        ax.tick_params(axis='both',which='major',labelsize=8)
        
        # show plots
        plt.show()

