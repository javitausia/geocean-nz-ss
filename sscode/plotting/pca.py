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

pca_EOFs_ttls = [
    'SLP in t', 'GRADIENT in t', 'SLP in t-1', 'GRADIENT in t-1',
    'SLP in t-2', 'GRADIENT in t-2', 'Winds in t']
pca_EOFs_borders = [2000, 10000, 2000, 10000, 2000, 10000, 1]


def plot_pcs(pca_data, pcs_scaler = None,
             n_plot: int = 3, pca_ttls = None, pca_borders = None,
             region: tuple = default_region):
    """
    Plot the EOFs/PCs for the n_plot first components

    Args:
        pca_data (xarray.Dataset): This is the data from PCA_DynamicPred()
        n_plot (int, optional): Number of PCs/EOFs to plot. 
            - Defaults to 3.
        region (tuple, optional): Region to plot the data in.
            - Defaults to default_region
    """

    # date axis locator
    yloc1 = mdates.YearLocator(1)
    yfmt = mdates.DateFormatter('%Y')

    # check ttls
    # pca_ttls = pca_ttls if pca_ttls else pca_EOFs_ttls

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
        eof_width = _figsize[0] / n_EOFs
        fig, axes = plt.subplots(ncols=n_EOFs,
            figsize=(eof_width*n_EOFs*1.2,_figsize_height),
            subplot_kw={'projection':ccrs.PlateCarree(
                central_longitude=default_location[0]
            )}
        )
        # calculate the transformed EOFs
        if pcs_scaler:
            slp_eof = pcs_scaler.inverse_transform(
                pca_data.EOFs.isel(n_components=i_comp).values * \
                    np.sqrt(variance[i_comp]).reshape(1,-1)
            ).reshape(-1)
        else:
            slp_eof = (pca_data.EOFs.isel(n_components=i_comp).values * \
                np.sqrt(variance[i_comp])
            ).reshape(-1)
        # different EOFs
        axes = axes if n_EOFs>1 else [axes] # allow individual plotting
        # plot colorbars
        plot_cbar = True if n_EOFs<=3 else False
        for ix,ax in enumerate(axes): # num_axes = n_EOFs
            eof_values = slp_eof[ix*(n_lons*n_lats):(ix+1)*(n_lons*n_lats)]\
                .reshape(n_lats,n_lons)
            if np.nanmean(eof_values)>80000:
                eof_values = eof_values-101300
            if not pca_borders:
                border = max(
                    np.abs(np.nanmin(eof_values)),
                    np.nanmax(eof_values)
                )
            else:
                border = pca_borders[ix]
            eofp = ax.pcolormesh(
                pca_data.pcs_lon.values,
                pca_data.pcs_lat.values,
                eof_values, # check coherence 
                transform=ccrs.PlateCarree(),
                cmap='RdBu_r',vmin=-border,vmax=border
            )
            ax.set_title(pca_ttls[ix]) if pca_ttls else None
            if plot_cbar:
                pos_ax = ax.get_position()
                pos_colbar = fig.add_axes([
                    pos_ax.x0 + pos_ax.width + 0.02, 
                    pos_ax.y0, 0.02, pos_ax.height
                ])
                fig.colorbar(eofp,cax=pos_colbar)
        # plot nz map
        plot_ccrs_nz(axes,plot_land=False,plot_region=(True,region),
                     plot_labels=(True,10,10))
        # plot variance in title
        fig.suptitle('EOF {}: {:.1%} of explained variance'.format(
            i_comp+1, (variance/np.sum(variance))[i_comp]
        ), fontsize=_fontsize_title)

        # plot the PCs
        fig, ax = plt.subplots(figsize=_figsize)
        (pca_data.PCs.isel(n_components=i_comp)/\
            np.sqrt(variance[i_comp])).plot(ax=ax,c='k')
        fig.suptitle('PC {}'.format(i_comp+1), fontsize=_fontsize_title)
        # configure axis
        ax.set_xlim(pca_data.time.values[0],pca_data.time.values[-1])
        ax.xaxis.set_major_locator(yloc1)
        ax.xaxis.set_major_formatter(yfmt)
        ax.grid(True,which='both',axis='x',linestyle='--',color='grey')
        ax.tick_params(axis='both',which='major',labelsize=8)

        # show plots
        plt.show()


def plot_recon_pcs(pca_data, pcs_scaler, 
                   n_pcs_recon: int = 10,
                   region: tuple = default_region,
                   return_slp: bool = False,
                   pca_ttls = None):
    """
    Plot the recon slp fields for n_components

    Args:
        pca_data (xarray.Dataset): This is the data from PCA_DynamicPred()
        pcs_scaler (sklearn.Fit): This is the standariser to transform PCs
        n_pcs_recon (int, optional): Number of PCs/EOFs to reconstruct. 
            - Defaults to 10.
        region (tuple, optional): Region to plot the data in.
            - Defaults to default_region
    """

    # check ttls
    pca_ttls = pca_ttls if pca_ttls else pca_EOFs_ttls

    # reconstruct the SLP values
    slp_stan = np.repeat(
        pca_data.PCs.isel(n_components=0).values,
        len(pca_data.n_features)).reshape(
            len(pca_data.time),len(pca_data.n_features)
        ) * pca_data.EOFs.isel(n_components=0).values
    for i_comp in range(1,n_pcs_recon):
        slp_stan += np.repeat(
            pca_data.PCs.isel(n_components=i_comp).values,
            len(pca_data.n_features)).reshape(
                len(pca_data.time),len(pca_data.n_features)
            ) * pca_data.EOFs.isel(n_components=i_comp).values
    # get the real SLP values
    slp = pcs_scaler.inverse_transform(slp_stan)

    # time to plot the data
    time_to_plot = np.random.randint(slp.shape[0])
    slp_plot = slp[time_to_plot,:]

    # calculate number of different EOFs in data
    n_features = len(pca_data.n_features.values)
    n_lons = len(pca_data.n_lon.values)
    n_lats = len(pca_data.n_lat.values)
    n_EOFs = int(n_features/(n_lons*n_lats))
    print('\n plotting reconstruction of {} EOFs... \n'\
        .format(n_EOFs))
    print('\n being the EOFs the slp, the gradient, in steps t, t-1... \n')
    
    # plot EOFs
    eof_width = _figsize[0] / n_EOFs
    fig, axes = plt.subplots(ncols=n_EOFs,
        figsize=(eof_width*n_EOFs*1.2,_figsize_height),
        subplot_kw={'projection':ccrs.PlateCarree(
            central_longitude=default_location[0]
        )}
    )
    # different EOFs
    axes = axes if n_EOFs>1 else [axes] # allow individual plotting
    # plot colorbars
    plot_cbar = True if n_EOFs<=3 else False
    for ix,ax in enumerate(axes): # num_axes = n_EOFs
        eofp = ax.pcolormesh(
            pca_data.pcs_lon.values,
            pca_data.pcs_lat.values,
            slp_plot[ix*(n_lons*n_lats):(ix+1)*(n_lons*n_lats)]\
                .reshape(n_lats,n_lons),
            transform=ccrs.PlateCarree(),
            cmap='RdBu_r',clim=(-1,1)
        )
        ax.set_title(pca_ttls[ix])
        if plot_cbar:
            pos_ax = ax.get_position()
            pos_colbar = fig.add_axes([
                pos_ax.x0 + pos_ax.width + 0.02, 
                pos_ax.y0, 0.02, pos_ax.height
            ])
            fig.colorbar(eofp,cax=pos_colbar)
    # plot nz map
    plot_ccrs_nz(axes,plot_land=False,plot_region=(True,region),
                 plot_labels=(True,10,10))
    # plot variance in title
    fig.suptitle(
        'Reconstructed SLP with {} PCs in {}'.format(
            n_pcs_recon, str(pca_data.time.values[time_to_plot])[:10]
        ),
        fontsize=_fontsize_title
    )
    # show results
    plt.show()

    # return the reconstructed slp... values
    if return_slp:
        return slp

