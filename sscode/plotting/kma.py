# arrays
import numpy as np
from numpy.core.fromnumeric import size
from numpy.core.numeric import False_
import pandas as pd

# time
from datetime import date, datetime
from cftime._cftime import DatetimeGregorian

# plotting
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs

# custom
from .config import _figsize_width, _figsize_height, _fontsize_title, \
    _fontsize_label, _mbar_diff
from ..config import default_location
from ..utils import GetBestRowsCols
from .utils import colors_dwt, custom_cmap
# from ..kma import cluster_probabilities


def Plot_DWTs_Mean_Anom(xds_KMA, kind='anom', 
                        scale_: bool = True,
                        press_diff = _mbar_diff):
    '''
    Plot Daily Weather Types (bmus mean)
    kind - mean/anom
    '''

    # bmus = xds_KMA['bmus'].values # add sorted
    n_clusters = len(xds_KMA.n_clusters.values)

    # var_max = np.nanmax(xds_var.values)
    # var_min = np.nanmin(xds_var.values)
    scale = 100.0 if scale_ else 1.0 # scale from Pa to mbar

    # get number of rows and cols for gridplot 
    n_rows, n_cols = GetBestRowsCols(n_clusters)

    # get cluster colors
    cs_dwt = colors_dwt(n_clusters)

    # plotting the SLP clusters
    fig, axes = plt.subplots(
        ncols=n_cols,nrows=n_rows,
        figsize=(n_cols*(_figsize_width/1.6),
                 n_rows*(_figsize_height/1.8)),
        subplot_kw={
            'projection': ccrs.PlateCarree(
                central_longitude=default_location[0]
            )
        }
    )
    fig.subplots_adjust(hspace=0.02,wspace=0.02)
    for i,ax in enumerate(axes.flatten()):
        if kind=='anom':
            # p = ((xds_var.isel(time=np.where(xds_KMA.bmus.values==i)[0])\
            #     .mean(dim='time') - (
            #     xds_var.mean(dim='time')
            # )) / scale).plot(
            #         cmap='RdBu_r',vmin=-press_diff,vmax=press_diff,
            #         ax=ax,transform=ccrs.PlateCarree(),
            #         add_colorbar=False
            #     )
            pslp = ((xds_KMA.slp_clusters.isel(n_clusters=i) - \
                xds_KMA.slp_clusters.mean(dim='n_clusters')) / scale).plot(
                    cmap='RdBu_r',vmin=-press_diff,vmax=press_diff,
                    ax=ax,transform=ccrs.PlateCarree(),
                    add_colorbar=False
                )
        else:
            # p = ((xds_var.isel(time=np.where(xds_KMA.bmus.values==i)[0])\
            #     .mean(dim='time')) / scale).plot(
            #         cmap='RdBu_r',vmin=1013-press_diff,vmax=1013+press_diff,
            #         ax=ax,transform=ccrs.PlateCarree(),
            #         add_colorbar=False
            #     )
            pslp = (xds_KMA.slp_clusters.isel(n_clusters=i) / scale).plot(
                    cmap='RdBu_r',vmin=1013-press_diff,vmax=1013+press_diff,
                    ax=ax,transform=ccrs.PlateCarree(),
                    add_colorbar=False
                )
        # axis customization
        ax.coastlines(linewidth=2)
        ax.set_title('')
        ax.text(
            184.5,-24,str(i), # add better text?
            transform=ccrs.PlateCarree(),
            zorder=100,size=12,
            bbox=dict(boxstyle='round',
                ec=(1.,0.5,0.5),
                fc=(1.,0.8,0.8),
            )
        )

    # add the colorbar
    cbar_ax = fig.add_axes([0.135,0.10,0.75,0.02])
    cb = fig.colorbar(pslp,cax=cbar_ax,orientation='horizontal')
    if kind=='mean':
        cb.set_label('Pressure [mbar]',fontsize=_fontsize_label)
    elif kind=='anom':
        cb.set_label('Pressure anomalies [mbar]',fontsize=_fontsize_label)

    # plotting the ss max clusters
    fig, axes = plt.subplots(
        ncols=n_cols,nrows=n_rows,
        figsize=(n_cols*(_figsize_width/1.6),
                 n_rows*(_figsize_height/1.8)),
        subplot_kw={
            'projection': ccrs.PlateCarree(
                central_longitude=default_location[0]
            )
        }
    )
    fig.subplots_adjust(hspace=0.02,wspace=0.02)
    for i,ax in enumerate(axes.flatten()):
        pss = xds_KMA.ss_clusters_mean.isel(n_clusters=i).plot(
            cmap=custom_cmap(15,'YlOrRd',0.15,0.9,'YlGnBu_r',0,0.85),
            vmin=-0.2,vmax=0.3,add_colorbar=False,
            ax=ax,transform=ccrs.PlateCarree(),
        )
        # axis customization
        ax.coastlines(linewidth=2)
        ax.set_title('')
        ax.text(
            182,-33.3,str(i), # add better text?
            transform=ccrs.PlateCarree(),
            zorder=100,size=12,
            bbox=dict(boxstyle='round',
                ec=(1.,0.5,0.5),
                fc=(1.,0.8,0.8),
            )
        )

    # add the colorbar
    cbar_ax = fig.add_axes([0.135,0.10,0.75,0.02])
    cb = fig.colorbar(pss,cax=cbar_ax,orientation='horizontal')
    cb.set_label('Storm surge mean [m]',fontsize=_fontsize_label)

    # show results
    plt.show()


def Plot_DWTs_Probs(bmus, n_clusters, show_cbar=True):
    '''
    Plot Daily Weather Types bmus probabilities
    '''

    wt_set = np.arange(n_clusters) + 1

    # best rows cols combination
    n_rows, n_cols = GetBestRowsCols(n_clusters)

    # figure
    fig = plt.figure(figsize=(18,14))

    # layout
    gs = gridspec.GridSpec(4, 7, wspace=0.10, hspace=0.25)

    # list all plots params
    l_months = [
        (1, 'January',   gs[1,3]),
        (2, 'February',  gs[2,3]),
        (3, 'March',     gs[0,4]),
        (4, 'April',     gs[1,4]),
        (5, 'May',       gs[2,4]),
        (6, 'June',      gs[0,5]),
        (7, 'July',      gs[1,5]),
        (8, 'August',    gs[2,5]),
        (9, 'September', gs[0,6]),
        (10, 'October',  gs[1,6]),
        (11, 'November', gs[2,6]),
        (12, 'December', gs[0,3]),
    ]

    l_3months = [
        ([12, 1, 2],  'DJF', gs[3,3]),
        ([3, 4, 5],   'MAM', gs[3,4]),
        ([6, 7, 8],   'JJA', gs[3,5]),
        ([9, 10, 11], 'SON', gs[3,6]),
    ]

    # plot total probabilities
    c_T = cluster_probabilities(bmus, wt_set)
    C_T = np.reshape(c_T, (n_rows, n_cols))

    ax_probs_T = plt.subplot(gs[:2, :3])
    pc = axplot_WT_Probs(ax_probs_T, C_T, ttl = 'DWT Probabilities')

    # plot counts histogram
    ax_hist = plt.subplot(gs[2:, :3])
    axplot_WT_Hist(ax_hist, bmus, n_clusters, ttl = 'DWT Counts')

    # plot probabilities by month
    vmax = 0.15
    for m_ix, m_name, m_gs in l_months:

        # get probs matrix
        c_M = ClusterProbs_Month(bmus, bmus.train_time.values, wt_set, m_ix)
        C_M = np.reshape(c_M, (n_rows, n_cols))

        # plot axes
        ax_M = plt.subplot(m_gs)
        axplot_WT_Probs(ax_M, C_M, ttl = m_name, vmax=vmax)

    # TODO: add second colorbar?

    # plot probabilities by 3 month sets
    vmax = 0.15
    for m_ix, m_name, m_gs in l_3months:

        # get probs matrix
        c_M = ClusterProbs_Month(bmus, bmus.train_time.values, wt_set, m_ix)
        C_M = np.reshape(c_M, (n_rows, n_cols))

        # plot axes
        ax_M = plt.subplot(m_gs)
        axplot_WT_Probs(ax_M, C_M, ttl = m_name, vmax=vmax, cmap='Greens')

    # add custom colorbar
    if show_cbar:
        pp = ax_probs_T.get_position()
        # cbar_ax = fig.add_axes([pp.x1+0.02, pp.y0, 0.02, pp.y1 - pp.y0])
        cbar_ax = fig.add_axes([pp.x0+0.02, pp.y0+0.03, pp.x1-0.04 - pp.x0, 0.02])
        cb = fig.colorbar(pc, cax=cbar_ax, cmap='Blues', orientation='horizontal')
        cb.ax.tick_params(labelsize=8)

    # show probability results
    plt.show()


def cluster_probabilities(series, set_values):
    'return series probabilities for each item at set_values'

    us, cs = np.unique(series, return_counts=True)
    d_count = dict(zip(us, cs))

    # cluster probabilities
    cprobs = np.zeros((len(set_values)))
    for i, c in enumerate(set_values):
        cprobs[i] = 1.0*d_count[c]/len(series) if c in d_count.keys() else 0.0

    return cprobs


def ClusterProbs_Month(bmus, time, wt_set, month_ix):
    'Returns Cluster probs by month_ix'

    # get months
    _, months, _ = get_years_months_days(time)

    if isinstance(month_ix, list):

        # get each month indexes
        l_ix = []
        for m_ix in month_ix:
            ixs = np.where(months == m_ix)[0]
            l_ix.append(ixs)

        # get all indexes     
        ix = np.unique(np.concatenate(tuple(l_ix)))

    else:
        ix = np.where(months == month_ix)[0]

    bmus_sel = bmus[ix]

    return cluster_probabilities(bmus_sel, wt_set)


def axplot_WT_Probs(ax, wt_probs,
                     ttl = '', vmin = 0, vmax = 0.1,
                     cmap = 'Blues', caxis='black'):
    'axes plot WT cluster probabilities'

    # clsuter transition plot
    pc = ax.pcolor(
        np.flipud(wt_probs),
        cmap=cmap, vmin=vmin, vmax=vmax,
        edgecolors='k',
    )

    # customize axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(ttl, {'fontsize':_fontsize_title, 'fontweight':'bold'})

    # axis color
    plt.setp(ax.spines.values(), color=caxis)
    plt.setp(
        [ax.get_xticklines(), ax.get_yticklines()],
        color=caxis,
    )

    # axis linewidth
    if caxis != 'black':
        plt.setp(ax.spines.values(), linewidth=3)

    return pc


def axplot_WT_Hist(ax, bmus, n_clusters, ttl=''):
    'axes plot WT cluster count histogram'

    # cluster transition plot
    ax.hist(
        bmus,
        bins = np.arange(1, n_clusters+2),
        edgecolor='k'
    )

    # customize axes
    # ax.grid('y')

    ax.set_xticks(np.arange(1,n_clusters+1)+0.5)
    ax.set_xticklabels(np.arange(1,n_clusters+1))
    ax.set_xlim([1, n_clusters+1])
    ax.tick_params(axis='both', which='major', labelsize=6)

    ax.set_title(ttl, {'fontsize':_fontsize_title, 'fontweight':'bold'})


def get_years_months_days(time):
    '''
    Returns years, months, days of time in separete lists

    (Used to avoid problems with dates type)
    '''

    t0 = time[0]
    if isinstance(t0, (date, datetime, DatetimeGregorian)):
        ys = np.asarray([x.year for x in time])
        ms = np.asarray([x.month for x in time])
        ds = np.asarray([x.day for x in time])

    else:
        tpd = pd.DatetimeIndex(time)
        ys = tpd.year
        ms = tpd.month
        ds = tpd.day

    return ys, ms, ds

