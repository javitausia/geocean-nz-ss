# arrays
import numpy as np
import xarray as xr

# maths
import numbers

# plotting
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# custom
from ..config import default_location
from .config import _figsize_width, _figsize_height, _figsize, \
    _fontsize_title
from .utils import plot_ccrs_nz


def qqplot(x, y, min_value=-0.3, max_value=0.6,
           quantiles=None, interpolation='nearest', ax=None, rug=False,
           rug_length=0.05, rug_kwargs=None, **kwargs):

    """Draw a quantile-quantile plot for `x` versus `y`.

    Parameters
    ----------
    x, y : array-like
        One-dimensional numeric arrays.

    ax : matplotlib.axes.Axes, optional
        Axes on which to plot. If not provided, the current axes will be used.

    quantiles : int or array-like, optional
        Quantiles to include in the plot. This can be an array of quantiles, in
        which case only the specified quantiles of `x` and `y` will be plotted.
        If this is an int `n`, then the quantiles will be `n` evenly spaced
        points between 0 and 1. If this is None, then `min(len(x), len(y))`
        evenly spaced quantiles between 0 and 1 will be computed.

    interpolation : {‘linear’, ‘lower’, ‘higher’, ‘midpoint’, ‘nearest’}
        Specify the interpolation method used to find quantiles when `quantiles`
        is an int or None. See the documentation for numpy.quantile().

    rug : bool, optional
        If True, draw a rug plot representing both samples on the horizontal and
        vertical axes. If False, no rug plot is drawn.

    rug_length : float in [0, 1], optional
        Specifies the length of the rug plot lines as a fraction of the total
        vertical or horizontal length.

    rug_kwargs : dict of keyword arguments
        Keyword arguments to pass to matplotlib.axes.Axes.axvline() and
        matplotlib.axes.Axes.axhline() when drawing rug plots.

    kwargs : dict of keyword arguments
        Keyword arguments to pass to matplotlib.axes.Axes.scatter() when drawing
        the q-q plot.
    """

    # Get current axes if none are provided
    if ax is None:
        ax = plt.gca()

    if quantiles is None:
        # quantiles = min(len(x), len(y))
        quantiles = 2000

    # Compute quantiles of the two samples
    if isinstance(quantiles, numbers.Integral):
        quantiles = np.linspace(start=0, stop=1, num=int(quantiles))
    else:
        quantiles = np.atleast_1d(np.sort(quantiles))
    x_quantiles = np.quantile(x, quantiles, interpolation=interpolation)
    y_quantiles = np.quantile(y, quantiles, interpolation=interpolation)

    # Draw the rug plots if requested
    if rug:
        # Default rug plot settings
        rug_x_params = dict(ymin=0, ymax=rug_length, c='gray', alpha=0.5)
        rug_y_params = dict(xmin=0, xmax=rug_length, c='gray', alpha=0.5)

        # Override default setting by any user-specified settings
        if rug_kwargs is not None:
            rug_x_params.update(rug_kwargs)
            rug_y_params.update(rug_kwargs)

        # Draw the rug plots
        for point in x:
            ax.axvline(point, **rug_x_params)
        for point in y:
            ax.axhline(point, **rug_y_params)

    # Draw the q-q plot
    ax.scatter(x_quantiles, y_quantiles, **kwargs)
    ax.plot([min_value,max_value],[min_value,max_value],
            c='royalblue',lw=3,zorder=10)
    ax.set_xlim(min_value,max_value)
    ax.set_ylim(min_value,max_value)
    ax.axis('square')
    

def scatterplot(x, y, ax=None, 
                min_value=-0.3, max_value=0.6,
                **kwargs):
    """
    Plots the data in a scatter plot

    TODO... 
    
    """

    # Get current axes if none are provided
    if ax is None:
        ax = plt.gca()

    # plot the scatter
    ax.scatter(x,y,**kwargs)
    ax.plot([min_value,max_value],[min_value,max_value],
            c='royalblue',lw=3,zorder=10)
    ax.set_xlim(min_value,max_value)
    ax.set_ylim(min_value,max_value)
    ax.axis('square')


def plot_stats(statistics_data, plot_stats, **kwargs):
    """
    Plots the validation data in a map plot

    TODO... 
    
    """

    # create fig and axes
    n_plots = len(plot_stats)
    fig, axes = plt.subplots(
        ncols=n_plots, # maybe just pearson, r_score and si
        figsize=(_figsize_width*5.0,_figsize_height),
        subplot_kw={
            'projection':ccrs.PlateCarree(
                central_longitude=default_location[0]
            )
        }
    )
    for i,var,ax in zip(range(n_plots),plot_stats,axes.flatten()):
        p = xr.plot.scatter(
            statistics_data, # this is the data
            x='longitude',y='latitude',hue=var,cmap='jet',
            ax=ax,transform=ccrs.PlateCarree(),zorder=40
        ) # TODO: add kwargs
        ax.set_facecolor('lightblue')
        ax.set_title(var,fontsize=20)
    # plot map
    plot_ccrs_nz(axes.flatten(),plot_labels=(True,5,5),
                 plot_coastline=(False,None,None))
    # figure title
    fig.suptitle('Model statistics for all the stations!!',
                 fontsize=_fontsize_title,y=1.1)
    # show results
    plt.show()

