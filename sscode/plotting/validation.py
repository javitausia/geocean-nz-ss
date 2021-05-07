# arrays
import numpy as np

# maths
import numbers

# plotting
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# custom
from ..config import default_location
from .config import _figsize_width, _figsize_height
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

    Args:
        x ([type]): [description]
        y ([type]): [description]
        ax ([type], optional): [description]. Defaults to None.

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


def plot_stats(model_longitudes, model_latitudes,
               statistics_array, **kwargs):

    # TODO: add docstring

    # create fig and axes
    fig, axes = plt.subplots(
        ncols=2,nrows=2,
        figsize=(_figsize_width*3.6,_figsize_height*2.6),
        subplot_kw={
            'projection':ccrs.PlateCarree(
                central_longitude=default_location[0]
            )
        }
    )
    for i,var,ax in zip(range(4),['bias','si','rmse','pearson'],axes.flatten()):
        p = ax.scatter(
            model_longitudes, model_latitudes,
            c=statistics_array[:,i],
            transform=ccrs.PlateCarree(),
            s=30,zorder=40,cmap='jet',
            **kwargs # add extra parameters
        )
        pos_ax = ax.get_position()
        pos_colbar = fig.add_axes([
            pos_ax.x0 + pos_ax.width + 0.07, pos_ax.y0, 0.02, pos_ax.height
        ])
        fig.colorbar(p,cax=pos_colbar)
        ax.set_facecolor('lightblue')
        ax.set_title(var,fontsize=20)
    # plot map
    plot_ccrs_nz(axes.flatten(),plot_labels=[True,10,10])

