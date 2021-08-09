# arrays and geomath
import numpy as np
import xarray as xr
import great_circle_calculator.great_circle_calculator as gcc
from math import sqrt

from xarray.backends.api import open_dataarray

# custom
from .config import default_location, data_path


def calculate_relative_winds(location: tuple = default_location,
                             uw = None, vw = None,
                             lat_name: str = 'lat',
                             lon_name: str = 'lon',
                             delete_direc: int = 180):
    """
    This function calculates the projected winds in the direction to
    the given location. Winds with an angle greater than delete_direc with 
    this direction are not considered.

    Args:
        location (tuple, optional): Location where the winds will be
            projected. Defaults to default_location. Ex: (181,-30)
        uw (xarray-DataArray): u10-component of winds in the direction where 
            they come (era5 raw data). Defaults to None.
        vw (xarray-DataArray): v10-component of winds in the direction where 
            they come (era5 raw data). Defaults to None.

    Returns:
        [xarray-Dataset]: an xarray Dataset with all the variables of
        interest, which are:
            - u10 (winds direction), v10
            - wind_proj: magnitud of the projected winds
            - bearings for all the lat/lon points (using great_circle_calculator)
            - direc_proj_math: mathematical representation of the bearings,
                this is useful when plotting the data
    """

    # we first join uw and vw (change winds to where they go)
    wind = xr.merge([-uw,-vw]).dropna(dim='time',how='all')
    print('\n calculating winds with: \n\n {} \n'.format(
        wind # these are the wind merged components
    )) if True else None
    # calculate directions of winds (where they go)
    wind_direcs = np.arctan2(wind[uw.name].values,
                             wind[vw.name].values) * 180/np.pi
    wind_direcs = np.where(wind_direcs<0,wind_direcs+360,wind_direcs)
    # calculate bearing for each lat/lon
    bearings = calculate_bearings(wind[lat_name].values,
                                  wind[lon_name].values,
                                  location=location)
    # calculate relative directions
    rel_direcs = bearings.reshape(1,*bearings.shape) - wind_direcs
    rel_direcs = np.where(rel_direcs<-180,rel_direcs+360,rel_direcs)
    rel_direcs = np.where(rel_direcs>180,rel_direcs-360,rel_direcs)
    # delete winds in opposite directions
    rel_direcs = np.where(
        (rel_direcs>delete_direc)|(rel_direcs<-delete_direc),np.nan,rel_direcs
    )

    return_winds = wind.assign({
        'wind_proj': (('time',lat_name,lon_name),np.cos(rel_direcs*np.pi/180) * \
            np.sqrt(
                wind[uw.name].values**2 + wind[vw.name].values**2 # add magnitude
            )),
        'bearings': ((lat_name,lon_name),bearings),
        'direc_proj_math': ((lat_name,lon_name),trans_geosdeg2mathdeg(bearings)*np.pi/180)
    }) # final dataset

    return return_winds.assign({
        'wind_proj_mask': (('time',lat_name,lon_name),
            return_winds.wind_proj * xr.open_dataarray(data_path+'/cfsr/cfsr_mapsta.nc')
        ) # TODO: check file existance
    })


def calculate_bearings(latitudes, longitudes,
                       location: tuple = default_location):
    """
    This function calculates the bearings to a point, given the location
    of the point of interest. In our case, this point is in New Zealand

    Args:
        latitudes (numpy-array): Latitudes array of 1d
        longitudes (numpy-array): Longitudes array of 1d
        location (tuple, optional): Location where the bearings will be
        calculated. Defaults to default_location.

    Returns:
        [numpy-array]: 2d numpy-array with all the bearings for each lat/lon
    """

    # check location coords
    location = location if location[0]<180 else (location[0]-360,location[1])
    # calculate the bearings
    bearings = np.zeros((len(latitudes),len(longitudes)))
    for ilat,lat in enumerate(latitudes):
        for ilon,lon in enumerate(longitudes):
            lon = lon if lon<180 else lon-360
            bearings[ilat,ilon] = gcc.bearing_at_p1((lon,lat),location)

    return np.where(bearings>0,bearings,bearings+360)


def trans_geosdeg2mathdeg(geosdir):
    """
    This function transform 0ºN geospatial data to mathematically
    understandable angles

    Args:
        geosdir (numpy-array): Directions in degrees

    Returns:
        [numpy-array]: Mathematically understandable angles in degrees
    """
    
    geosdir = np.where(geosdir<90,90-geosdir,geosdir)
    geosdir = np.where((geosdir>90)&(geosdir<180),-(geosdir-90),geosdir)
    geosdir = np.where((geosdir>180)&(geosdir<270),-90-(geosdir-180),geosdir)
    geosdir = np.where(geosdir>270,(360-geosdir)+90,geosdir)
    
    return geosdir


def degC2degN(degC):
    """
    Converts a wind direction on a unit circle (degrees cartesian) to 
    a direction in nautical convention (degrees north).
    
    Angle in a unit circle: 
    The angle between the horizontal east (E) and the head (pointing outwards), counter-clockwise
    
                ^ N (90)
                |
    W (180) <~~   ~~> E (0)
                |
                v S (270)
    
    Angle in nautical convention: 
    The angle between the vertical up (N) and the tail (pointing inwards), clockwise
    
                | N (0)
                v
    W (270) ~~>   <~~ E (90)
                ^
                | S (180)

    author: Sara Ortega 
    """

    degN = np.mod(-degC + 270, 360)
    
    return degN
 

def degN2degC(degN):
    """
    Converts a wind direction in nautical convention (degrees north) to a  
    a direction on a unit circle (degrees cartesian)

    author: Sara Ortega
    """

    degC = np.mod(-degN + 270, 360)
    
    return degC


def spatial_gradient(xdset, var_name='msl'):
    """
    Calculate spatial gradient

    xdset:
        (longitude, latitude, time), var_name

    returns xdset with new variable "var_name_gradient"
    """

    # TODO:check/ADD ONE ROW/COL EACH SIDE
    try:
        var_grad = np.zeros(xdset[var_name].shape)
    except:
        xdset = xdset.to_dataset()
        var_grad = np.zeros(xdset[var_name].shape)

    # save latitudes to recalculate things
    lat = xdset.latitude.values

    for it in range(len(xdset.time)):
        # sel time
        var_val = xdset[var_name].isel(time=it).values

        # calculate gradient (matrix)
        m_c = var_val[1:-1,1:-1]
        m_l = np.roll(var_val, -1, axis=1)[1:-1,1:-1]
        m_r = np.roll(var_val, +1, axis=1)[1:-1,1:-1]
        m_u = np.roll(var_val, -1, axis=0)[1:-1,1:-1]
        m_d = np.roll(var_val, +1, axis=0)[1:-1,1:-1]
        m_phi = np.pi*np.abs(lat)/180.0
        m_phi = m_phi[1:-1]

        dpx1 = (m_c - m_l)/np.cos(m_phi[:,None])
        dpx2 = (m_r - m_c)/np.cos(m_phi[:,None])
        dpy1 = m_c - m_d
        dpy2 = m_u - m_c

        vg = (dpx1**2+dpx2**2)/2 + (dpy1**2+dpy2**2)/2
        var_grad[it, 1:-1, 1:-1] = vg

    # store gradient
    xdset['{0}_gradient'.format(var_name)]= (
        ('time', 'latitude', 'longitude'), var_grad)

    return xdset


def GetBestRowsCols(n):
    
    # try to square number n, used at gridspec plots'

    sqrt_n = sqrt(n)
    if sqrt_n.is_integer():
        n_r = int(sqrt_n)
        n_c = int(sqrt_n)
    else:
        l_div = GetDivisors(n)
        n_c = l_div[int(len(l_div)/2)]
        n_r = int(n/n_c)

    return n_r, n_c


def GetDivisors(x):
    
    l_div = []
    i = 1
    while i<x:
        if x%i == 0:
            l_div.append(i)
        i = i + 1
    return l_div

