# arrays and geomath
import numpy as np
import xarray as xr
import great_circle_calculator.great_circle_calculator as gcc

# custom
from .config import default_location


def calculate_relative_winds(location: tuple = default_location,
                             uw = None, vw = None):
    """
    This function calculates the projected winds in the direction to
    the given location. Winds with an angle greater than 90º with this
    direction are not considered.

    Args:
        location (tuple, optional): Location where the winds will be
        projected. Defaults to default_location.
        uw (xarray-DataArray): u10-component of winds in the direction where 
        they come (era5 raw data). Defaults to None.
        vw (xarray-DataArray): v10-component of winds in the direction where 
        they come (era5 raw data). Defaults to None.

    Returns:
        [xarray-Dataset]: an xarray Dataset with all the variables of
        interest, which are_
            - u10 (winds direction), v10
            - wind_proj: magnitud of the projected winds
            - bearings for all the lat/lon points (using great_circle_calculator)
            - direc_proj_math: mathematical representation of the bearings
    """

    # we first join uw and vw (change winds to where they go)
    wind = xr.merge([-uw,-vw]).dropna(dim='time')
    # calculate directions of winds (where they go)
    wind_direcs = np.arctan2(wind[uw.name].values,
                             wind[vw.name].values) * 180/np.pi
    wind_direcs = np.where(wind_direcs<0,wind_direcs+360,wind_direcs)
    # calculate bearing for each lat/lon
    bearings = calculate_bearings(wind.latitude.values,
                                  wind.longitude.values,
                                  location=location)
    # calculate relative directions
    rel_direcs = bearings.reshape(1,*bearings.shape) - wind_direcs
    rel_direcs = np.where(rel_direcs<-180,rel_direcs+360,rel_direcs)
    rel_direcs = np.where(rel_direcs>180,rel_direcs-360,rel_direcs)
    # delete winds in opposite directions
    rel_direcs = np.where((rel_direcs>90)|(rel_direcs<-90),np.nan,rel_direcs)

    return wind.assign({
        'wind_proj': (('time','latitude','longitude'),np.cos(rel_direcs*np.pi/180)),
        'bearings': (('latitude','longitude'),bearings),
        'direc_proj_math': (('latitude','longitude'),trans_geosdeg2mathdeg(bearings)*np.pi/180)
    }) # final dataset


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
