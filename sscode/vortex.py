# This code was created and is hosted at:
# https://gitlab.com/geoocean/bluemath/hybrid-models/hywaves/-/tree/14-generate-stop-motion-cases
# by Nicolás Ripoll and Sara Ortega

# import arrays
import numpy as np
import pandas as pd
import xarray as xr
# and math
from math import radians, degrees, sin, cos, sqrt, atan2, pi


def gc_distance(lat1, lon1, lat2, lon2):
    '''
    Calculates great circle distance and azimuth (exact parsed ml)
    '''

    # distance
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    a = sin((lat2-lat1)/2)**2 + cos(lat1) * cos(lat2) * sin((lon2-lon1)/2)**2;
    if a < 0: a = 0
    if a > 1: a = 1

    r = 1
    rng = r * 2 * atan2(sqrt(a), sqrt(1-a))
    rng = degrees(rng)

    # azimuth
    az = atan2(
        cos(lat2) * sin(lon2-lon1),
        cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(lon2-lon1)
    )
    
    if lat1 <= -pi/2: az = 0
    if lat2 >=  pi/2: az = 0
    if lat2 <= -pi/2: az = pi
    if lat1 >=  pi/2: az = pi

    az = az % (2*pi)
    az = degrees(az)

    return rng, az


def get_xy_labels(coords_mode):
    '''
    Returns labels for x,y axis. Function of swan coordinates mode
    '''

    if coords_mode == 'SPHERICAL':
        lab_x, lab_x_long = 'lon', 'Longitude (º)'
        lab_y, lab_y_long = 'lat', 'Latitude (º)'

    elif coords_mode == 'CARTESIAN':
        lab_x, lab_x_long = 'X', 'X (m)'
        lab_y, lab_y_long = 'Y', 'Y (m)'

    return lab_x, lab_x_long, lab_y, lab_y_long


def geo_distance_meters(y_matrix, x_matrix, y_point, x_point, coords_mode):
    '''
    Returns distance between matrix and point (in meters)
    '''

    RE = 6378.135 * 1000 # Earth radius [m]

    if coords_mode == 'SPHERICAL':
        arcl, _ = geo_distance_azimuth(y_matrix, x_matrix, y_point, x_point)
        r = arcl * np.pi / 180.0 * RE  # to meteres

    if coords_mode == 'CARTESIAN':
        r = geo_distance_cartesian(y_matrix, x_matrix, y_point, x_point)

    return r


def geo_distance_azimuth(lat_matrix, lon_matrix, lat_point, lon_point):
    '''
    Returns geodesic distance and azimuth between lat,lon matrix and lat,lon
    point in degrees
    '''

    arcl = np.zeros(lat_matrix.shape) * np.nan
    azi = np.zeros(lat_matrix.shape) * np.nan

    sh1, sh2 = lat_matrix.shape

    for i in range(sh1):
        for j in range(sh2):
            arcl[i,j], azi[i,j] = gc_distance(
                lat_point, lon_point, lat_matrix[i][j], lon_matrix[i][j]
            )

    return arcl, azi


def geo_distance_cartesian(y_matrix, x_matrix, y_point, x_point):
    '''
    Returns cartesian distance between y,x matrix and y,x point
    '''
    
    dist = np.zeros(y_matrix.shape) * np.nan

    sh1, sh2 = y_matrix.shape

    for i in range(sh1):
        for j in range(sh2):
            dist[i,j] = sqrt(
                (y_point - y_matrix[i][j])**2 + (x_point - x_matrix[i][j])**2
            )

    return dist


def get_category(ycpres):
    '''
    Defines storm category according to minimum pressure center
    '''

    categ = []
    for i in range(len(ycpres)):
        if (ycpres[i] == 0) or (np.isnan(ycpres[i])):
            categ.append(6)
        elif ycpres[i] < 920:  categ.append(5)
        elif ycpres[i] < 944:  categ.append(4)
        elif ycpres[i] < 964:  categ.append(3)
        elif ycpres[i] < 979:  categ.append(2)
        elif ycpres[i] < 1000: categ.append(1)
        elif ycpres[i] >= 1000: categ.append(0)

    return categ


def ibtrac_basin_fitting(x0, y0):
    '''
    Assigns cubic polynomial fitting curve coefficient for each basin of
    historical TCs data (IBTrACS)
    '''

    # determination of the location basin 
    if y0 < 0:                  basin = 5
    elif (y0 > 0) & (x0 > 0):   basin = 3
    else:                       print('Basin not defined')

    # cubic polynomial fitting curve for Ibtracs and each basin
    # TODO: obtain all basin fitting coefficients

    if basin == 3:      # West Pacific
        p1 = -7.77328602747578e-06
        p2 = 0.0190830514629838
        p3 = -15.9630945598490
        p4 = 4687.76462404360

    elif basin == 5:    # South Pacific
        p1 = -4.70481986864773e-05
        p2 = 0.131052968357409
        p3 = -122.487981649828
        p4 = 38509.7575283218

    return p1, p2, p3, p4


def shoot(lon, lat, azimuth, maxdist=None):
    '''
    Shooter Function
    Original javascript on http://williams.best.vwh.net/gccalc.htm
    Translated to python by Thomas Lecocq
    '''

    glat1 = lat * np.pi / 180.
    glon1 = lon * np.pi / 180.
    s = maxdist / 1.852         # from km to n mi
    faz = azimuth * np.pi / 180.

    EPS= 0.00000000005
    if ((np.abs(np.cos(glat1))<EPS) and not (np.abs(np.sin(faz))<EPS)):
        print("Only N-S courses are meaningful, starting at a pole!")

    a=6378.13/1.852
    f=1/298.257223563
    r = 1 - f
    tu = r * np.tan(glat1)
    sf = np.sin(faz)
    cf = np.cos(faz)
    if (cf==0):
        b=0.
    else:
        b=2. * np.arctan2 (tu, cf)

    cu = 1. / np.sqrt(1 + tu * tu)
    su = tu * cu
    sa = cu * sf
    c2a = 1 - sa * sa
    x = 1. + np.sqrt(1. + c2a * (1. / (r * r) - 1.))
    x = (x - 2.) / x
    c = 1. - x
    c = (x * x / 4. + 1.) / c
    d = (0.375 * x * x - 1.) * x
    tu = s / (r * a * c)
    y = tu
    c = y + 1
    while (np.abs (y - c) > EPS):

        sy = np.sin(y)
        cy = np.cos(y)
        cz = np.cos(b + y)
        e = 2. * cz * cz - 1.
        c = y
        x = e * cy
        y = e + e - 1.
        y = (((sy * sy * 4. - 3.) * y * cz * d / 6. + x) *
              d / 4. - cz) * sy * d + tu

    b = cu * cy * cf - su * sy
    c = r * np.sqrt(sa * sa + b * b)
    d = su * cy + cu * sy * cf
    glat2 = (np.arctan2(d, c) + np.pi) % (2*np.pi) - np.pi
    c = cu * cy - su * sy * cf
    x = np.arctan2(sy * sf, c)
    c = ((-3. * c2a + 4.) * f + 4.) * c2a * f / 16.
    d = ((e * cy * c + cz) * sy * c + y) * sa
    glon2 = ((glon1 + x - (1. - c) * d * f + np.pi) % (2*np.pi)) - np.pi

    baz = (np.arctan2(sa, b) + np.pi) % (2 * np.pi)

    glon2 *= 180./np.pi
    glat2 *= 180./np.pi
    baz *= 180./np.pi

    return (glon2, glat2, baz)


def preprocess_track(track, x0, y0,
    track_vars={
        'longitude':'lon','latitude':'lat',
        'pressure':'mslp','time':'etc_datetime',
        'maxwinds':'vel850'
    }, region_limits=[140,240,-10,-80],
    time_resolution='1H',
    interpolation=True,
    interpolation_step='20MIN',
    interpolation_mode='mean', # 'first' available
    great_circle=True, 
    winds_fit=True
    ):

    '''
    Edits initial track dataframe to work with vortex model
    '''

    # convert data to bar
    track[track_vars['pressure']] = track[track_vars['pressure']] \
        if track[track_vars['pressure']].mean()<2000 else track[track_vars['pressure']]/100

    # round dataframe to desired resolution and set index
    track = track.set_index(
        track[track_vars['time']].round(time_resolution)
    )

    # get category of tc given the pressure
    track['category'] = get_category(
        track[track_vars['pressure']]
    )

    # longitude convention: [0º,360º]
    track[track_vars['longitude']+'_360'] = np.where(
        track[track_vars['longitude']]>0, track[track_vars['longitude']],
        track[track_vars['longitude']]+360 # sum 360 to negative longitudes
    )

    # storm variables timestep [hours]
    ts = track.index[1:] - track.index[:-1]
    ts = [tsi.total_seconds() / 3600 for tsi in ts]
    
    # calculate Vmean
    RE = 6378.135 # Earth radius [km]
    vmean = []
    for i in range(len(track.index)-1):
        # consecutive storm coordinates
        lon1, lon2 = track[track_vars['longitude']+'_360'][i], track[track_vars['longitude']+'_360'][i+1]
        lat1, lat2 = track[track_vars['latitude']][i], track[track_vars['latitude']][i+1]
        # translation speed
        arcl_h, gamma_h = gc_distance(lat2, lon2, lat1, lon1)
        r = arcl_h * np.pi / 180.0 * RE     # distance between consecutive coordinates (km)
        vmean.append(r / ts[i] / 1.852)     # translation speed (km/h to kt) 
        
    # mean value
    vmean = np.mean(vmean)  # [kt]

    # cubic polynomial fitting coefficients for IBTrACS basins Pmin-Vmax relationship
    if winds_fit:
        p1, p2, p3, p4 = ibtrac_basin_fitting(x0, y0)
        track[track_vars['maxwinds']+'_fit'] = np.where(
            track[track_vars['maxwinds']]!=0, track[track_vars['maxwinds']],
            p1 * np.power(track[track_vars['pressure']],3) \
                + p2 * np.power(track[track_vars['pressure']],2) \
                + p3 * np.power(track[track_vars['pressure']],1) + p4
        )

    # number of time steps between consecutive interpolated storm coordinates
    if time_resolution[-1]=='H':
        interpolation_steps = int(time_resolution[:-1])*60 / int(interpolation_step[:-3])
    else:
        interpolation_steps = int(time_resolution[:-3]) / int(interpolation_step[:-3])

    # initialize new variables
    move, vmean, pn, p0, lon, lat, vmax = [], [], [], [], [], [], []
    vu, vy = [], []
    interpolated_times = []

    for i, interp_time in enumerate(track.index[:-1]):
        # create new times to save interpolation
        interp_times = pd.date_range(
            start=interp_time, periods=interpolation_steps, freq=interpolation_step
        )
        for interp_time_i in interp_times:
            interpolated_times.append(interp_time_i)
        # consecutive storm coordinates
        lon1, lon2 = track[track_vars['longitude']+'_360'][i], track[track_vars['longitude']+'_360'][i+1]
        lat1, lat2 = track[track_vars['latitude']][i], track[track_vars['latitude']][i+1]
        # translation speed 
        arcl_h, gamma_h = gc_distance(lat2, lon2, lat1, lon1)
        r = arcl_h * np.pi / 180.0 * RE     # distance between consecutive storm coordinates [km]
        dx = r / ts[i]                      # distance during time step
        vx = float(dx) / ts[i] / 3.6        # translation speed [km to m/s]
        vx = vx / 0.52                      # translation speed [m/s to kt]

        # loop over times to interpolate for interp_time and interp_time+1
        for j, interp_time_j in enumerate(interp_times):
            # append storm track parameters
            move.append(gamma_h)
            vmean.append(vx)
            vu.append(vx * np.sin((gamma_h+180)*np.pi/180))
            vy.append(vx * np.cos((gamma_h+180)*np.pi/180))
            pn.append(1013)
            # append pmin, wind with/without interpolation along the storm track
            if not interpolation:       
                p0.append(track[track_vars['pressure']][i] + \
                    j*(track[track_vars['pressure']][i+1]-track[track_vars['pressure']][i]) / ts[i]
                )
            else:   
                if interpolation_mode=='mean':    
                    p0.append(np.mean((
                        track[track_vars['pressure']][i], track[track_vars['pressure']][i+1]
                    )))
                elif interpolation_mode=='first':
                    p0.append(track[track_vars['pressure']][i])
            if winds_fit:
                if not interpolation:   
                    vmax.append(track[track_vars['maxwinds']][i] + \
                        j*(track[track_vars['maxwinds']][i+1]-track[track_vars['maxwinds']][i]) / ts[i])
                else:
                    if interpolation_mode=='mean': 
                        vmax.append(np.mean((
                            track[track_vars['maxwinds']][i], track[track_vars['maxwinds']][i+1]
                        )))
                    elif interpolation_mode=='first':
                        vmax.append(track[track_vars['maxwinds']][i])
            # calculate timestep lon, lat
            if not great_circle:
                lon_h = lon1 - (dx*180/(RE*np.pi)) * np.sin(gamma_h*np.pi/180) * j
                lat_h = lat1 - (dx*180/(RE*np.pi)) * np.cos(gamma_h*np.pi/180) * j
            else:
                lon_h, lat_h, baz = shoot(lon1, lat1, gamma_h + 180, float(dx) * j)
            lon.append(lon_h)
            lat.append(lat_h)

    # to array
    move = np.array(move)
    vmean = np.array(vmean)
    vu = np.array(vu)
    vy = np.array(vy)
    p0 = np.array(p0)
    vmax = np.array(vmax)
    lon = np.array(lon)
    lon = np.where(lon>0,lon,lon+360)
    lat = np.array(lat)

    # select interpolation coordinates within the target domain area
    loc = []
    for i, (lo,la) in enumerate(zip(lon, lat)):
        if (lo<=region_limits[1]) & (lo>=region_limits[0]) & \
            (la<=region_limits[2]) & (la>=region_limits[3]):
            # append lons and lats in region limitis
            loc.append(i)

    # storm track (pd.DataFrame)
    st = pd.DataFrame(
        data={
            'move': move[loc],      # gamma, forward direction
            'vf': vmean[loc],     # translational speed [kt]
            'vfx': vu[loc],         # x-component
            'vfy': vy[loc],         # y-component
            'pn': 1013,             # average pressure at the surface [mbar]
            'p0': p0[loc],          # minimum central pressure [mbar]
            'lon': lon[loc],      # longitude coordinate
            'lat': lat[loc]      # latitude coordinate
        }, index=np.array(interpolated_times)[loc]
    )
    
    # maximum wind speed (if no value is given it is assigned the empirical Pmin-Vmax basin-fitting)
    if winds_fit:  
        st['vmax'] = vmax[loc]  # [kt]
    else:                   
        st['vmax'] = p1 * np.power(p0[loc],3) + p2 * np.power(p0[loc],2) + \
            p3 * np.power(p0[loc],1) + p4   # [kt]

    # add some metadata
    # TODO: move to st.attrs (this metada gets lost with any operation with st)
    st.x0 = x0
    st.y0 = y0
    st.R = 4

    return st, np.array(interpolated_times)[loc]


def vortex_model(storm_track, coords_mode='SPHERICAL', 
                 cg_lon_attrs=[140,180,20],
                 cg_lat_attrs=[-80,-20,20]):
    '''
    Uses winds vortex model to generate wind fields from storm track parameters

    Wind model code (from ADCIRC, transcribed by Antonio Espejo) and
    later slightly modified by Sara Ortega to include TCs at southern

    storm_track - (pandas.DataFrame)
    - obligatory fields:
        vfx
        vfy
        p0
        pn
        index
        vmax

    - optional fields:
        rmw  (optional)

    - for SPHERICAL coordinates
        lon / x
        lat / y

    - for CARTESIAN coordiantes:
        x
        y
        latitude

    swan_mesh.computational_grid
        mxc
        myc
        xpc
        ypc
        xlenc
        ylenc

    coords_mode - 'SHPERICAL' / 'CARTESIAN' swan project coordinates mode
    '''

    # parameters
    RE = 6378.135 * 1000            # Earth radius [m]
    beta = 0.9                      # conversion factor of wind speed
    rho_air = 1.15                  # air density
    w = 2 * np.pi / 86184.2         # Earth's rotation velocity (rad/s)
    pifac = np.arccos(-1) / 180     # pi/180
    one2ten = 0.8928                # conversion from 1-min to 10-min

    # wind variables
    storm_vfx  = storm_track.vfx.values[:]
    storm_vfy  = storm_track.vfy.values[:]
    storm_p0   = storm_track.p0.values[:]
    storm_pn   = storm_track.pn.values[:]
    times      = storm_track.index[:]
    storm_vmax = storm_track.vmax.values[:]

    # optional wind variables
    if 'rmw' in storm_track:
        storm_rmw  = storm_track.rmw.values[:]
    else:
        storm_rmw = [None] * len(storm_vfx)

    # coordinate system dependant variables 
    if coords_mode == 'SPHERICAL':
        storm_x  = storm_track.lon.values[:]
        storm_y  = storm_track.lat.values[:]
        storm_lat  = storm_track.lat.values[:]

    if coords_mode == 'CARTESIAN':
        storm_x    = storm_track.x.values[:]
        storm_y    = storm_track.y.values[:]
        storm_lat  = storm_track.latitude.values[:]

    # Correction when track is in south hemisphere for vortex generation 
    south_hemisphere = any (i < 0 for i in storm_lat)

    # swan mesh: computational grid (for generating vortex wind)
    # mxc  = swan_mesh.cg['mxc']
    # myc  = swan_mesh.cg['myc']
    # xpc = swan_mesh.cg['xpc']
    # ypc = swan_mesh.cg['ypc']
    # xpc_xlenc = swan_mesh.cg['xpc'] + swan_mesh.cg['xlenc']
    # ypc_ylenc = swan_mesh.cg['ypc'] + swan_mesh.cg['ylenc']

    # prepare meshgrid
    cg_lon = np.linspace(
        cg_lon_attrs[0],cg_lon_attrs[1],cg_lon_attrs[2]
    )
    cg_lat = np.linspace(
        cg_lat_attrs[0],cg_lat_attrs[1],cg_lat_attrs[2]
    )
    mg_lon, mg_lat = np.meshgrid(cg_lon, cg_lat)

    # wind and slp output holder
    hld_W = np.zeros((len(cg_lat), len(cg_lon), len(storm_p0)))
    hld_D = np.zeros((len(cg_lat), len(cg_lon), len(storm_p0)))
    hld_slp = np.zeros((len(cg_lat), len(cg_lon), len(storm_p0)))

    # each time needs 2D (mesh) wind files (U,V)
    for c, (lo, la, la_orig, p0, pn, ut, vt, vmax, rmw) in enumerate(zip(
        storm_x, storm_y, storm_lat, storm_p0, storm_pn,
        storm_vfx, storm_vfy, storm_vmax, storm_rmw)):

        # generate vortex field when storm is given
        if all (np.isnan(i) for i in (lo, la, la_orig, p0, pn, ut, vt, vmax)) == False:

            # get distance and angle between points 
            r = geo_distance_meters(mg_lat, mg_lon, la, lo, coords_mode)

            # angle correction for southern hemisphere
            if south_hemisphere:
                thet = np.arctan2((mg_lat-la)*pifac, -(mg_lon-lo)*pifac)
            else:
                thet = np.arctan2((mg_lat-la)*pifac, (mg_lon-lo)*pifac)

            # ADCIRC model 
            CPD = (pn - p0) * 100    # central pressure deficit [Pa]
            if CPD < 100: CPD = 100  # limit central pressure deficit

            # Wind model 
            f = 2 * w * np.sin(abs(la_orig)*np.pi/180)  # Coriolis

            # Substract the translational storm speed from the observed maximum 
            # wind speed to avoid distortion in the Holland curve fit. 
            # The translational speed will be added back later
            vkt = vmax - np.sqrt(np.power(ut,2) + np.power(vt,2))  # [kt]

            # Convert wind speed from 10m altitude to wind speed at the top of 
            # the atmospheric boundary layer
            vgrad = vkt / beta  # [kt]
            v = vgrad
            vm = vgrad * 0.52  # [m/s]

            # TODO revisar 
            # optional rmw
            # if rmw == None:
            if True:

                # Knaff et al. (2016) - Radius of maximum wind (RMW)
                rm = 218.3784 - 1.2014*v + np.power(v/10.9844,2) - \
                        np.power(v/35.3052,3) - 145.509*np.cos(la_orig*pifac)  # nautical mile
                rm = rm * 1.852 * 1000   # from nautical mile to meters 

            else:
                rm = rmw

            rn = rm / r  # dimensionless

            # Holland B parameter with upper and lower limits
            B = rho_air * np.exp(1) * np.power(vm,2) / CPD
            if B > 2.5: B = 2.5
            elif B < 1: B = 1

            # Wind velocity at each node and time step   [m/s]
            vg = np.sqrt(np.power(rn,B) * np.exp(1-np.power(rn,B)) * \
                         np.power(vm,2) + np.power(r,2)*np.power(f,2)/4) - r*f/2

            # Determine translation speed that should be added to final storm  
            # wind speed. This is tapered to zero as the storm wind tapers to 
            # zero toward the eye of the storm and at long distances from the storm
            vtae = (abs(vg) / vgrad) * ut    # [m/s]
            vtan = (abs(vg) / vgrad) * vt

            # Find the velocity components and convert from wind at the top of the 
            # atmospheric boundary layer to wind at 10m elevation
            hemisphere_sign = 1 if south_hemisphere else -1
            ve = hemisphere_sign * vg * beta * np.sin(thet)  # [m/s]
            vn = vg * beta * np.cos(thet)

            # Convert from 1 minute averaged winds to 10 minute averaged winds
            ve = ve * one2ten    # [m/s]
            vn = vn * one2ten

            # Add the storm translation speed
            vfe = ve + vtae      # [m/s]
            vfn = vn + vtan

            # wind module
            W = np.sqrt(np.power(vfe,2) + np.power(vfn,2))  # [m/s]

            # Surface pressure field
            pr = p0 + ((pn-p0) * np.exp(- np.power(rn,B)))      # [mbar]
            py, px = np.gradient(pr)
            ang = np.arctan2(py, px) + np.sign(la_orig) * np.pi/2.0

            # hold wind data (m/s) and slp data
            hld_W[:,:,c] = W
            hld_D[:,:,c] =  270 - np.rad2deg(ang)  # direction (º clock. rel. north)
            hld_slp[:,:,c] = pr

        else:

            # hold wind data (m/s) and slp data
            hld_W[:,:,c] = 0
            hld_D[:,:,c] = 0  # direction (º clock. rel. north)
            hld_slp[:,:,c] = p0

    # spatial axis labels
    lab_x, lab_x_long, lab_y, lab_y_long = get_xy_labels(coords_mode)

    # generate vortex dataset 
    xds_vortex = xr.Dataset(
        {
            'W':   ((lab_y, lab_x, 'time'), hld_W, {'units':'m/s'}),
            'Dir': ((lab_y, lab_x, 'time'), hld_D, {'units':'º'}),
            'p0': ((lab_y, lab_x, 'time'), hld_slp, {'units':'mbar'})
        },
        coords={
            lab_y : cg_lat,
            lab_x : cg_lon,
            'time' : times,
        }
    )
    xds_vortex.attrs['xlabel'] = lab_x_long
    xds_vortex.attrs['ylabel'] = lab_y_long

    return xds_vortex

