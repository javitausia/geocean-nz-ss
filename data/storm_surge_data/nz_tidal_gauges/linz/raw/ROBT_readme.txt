SITE INFORMATION
-----------------
NAME:             Cape Roberts
CODE:             ROBT
WGS-84 POSITION:   77° 02'S
                  163° 11'E
UTM COORDINATES:    604 700 mE
ZONE 57 (SOUTH)   1 445 800 mN	

===========================================================

PRESSURE SENSOR DETAILS
------------------------
DATA COMMENCED:   20 November 1990 (UTC), julian day 324
RECORD INTERVAL:  5 minutes
TIME SYSTEM:      UTC
MANUFACTURER:     Geokon 
SENSOR MODEL:     4500
RANGE:            350kPa
           
===========================================================

REFERENCE BENCH MARKS		
----------------------		  
CAPE ROBERTS TGBM1, LINZ geodetic code B93M  
ROB3, LINZ geodetic code ROB3 

===========================================================

DATA AVAILABILITY		
------------------		  
Contact LINZ (customer support@linz.govt.nz) for data prior
to 8 November 2007.  

************************************************
* DATA UNRELIABLE: 29 August - 7 December 2009 *
************************************************

===========================================================

SUMMARY OF TIDE GAUGE ZERO BELOW B93M (metres)
-----------------------------------------------

Calibration Date

November 2007      8.268 m
November 2008      8.245 m


November 2011      7.993 m
November 2012      8.022 m
November 2013      8.019 m
November 2014      7.997 m
November 2015      8.075 m
 
===========================================================

FILE NAME FORMAT NOTES
-----------------------

The sea level data .zip file is named using the format
ROBT_yyyyddd.zip, where:

yyyy is the year (eg 2009)
ddd is the day of the year (eg 110, for 20 April).

The .zip file contains a single file of sea level data, 
named in the same format but using a .csv file extension.

===========================================================

SEA LEVEL DATA FORMAT - comma delimited
----------------------------------------

Prior to 29 August 2009: 

1st field: site code (ROBT)
2nd field: date and time (yyyy-mm-dd hh:mm) UTC, 24 hour
3rd field: sea level height (metres)

eg:

ROBT,2008-01-17 00:05,6.047
ROBT,2008-01-17 00:10,6.004
ROBT,2008-01-17 00:15,5.992


Since 21 November 2011: 

1st field: site code (ROBT)
2nd field: date and time (yyyy-mm-dd hh:mm) UTC, 24 hour
3rd field: sea level height (metres)
4th field: number of measurements per 5 min sample (max 30)
5th field: sea water temperature °C
6th field: atmospheric pressure hPa 

eg:

ROBT,2014-01-25 02:30,5.821,30,-1.078,993.64
ROBT,2014-01-25 02:35,5.808,30,-1.101,993.48
ROBT,2014-01-25 02:40,5.832,28,-1.108,993.8

===========================================================