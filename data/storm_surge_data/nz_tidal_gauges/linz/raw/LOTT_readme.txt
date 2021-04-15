SITE LOCATION NOTES
--------------------
NAME:             Lottin Point
CODE:             LOTT
WGS-84 POSITION:   37° 33'S
                  178° 10'E
NZTM COORDINATES: 2 055 850 mE
                  5 831 600 mN	
=====================================================================

REFERENCE BENCH MARKS		
----------------------		  
BM LP 1, LINZ geodetic code ECPK  
BM LP 2, LINZ geodetic code ECPH
BM LP 3, LINZ geodetic code ECPJ
=====================================================================

PRESSURE SENSOR DETAILS
------------------------
MANUFACTURER:     Druck 
SENSOR MODEL:     PTX 1830
RANGE:            0 - 20 metres (4-20mA output)
=====================================================================

SEA LEVEL DATA INFORMATION
---------------------------
DATA COMMENCED:   10 October 2008 (UTC), day 284
RECORD INTERVAL:  1 minute
TIME SYSTEM:      UTC
DATA ZERO:        4.59 metres below BM LP 1
=====================================================================

SUMMARY OF LEVELS
------------------

   BM LP 2  ---- 4.916
               |
   BM LP 1  ---- 4.591
   BM LP 3  ---- 4.464
               |
               |
               |
               |
SENSOR ZERO ---- 0.000
=====================================================================

FILE NAME FORMAT NOTES
-----------------------

The sea level data .zip file is named using the format
GGGG_yyyyddd.zip (eg LOTT_2008295.zip), where:

GGGG is the code for the monitoring site (LOTT, for Lottin Point)
yyyy is the year (eg 2008)
ddd is the day of the year (eg 295, for 21 October).

The .zip file contains a single file of sea level data, named in the 
same format but using a .csv file extension (eg LOTT_2008295.csv).
=====================================================================

SEA LEVEL DATA FORMAT - comma delimited
----------------------------------------

1st field: site code (LOTT)
2nd field: date and time (dd-mm-yyyy hh:mm) UTC, 24 hour
3rd field: sea level height (metres)

eg:

LOTT,21/10/2008 00:00,3.962
LOTT,21/10/2008 00:01,3.976
LOTT,21/10/2008 00:02,3.961
=====================================================================			  