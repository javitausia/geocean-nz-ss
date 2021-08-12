This folder contains extra-tropical cyclone tracks extracted from the ERA5 reanalysis mslp and temperature fileds using the software tempest extreme (https://github.com/ClimateGlobalChange/tempestextremes) and the downloaded IBTrACS (https://www.ncdc.noaa.gov/ibtracs/index.php?name=ib-v4-access) file, cropped to etc in New Zealand.

For the first data, the methodology followed is that described in Ullrich, P.A. and C.M. Zarzycki (2017) "TempestExtremes v1.0: A framework for scale-insensitive pointwise feature tracking on unstructured grids" Geosci. Model. Dev. 10, pp. 1069-1090, doi: 10.5194/gmd-10-1069-2017 with minor adjustments to reflect the fact that we used the hourly ERA5 data instead of 6 hourly data. The pythons scripts used to extract the tracks will soon be added to the storm surge reposotory.

The data is stored in the tarball era5_etc_tracks.tar.gz which contains one file per identified etc track. Files are in text format. Each line correspond to a point in the track with:
- columns 1 and 2 identifying the grid cell where the pressure minima was found.
- columns 3 and 4 respectively correspond to the longitude and latitude of the ETC center.
- column 5 is the mean sea level pressure at the minima
- column 6 is the velocity maximum at the 850 pressure level within 4 degrees of the mean sea level pressure minima
- column 7 is the altitude at the mean sea level pressure minima
- columns 8/9/10/11 are respectively the year/month/day/hour of the record
