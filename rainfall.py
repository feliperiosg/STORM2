import warnings

# https://stackoverflow.com/a/9134842/5885810   (supress warning by message)
warnings.filterwarnings('ignore', message='You will likely lose important '\
    'projection information when converting to a PROJ string from another format')
# WOS doesn't deal with "ecCodes"
warnings.filterwarnings('ignore', message='Failed to load cfgrib - most likely '\
    'there is a problem accessing the ecCodes library.')
# https://github.com/slundberg/shap/issues/2909    (suppresing the one from numba 0.59.0)
warnings.filterwarnings('ignore', message=".*The 'nopython' keyword.*")

# https://stackoverflow.com/a/248066/5885810
from os.path import abspath, dirname, join
parent_d = dirname(__file__)    # otherwise, will append the path.of.the.tests
# parent_d = './'               # to be used in IPython

import numpy as np
import pandas as pd
# https://stackoverflow.com/a/65562060/5885810  (ecCodes in WOS)
import xarray as xr
import pyproj as pp
import netCDF4 as nc4
import geopandas as gpd
from scipy import stats
from numpy import random as npr
from statsmodels.distributions.copula.api import GaussianCopula
#from statsmodels.distributions.empirical_distribution import ECDF
from scipy.interpolate import interp1d

from osgeo import gdal
# https://gdal.org/api/python_gotchas.html#gotchas-that-are-by-design-or-per-history
# https://github.com/OSGeo/gdal/blob/master/NEWS.md#ogr-370---overview-of-changes
if gdal.__version__.__getitem__(0) == '3':# enable exceptions for GDAL<=4.0
    gdal.UseExceptions()
    # gdal.DontUseExceptions()
    # gdal.__version__ # wos_ '3.6.2' # linux_ '3.7.0'

from pyproj import Transformer
from rasterio import fill
from pointpats.random import poisson
from tqdm import tqdm
from zoneinfo import ZoneInfo
from datetime import timedelta, timezone, datetime
from dateutil.tz import tzlocal
from dateutil.relativedelta import relativedelta

from chunking import CHUNK_3D
from parameters import *

""" only necessary if Z_CUTS & SIMULATION used """
from rasterstats import zonal_stats
""" STORM2.0 ALSO runs WITHOUT this library!!! """
import vonMisesMixtures as vonmises


#%% GREETINGS

"""
STORM [STOchastic Rainfall Model] produces realistic regional or watershed rainfall under various
climate scenarios based on empirical-stochastic selection of historical rainfall characteristics.

Based on Singer, M. B., and Michaelides, K. (2017), Deciphering the expression of climate change
within the Lower Colorado River basin by stochastic simulation of convective rainfall.
[ https://doi.org/10.1088/1748-9326/aa8e50 ]

version name: STORM2

Authors:
    Michael Singer 2017
    Manuel F. Rios Gaona 2022
Date created : 2015/06/
Last modified: 2023/03/13
"""


#%% INPUT PARAMETERS

"""
# STORM2.0 RUNS WITH THE PARAMETERS BELOW (PLACED HERE FOR 'ILUSTRATIVE' PURPOSES).
# THEIR TWEAKING SHOULD EXCLUSIVELY BE DONE IN THE FILE 'parameters.py'.
# 'parameters.py' ALSO OFFERS A MORE DETAILED EXPLANATION ON THEIR MEANING/VALUES.

MODE = 'SImuLAtiON'     # Type of Run (case-insensitive). Either 'SIMULATION' or 'VALIDATION'
# MODE = 'valiDaTION'     # Type of Run (case-insensitive). Either 'SIMULATION' or 'VALIDATION'
SEASONS = 1             # Number of Seasons (per Run)
NUMSIMS = 2             # Number of runs per Season
NUMSIMYRS = 3           # Number of years per run (per Season)

# # PARAMETER = [ S1 ]
PTOT_SC       = [0.00]
PTOT_SF       = [ 0.0]
STORMINESS_SC = [-0.0]
STORMINESS_SF = [+0.0]

# PARAMETER   = [ S1 ,  S2 ]
PTOT_SC       = [ 0. , - .0]
PTOT_SF       = [+0.0, -0. ]
STORMINESS_SC = [ 0.0, + .0]
STORMINESS_SF = [-0.0,  0.0]

# PRE_FILE = './model_input/ProbabilityDensityFunctions_ONE--ANALOG.csv'      # output from 'pre_processing.py'
# PRE_FILE = './model_input/ProbabilityDensityFunctions_ONE--ANALOG-pmf.csv'  # output from 'pre_processing.py'
PRE_FILE = './model_input/ProbabilityDensityFunctions_TWO--ANALOG-py.csv'   # output from 'pre_processing.py'
GAG_FILE = './model_input/data_WG/gage_data--gageNetworkWG--DIGITAL.csv'    # gage (meta-)data (optional*)
# GAG_FILE = None
SHP_FILE = './model_input/shp/WG_Boundary.shp'                      # catchment shape-file in WGS84
DEM_FILE = './model_input/dem/WGdem_wgs84.tif'                      # aoi raster-file (optional**)
# DEM_FILE = './model_input/dem/WGdem_26912.tif'                    # aoi raster-file in local CRS (***)
# DEM_FILE = None
OUT_PATH = './model_output'                                         # output folder

# Z_CUTS = None           # (or Z_CUTS = []) for INT-DUR copula modelling regardless altitude
Z_CUTS = [1350, 1500]   # in meters!
Z_STAT = 'median'       # statistic to retrieve from the DEM ['mean' or 'min'|'max'?? not 'count']

WGEPSG    = 26912       # EPSG Code of the local/regular Coordinate Reference System (CRS)
X_RES     = 1000        # in meters! (for the 'regular/local' CRS)
Y_RES     = 1000        # in meters! (for the 'regular/local' CRS)
BUFFER    = 5000        # in meters! -> buffer distance (out of the catchment)
CLOSE_DIS = 0.15        # in km -> small circle emulating the storm centre's point/dot
MINRADIUS =  max([X_RES, Y_RES]) /1e3
RINGS_DIS =  MINRADIUS *(2) +.1         # in km -> distance between (rainfall) rings; heavily dependant on X_Y_RES

MIN_DUR = 2             # in minutes!
MAX_DUR = 60*24*5       # in minutes! -> 5 days (in this case)
# # OR:
# MIN_DUR = []          # use 'void' arrays if you want NO.CONSTRAINT on storm-duration
# MAX_DUR = []          # ... in either (or both) MIN_/MAX_DUR parameters/constants

### these parameters allow to pin down a time-dimension to the storms
# SEED_YEAR  = None                         # for your SIM/VAL to start in the current year
SEED_YEAR    = 2023                         # for your SIM/VAL to start in 2050
### bear in mind the 'SEASONS' variable!... (when toying with 'SEASONS_MONTHS')
# SEASONS_MONTHS = [[6,10], None]             # JUNE through OCTOBER (just ONE season)
# # OR:
# SEASONS_MONTHS = [[10,5], ['jul','sep']]  # OCT[y0] through MAY[y1] (& JULY[y1] through SEP[y1])
SEASONS_MONTHS = [['may','sep'],[11, 1]]  # MAY through SEP (& e.g., NOV trhough DEC)
TIME_ZONE      = 'US/Arizona'               # Local Time Zone (see links below for more names)
# # OR:
# TIME_ZONE    = 'UTC'
# # https://stackoverflow.com/a/64861179/5885810    (zoneinfo list)
# # https://pynative.com/list-all-timezones-in-python/#h-get-list-of-all-timezones-name
# # https://www.timeanddate.com/time/map/
# # https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
DATE_ORIGIN    = '1950-01-01'               # to store dates as INT

### only touch this parameter if you really know what you're doing ;)
INTEGER = 2     # number of (unsigned) Bytes (2, 4, 6 or 8) to store the RAINFALL variable (into)
"""


#%% FUNCTIONS' DEFINITION

#~ replace FILE.PARAMETERS with those read from the command line ~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def PAR_UPDATE( args ):
    for x in list(vars( args ).keys()):
# https://stackoverflow.com/a/2083375/5885810   (exec global... weird)
        exec(f'globals()["{x}"] = args.{x}')
    # print([PTOT_SC, PTOT_SF])


#~ DEFINE THE DAYS OF THE SEASON (to 'sample' from) ~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def WET_SEASON_DAYS():
    global SEED_YEAR, M_LEN, DATE_POOL, DOY_POOL, DATE_ORIGIN
    SEED_YEAR = SEED_YEAR if SEED_YEAR else datetime.now().year
# which season is None/void/null
    mvoid = list(map(lambda x: None in x, zip(SEASONS_MONTHS)))
# transform months into numbers (if passed as strings)
    month = list(map( lambda m,v: None if v else\
        list(map(lambda m: m if type(m) == int else datetime.strptime(m,'%b').month, m )),
        SEASONS_MONTHS, mvoid ))
# compute monthly duration (12 months in a year)
    M_LEN = [None if v else \
        [1+m.__getitem__(1)-m.__getitem__(0) if m.__getitem__(1)-m.__getitem__(0)>=0 else \
             1+12+m.__getitem__(1)-m.__getitem__(0)] for m,v in zip(month, mvoid)]
# construct the date.times & update their years
    DATE_POOL = [None if v else \
        [datetime(year=SEED_YEAR,month=m[0],day=1),
          datetime(year=SEED_YEAR,month=m[0],day=1) + relativedelta(months=l[0])] \
            for m,l,v in zip(month, M_LEN, mvoid)]
    for i in range(len(DATE_POOL))[1:]:
        DATE_POOL[i] = None if mvoid[i] else \
            [DATE_POOL[i].__getitem__(0).replace(year=DATE_POOL[i-1].__getitem__(-1).year),
              DATE_POOL[i].__getitem__(0).replace(year=DATE_POOL[i-1].__getitem__(-1).year)\
                  + relativedelta(months=M_LEN[i].__getitem__(0))]
# extract Day(s)-Of-Year(s)
# https://stackoverflow.com/a/623312/5885810
    DOY_POOL = list(map(lambda v,d: None if v else
        list(map(lambda d: d.timetuple().tm_yday, d)), mvoid, DATE_POOL ))
# convert DATE_ORIGIN into 'datetime' (just to not let this line hanging out all alone)
# https://stackoverflow.com/q/70460247/5885810  (timezone no pytz)
# https://stackoverflow.com/a/65319240/5885810  (replace timezone)
    DATE_ORIGIN = datetime.strptime(DATE_ORIGIN, '%Y-%m-%d').replace(
        tzinfo=ZoneInfo(TIME_ZONE))


#~ WORKING THE CATCHMENT (& ITS BUFFER) MASK(S) OUT ~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def SHP_OPS():
    global llim, rlim, blim, tlim, BUFFRX, CATCHMENT_MASK, XS, YS
# read WG-catchment gauge.data (if necessary)
    if MODE.lower() == 'validation':
        gagnet = pd.read_csv(GAG_FILE, sep=',', header='infer', comment='#')
    # just verify that your own gage.network has 'gage_id', X, Y, (Z optional)
        gagnet = gagnet.loc[(gagnet['within_c']==1) & (gagnet['with_dat']==1)]
    # put it into GeoPandas
        gagnet = gpd.GeoDataFrame(gagnet.gage, geometry=gpd.points_from_xy(
            gagnet.X, gagnet.Y, gagnet.Z, crs=f'EPSG:{WGEPSG}'))
# read WG-catchment shapefile (assumed to be in WGS84)
    wtrwgs = gpd.read_file( SHP_FILE )
# transform it into EPSG:26912 & make the buffer
# https://gis.stackexchange.com/a/328276/127894 (geo series into gpd)
    wtrshd = wtrwgs.to_crs( epsg=WGEPSG )
    BUFFRX = gpd.GeoDataFrame(geometry=wtrshd.buffer( BUFFER ))#.to_crs(epsg=4326)
    # # OR
    # from osgeo import ogr
    # # https://gis.stackexchange.com/a/113808/127894
    # BUFFRX = ogr.Open("./model_input/Bndry_buffer.shp").GetLayer(0).GetFeature(0)     # SHP's 1st-feature
    # BUFFRX.ExportToJson()
    # # update accordingly if used in 'gdal.Rasterize('
    # # https://gis.stackexchange.com/a/359025/127894
    # BUFFRX.geometry.xs(0).minimum_rotated_rectangle.boundary

# infering (and rounding) the limits of the buffer-zone
    llim = np.floor( BUFFRX.bounds.minx[0] /X_RES ) *X_RES #+X_RES/2
    rlim = np.ceil(  BUFFRX.bounds.maxx[0] /X_RES ) *X_RES #-X_RES/2
    blim = np.floor( BUFFRX.bounds.miny[0] /Y_RES ) *Y_RES #+Y_RES/2
    tlim = np.ceil(  BUFFRX.bounds.maxy[0] /Y_RES ) *Y_RES #-Y_RES/2
    # llim = np.floor( BUFFRX.bounds.minx /X_RES ) *X_RES #+X_RES/2
    # rlim = np.ceil(  BUFFRX.bounds.maxx /X_RES ) *X_RES #-X_RES/2
    # blim = np.floor( BUFFRX.bounds.miny /Y_RES ) *Y_RES #+Y_RES/2
    # tlim = np.ceil(  BUFFRX.bounds.maxy /Y_RES ) *Y_RES #-Y_RES/2

# # BURN A SHP INTO RASTER & VISUALIZE IT
#     tmp_file = 'tmp-raster.tif'
#     tmp = gdal.Rasterize(tmp_file, BUFFRX.to_json(), xRes=X_RES, yRes=Y_RES,
#         allTouched=True, burnValues=1, noData=0, outputType=gdal.GDT_Int16,
#         targetAlignedPixels=True, outputBounds=[llim, blim, rlim, tlim],
#         outputSRS=f'EPSG:{WGEPSG}', format='GTiff')
#     var = tmp.ReadAsArray()
#     tmp = None

#     import matplotlib.pyplot as plt
#     plt.imshow(var, interpolation='none')
#     plt.show()
#     # OR
#     from rasterio.plot import show
#     tmp_file = 'tmp-raster.tif'
#     srcras = rasterio.open(tmp_file)
#     fig, ax = plt.subplots()
#     ax = rasterio.plot.show(srcras, ax=ax, cmap='viridis', extent=[
#         srcras.bounds[0], srcras.bounds[2], srcras.bounds[1], srcras.bounds[3]])
#     srcras.close()

# BURN THE CATCHMENT SHP INTO RASTER (WITH CATCHMENT-BUFFER EXTENSION)
# https://stackoverflow.com/a/47551616/5885810  (idx polygons intersect)
# https://gdal.org/programs/gdal_rasterize.html
    tmp = gdal.Rasterize(''
        , gagnet.to_json() if MODE.lower() == 'validation' else wtrshd.to_json()
        , xRes=X_RES, yRes=Y_RES, allTouched=True, noData=0, burnValues=1
        , add=(1 if MODE.lower() == 'validation' else 0)
        , outputType=gdal.GDT_Int16, targetAlignedPixels=True, format='MEM'
        , outputBounds=[llim, blim, rlim, tlim], outputSRS=f'EPSG:{WGEPSG}'
        # , width=(abs(rlim-llim)/X_RES).astype('u2'), height=(abs(tlim-blim)/X_RES).astype('u2')
        )
    CATCHMENT_MASK = tmp.ReadAsArray()
    tmp = None           # flushing!
    # plt.imshow(CATCHMENT_MASK, interpolation='none', aspect='equal', origin='upper',
    #     cmap='nipy_spectral_r', extent=(llim[0], rlim[0], blim[0], tlim[0]))
    # plt.show()

    # pd.DataFrame(CATCHMENT_MASK).to_csv('rainfall_catch.csv',sep=' ',header=False,index=False)
    # #CATCHMENT_MASK.sum() == len(gagnet)
    # pd.DataFrame(CATCHMENT_MASK).to_csv('rainfall_gatch.csv',sep=' ',header=False,index=False)

    # # https://gis.stackexchange.com/q/344942/127894   (flipped raster)
    # ds = gdal.Open('tmp-raster.tif')
    # gt = ds.GetGeoTransform()
    # if gt[2] != 0.0 or gt[4] != 0.0: print ('file is not stored with north up')

# DEFINE THE COORDINATES OF THE XY.AXES
    # XS, YS = list(map( lambda a,b,c: np.arange(a.item() +c/2, b.item() +c/2, c),
    #                   [llim,blim],[rlim,tlim],[X_RES,Y_RES] ))
    XS, YS = list(map( lambda a,b,c: np.arange(a +c/2, b +c/2, c),
                      [llim,blim],[rlim,tlim],[X_RES,Y_RES] ))
# flip YS??
    YS = np.flipud( YS )      # -> important...so rasters are compatible with numpys


#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#- CONSTRUCT THE PDFs (TO SAMPLE FROM) ------------------------------- (START) #
#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#~ READ THE CSV.FILE(s) PRODUCED BY THE preprocessing.py SCRIPT ~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def READ_PDF_PAR():
    global PDFS
# read PDF-parameters
# https://stackoverflow.com/a/58227453/5885810  (import tricky CSV)
    PDFS = pd.read_fwf(PRE_FILE, header=None)
    PDFS = PDFS.__getitem__(0).str.split(',', expand=True).set_index(0).astype('f8')


#~ CONSTRUCT PDFs FROM PARAMETERS (stored in 'PDFS') ~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def RETRIEVE_PDF( TAG ):
# TAG: core label/index (in PDFS variable) to construct the pdf on
    subset = PDFS[PDFS.index.str.contains( TAG )].dropna(how='all', axis='columns')

# necessary block as tags COPULA_RHO, MAXINT_PDF, AVGDUR_PDF might have Z-bands
    line = subset.index.str.contains(pat='[Z]\d{1,2}(?!\d)|100')
# https://stackoverflow.com/a/6400969/5885810   # regex for 1-100
# ...in case somebody goes crazy having up to 100 Z-bands!! ('[Z][1-9]' -> otherwise)
    if line.any() == True and Z_CUTS:
        # print('correct!')
        subset = subset[ line ]
        name = np.unique( list(zip( *subset.index.str.split('+') )).__getitem__(1) )
    else:
        subset = subset[ ~line ]
        name = ['']                 # makes "distros" 'universal'

# https://www.geeksforgeeks.org/python-get-first-element-of-each-sublist/
    first = list(list(zip( *subset.index.str.split('+') )).__getitem__(0))
# https://stackoverflow.com/a/6979121/5885810   (numpy argsort equivalent)
# https://stackoverflow.com/a/5252867/5885810
# https://stackoverflow.com/a/46453340/5885810  (difference between strings)
    sort_id = np.unique( list(map( lambda x: x.replace(TAG, ''), first )) )
# the line below makes 1st-PDFs be chosen by default
    sort_id = sort_id[ np.argsort( sort_id.astype('int') )  ]
# # TIP: USE THE LINE BELOW (REPLACING THE LINE ABOVE) IF YOU PREFER 2nd-ids-PDF INSTEAD
# # https://stackoverflow.com/a/16486305/5885810
#     sort_id = sort_id[ np.argsort( sort_id.astype('int') )[::-1]  ]
    group = [subset[subset.index.str.contains( f'{TAG}{i}' )].dropna(
        how='all', axis='columns') for i in sort_id]

    if TAG == 'DATIME_VMF' or TAG == 'DOYEAR_VMF':
# https://cmdlinetips.com/2018/01/5-examples-using-dict-comprehension/
# https://blog.finxter.com/how-to-create-a-dictionary-from-two-numpy-arrays/
        distros = [{A:B for A, B in zip(['p','mus','kappas'],
            [i.to_numpy() for item, i in G.T.iterrows()])} for G in group]
    elif TAG == 'COPULA_RHO':
        distros = [{A:B for A, B in zip(name if Z_CUTS else name,\
            [i.values.ravel().__getitem__(0) for item, i in G.iterrows()])} for G in group]
    else:
        distros = [{A:B for A, B in zip(name,
            [eval(f"stats.{item.split('+').__getitem__(-1)}"\
                  # f"({','.join( i.astype('str').values.ravel() )})")\
                  f"({','.join( i.dropna().astype('str').values.ravel() )})")\
                 for item, i in G.iterrows()] )} for G in group]

    return distros


#~ RETRIEVE THE PDFs & EVALUATE THEIR 'CONSISTENCY' AGAINST #SEASONS ~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def CHECK_PDF():
# https://stackoverflow.com/a/10852003/5885810
# https://stackoverflow.com/q/423379/5885810    (global variables)
    global DATIME, DOYEAR, COPULA, TOTALP, RADIUS, BETPAR, MAXINT, AVGDUR#, Z_CUTS

    try:
        DATIME = RETRIEVE_PDF( 'DATIME_VMF' )
    except IndexError:
        # DATIME = [stats.uniform() for x in range(SEASONS)]
        DATIME = [None for x in range(SEASONS)]
        warnings.warn(f'\nNo DATIME_VMF parameters were found in "{PRE_FILE}".'\
            '\nSTORM2.0 will proceed with TOD (Times Of Day) sampled from a '\
            'UNIFORM distribution. If this is not what you want, please '\
            'accordingly update the aforementioned file.', stacklevel=2)

    try:
        DOYEAR = RETRIEVE_PDF( 'DOYEAR_VMF' )
    except IndexError:
        DOYEAR = RETRIEVE_PDF( 'DOYEAR_PMF' )

    TOTALP = RETRIEVE_PDF( 'TOTALP_PDF' )
    RADIUS = RETRIEVE_PDF( 'RADIUS_PDF' )
    BETPAR = RETRIEVE_PDF( 'BETPAR_PDF' )
    MAXINT = RETRIEVE_PDF( 'MAXINT_PDF' )
    AVGDUR = RETRIEVE_PDF( 'AVGDUR_PDF' )
    COPULA = RETRIEVE_PDF( 'COPULA_RHO' )

# evaluate consistency between lists (lengths must be consistent with #SEASONS)
    test = ['DATIME', 'DOYEAR', 'COPULA', 'TOTALP', 'RADIUS', 'BETPAR', 'MAXINT', 'AVGDUR']
    lens = list(map(len, list(map( eval, test )) ))
# is there are variables having more PDFs than others
    assert len(np.unique(lens)) == 1, 'There are less (defined) PDFs for '\
        f'{" & ".join(np.asarray(test)[np.where(lens==np.unique(lens).__getitem__(0)).__getitem__(0)])} than for '\
        f'{" & ".join(np.delete(np.asarray(test),np.where(lens==np.unique(lens).__getitem__(0)).__getitem__(0)))}.'\
        f'\nPlease modify the file "{PRE_FILE}" (accordingly) to ensure that '\
        'for each of the aforementioned variables exists at least as many PDFs '\
        f'as the number of SEASONS to model ({SEASONS} seasons per year, '\
        'according to your input).'
# if there are more PDFs than the number of seasons (which is not wrong at all)
    if np.unique(lens) > SEASONS:
        warnings.warn(f'\nThe file "{PRE_FILE}" contains parameters for '\
            f'{np.unique(lens).__getitem__(0)} season(s) but you chose to model'\
            f' {SEASONS} season(s) per year.\nSTORM2.0 will proceed using '\
            "PDF-parameters for the season with the lowest 'ID-IndeX' (e.g., "\
            "'TOTALP_PDF1+...', 'RADIUS_PDF1+...', and so forth.)", stacklevel=2)
# if there are more number of seasons than PDFs (STORM duplicates PDFs)
    if np.unique(lens) < SEASONS:
# https://stackoverflow.com/a/45903502/5885810  (replicate list elements)
# https://stackoverflow.com/a/5599313/5885810   (using exec instead of eval)
        for x in test:
            exec( f'{x} = {x}*{SEASONS}' )
        warnings.warn(f'\nThe file "{PRE_FILE}" contains parameters for '\
            f'{np.unique(lens).__getitem__(0)} season(s) but you chose to model'\
            f' {SEASONS} season(s) per year.\nSTORM2.0 will proceed using these'\
            ' parameters for all seasons. If this is not what you want, please'\
            ' update the aforementioned file accordingly.', stacklevel=2)

#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#- CONSTRUCT THE PDFs (TO SAMPLE FROM) --------------------------------- (END) #
#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#- RANDOM SMAPLING --------------------------------------------------- (START) #
#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#~ N-RANDOM SAMPLES FROM 'ANY' GIVEN PDF ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def RANDOM_SAMPLING( PDF, N ):
# PDF: scipy distribution_infrastructure (constructed PDF)
# N  : number of (desired) random samples
    xample = PDF.rvs( size=N )
    # # for reproducibility
    # xample = PDF.rvs( size=N, random_state=npr.RandomState(npr.Philox(12345)) )
    # xample = PDF.rvs( size=N, random_state=npr.RandomState(npr.PCG64DXSM(1337)) )
    # xample = PDF.ppf( npr.RandomState(777).random( N ) ) # -> TOTALP for Matlab 'compatibility'
    return xample


#~ TRUNCATED N-RANDOM SAMPLES FROM 'ANY' GIVEN PDF ~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def TRUNCATED_SAMPLING( PDF, LIMITS, N ):
# LIMITS: PDF boundaries to constrain the sampling
# find the CDF for the limit (if any limit at all)
    LIMITS = [PDF.cdf(x) if x else None for x in LIMITS]
# if there are None, replace it by the lowest/highest possible CDF(s)
    if None in LIMITS:
# https://stackoverflow.com/a/50049044/5885810  (None to NaN to Zero)
        LIMITS = np.nan_to_num( np.array(LIMITS, dtype='f8') ) + np.r_[0, 1]
    xample = PDF.ppf( npr.uniform(LIMITS[0], LIMITS[-1], N) )
    # # for reproducibility
    # xample = PDF.ppf( npr.RandomState(npr.SFC64(54321)).uniform(LIMITS[0], LIMITS[-1], N) )   # -> RADIUS
    # xample = PDF.ppf( npr.RandomState(npr.PCG64(2001)).uniform(LIMITS[0], LIMITS[-1], N) )    # -> BETPAR
    # xample = PDF.ppf( npr.RandomState(555).uniform(LIMITS[0], LIMITS[-1], N) ) # -> RADIUS for Matlab 'compatibility'
    # xample = PDF.ppf( npr.RandomState(999).uniform(LIMITS[0], LIMITS[-1], N) ) # -> BETPAR for Matlab 'compatibility'
    return xample


#~ RETRIEVE TOTAL SEASONAL/MONSOONAL ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def SEASONAL_RAIN( PDF, seas, BAND='', N=1 ):
# sample N values of TOTALP & transform them from ln-space
    total = np.exp( RANDOM_SAMPLING( PDF[ seas ][ BAND ], N ) )
    return total


#~ SAMPLE FROM A COPULA & "CONDITIONAL" I_MAX-AVG_DUR PDFs ~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def COPULA_SAMPLING( COP, seas, BAND='', N=1 ):
# create the copula & sample from it
# https://stackoverflow.com/a/12575451/5885810  (1D numpy to 2D)
# (-1, 2) -> because 'GaussianCopula' will always give 2-cols (as in BI-variate copula)
# the 'reshape' allows for N=1 sampling
    IntDur = GaussianCopula(corr=COP[ seas ][ BAND ], k_dim=2).rvs( nobs=N ).reshape(-1, 2)
    # # for reproducibility
    # IntDur = GaussianCopula(corr=COP[ seas ][ BAND ], k_dim=2).rvs(
    #     nobs=N, random_state=npr.RandomState(npr.PCG64(20220608))).reshape(-1, 2)
    i_max = MAXINT[ seas ][ BAND ].ppf( IntDur[:, 0] )
    s_dur = AVGDUR[ seas ][ BAND ].ppf( IntDur[:, 1] )
    return i_max, s_dur


#~ REMOVE DUPLICATED TIME.STAMPS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def DUAL_STAMP( stamps ):
# finding and removing duplicated STAMPS (to avoid crashes from '.DROP_ISEL')
# https://stackoverflow.com/a/11528676/5885810
    repeat = np.setdiff1d(range(len(stamps)), np.unique(stamps, return_index=True).__getitem__(1))
    while len(repeat) != 0:
        stamps[repeat] = stamps[repeat] +1          # just add 1 second
        repeat = np.setdiff1d(range(len(stamps)), np.unique(stamps, return_index=True).__getitem__(1))
#!-WHAT HAPPENS WHE THE AGGREGATION EXCEEDS THE SEASON.LIMITS!??
    return stamps


#~ SAMPLE DAYS.OF.YEAR and TIMES.OF.DAY (CIRCULAR approach) ~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def TOD_CIRCULAR( N, seas, simy ):# N=NUM_S
    # M = N
    M = 0
    all_dates = []
    # while M>0:
    while M<N:
        doys = vonmises.tools.generate_mixtures( p=DOYEAR[ seas ]['p'],
            mus=DOYEAR[ seas ]['mus'], kappas=DOYEAR[ seas ]['kappas'], sample_size=N)
# to DOY
        doys = (doys +np.pi) /(2*np.pi) *365 -1
    # negatives are giving me troubles (difficult to discern if they belong to january/december)
        doys = doys[ doys>0 ]
        # # to check out if the sampling is done correctly
        # plt.hist(doys, bins=365)
# into actual dates
        dates = list(map(lambda d:
            datetime(year=DATE_POOL[ seas ][0].year,month=1,day=1) +\
            relativedelta(yearday=int( d )), doys.round(0) ))
        sates = pd.Series( dates )              # to pandas
# chopping into limits
        sates = sates[(sates>=DATE_POOL[ seas ][0]) & (sates<=DATE_POOL[ seas ][-1])]
        # M = len(dates) - len(sates)
        # print(M)
# updating to SIMY year (& storing)
        # all_dates.append( sates + pd.DateOffset(years=simy) )
        all_dates.append( sates.map(lambda d:d +relativedelta(years=simy)) )
        # # the line above DOES NOT give you errors when dealing with VOID arrays
        M = np.sum(list(map(len, all_dates)))
    all_dates = pd.concat( all_dates, ignore_index=True )
    # select random N-samples from an overflooded list
    all_dates = all_dates.sample(n=N, replace=False,)#random_state=1)

    """
If you're doing "CIRCULAR" for DOY that means you did install "vonMisesMixtures"
... therefore, sampling for TOD 'must' also be circular (why don't ya)
    """
# TIMES
# sampling from MIXTURE.of.VON_MISES-FISHER.distribution
    times = vonmises.tools.generate_mixtures(p=DATIME[ seas ]['p'],
        mus=DATIME[ seas ]['mus'], kappas=DATIME[ seas ]['kappas'], sample_size=N)
# from radians to decimal HH.HHHH
    times = (times +np.pi) /(2*np.pi) *24
    # # to check out if the sampling is done correctly
    # plt.hist(times, bins=24)
# SECONDS since DATE_ORIGIN
# https://stackoverflow.com/a/50062101/5885810
    stamps = list(map(lambda d,t:
        ((d + timedelta(hours=t)) - DATE_ORIGIN).total_seconds(),
        all_dates.dt.tz_localize( TIME_ZONE ), times))
# # pasting and formatting
# # https://stackoverflow.com/a/67105429/5885810  (chopping milliseconds)
#     stamps = list(map(lambda d,t: (d + timedelta(hours=t)).isoformat(timespec='seconds'), dates, times))
    stamps = np.round(stamps, 0).astype('u8')       # i root for .astype('u4') instead
    return DUAL_STAMP( stamps )
    # return DUAL_STAMP( np.sort( stamps ) )


#~ SAMPLE DAYS.OF.YEAR and TIMES.OF.DAY (DISCRETE approach) ~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def TOD_DISCRETE( N, seas, simy ):# N=NUM_S
    # M = N
    M = 0
    all_dates = []
    # while M>0:
    while M<N:
        soys = RANDOM_SAMPLING( DOYEAR[ seas ][''], N )
# chopping into limits
        doys = soys[(soys>=DATE_POOL[ seas ].__getitem__(0).timetuple().tm_yday) &\
                    (soys<=DATE_POOL[ seas ].__getitem__(1).timetuple().tm_yday)]
        # plt.hist(doys, bins=365)
# into actual dates
        dates = list(map(lambda d:
            datetime(year=DATE_POOL[ seas ].__getitem__(0).year,month=1,day=1) +\
            relativedelta(yearday=d), doys ))
        sates = pd.Series( dates )              # to pandas
        # M = len(soys) - len(sates)
        # print(M)
# updating to SIMY year (& storing)
        # all_dates.append( sates + pd.DateOffset(years=simy) )
        all_dates.append( sates.map(lambda d:d +relativedelta(years=simy)) )
        # # the line above DOES NOT give you errors when dealing with VOID arrays
        M = np.sum(list(map(len, all_dates)))
    all_dates = pd.concat( all_dates, ignore_index=True )
    # select random N-samples from an overflooded list
    all_dates = all_dates.sample(n=N, replace=False,)#random_state=1)

    """
If you're unlucky to be stuck with "DISCRETE"...
then there's no point in using circular on TOD, is it?'
    """
# TIMES
# sampling from a NORMAL distribution
    times = npr.uniform(0, 1, N) *24
    # # to check out if the sampling is done correctly
    # plt.hist(times, bins=24)
# SECONDS since DATE_ORIGIN
# https://stackoverflow.com/a/50062101/5885810
# https://stackoverflow.com/a/67105429/5885810  (chopping milliseconds)
    stamps = list(map(lambda d,t:
        ((d + timedelta(hours=t)) - DATE_ORIGIN).total_seconds(),
        all_dates.dt.tz_localize(TIME_ZONE), times))
# # pasting and formatting
# # https://stackoverflow.com/a/67105429/5885810  (chopping milliseconds)
#     stamps = list(map(lambda d,t: (d + timedelta(hours=t)).isoformat(timespec='seconds'),
#                       dates, times))
    stamps = np.round(stamps, 0).astype('u8')       # i root for .astype('u4') instead
    return DUAL_STAMP( stamps )
    # return DUAL_STAMP( np.sort( stamps ) )

#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#- RANDOM SMAPLING ----------------------------------------------------- (END) #
#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#- RASTER MANIPULATION ----------------------------------------------- (START) #
#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#~ CREATE AN OUTER RING/POLYGON ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# *1e3 to go from km to m
def LAST_RING( all_radii, CENTS ):# all_radii=RADII
# "resolution" is the number of segments in which a.quarter.of.a.circle is divided into.
# ...now it depends on the RADII/RES; the larger a circle is the more resolution it has.
    ring_last = list(map(lambda c,r: gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(x=[c[0]], y=[c[1]] ).buffer( r *1e3,
            # resolution=int((3 if r < 1 else 2)**np.ceil(r /2)) ),
            # resolution=np.ceil(r /MINRADIUS) +1 ), # or maybe... "+1"??
            resolution=np.ceil(r /MINRADIUS) +2 ), # or maybe... "+2"??
        crs=f'EPSG:{WGEPSG}'), CENTS, all_radii))
    return ring_last


#~ CREATE CIRCULAR SHPs (RINGS & CIRCLE) & ASSING RAINFALL TO C.RINGS ~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def LOTR( RADII, MAX_I, DUR_S, BETAS, CENTS ):
    all_radii = list(map(lambda r:
        np.r_[np.arange(r, CLOSE_DIS, -RINGS_DIS), CLOSE_DIS], RADII))

    all_rain = list(map(lambda i,d,b,r: list(map( lambda r:
    # # model: FORCE_BRUTE -> a * np.exp(-2 * b * x**2)
    #     i * d *1/60 * np.exp( -2* b * r**2 ), r)), MAX_I, DUR_S, BETAS, all_radii))
    # model: BRUTE_FORCE -> a * np.exp(-2 * b**2 * x**2)
        i * d *1/60 * np.exp( -2* b**2 * r**2 ), r)), MAX_I, DUR_S, BETAS, all_radii))

# BUFFER_STRINGS
# https://www.knowledgehut.com/blog/programming/python-map-list-comprehension
# https://stackoverflow.com/a/30061049/5885810  (map nest)
# r,p are lists (at first instance), and the numbers/atoms (in the second lambda)
# .boundary gives the LINESTRING element
# *1e3 to go from km to m
# np.ceil(r /MINRADIUS) +2 ) is an artifact to lower the resolution of small circles
# ...a lower resolution in such circles increases the script.speed in the rasterisation process.
    rain_ring = list(map(lambda c,r,p: pd.concat( list(map(lambda r,p: gpd.GeoDataFrame(
        {'rain':p, 'geometry':gpd.points_from_xy( x=[c[0]], y=[c[1]] ).buffer( r *1e3,
            # resolution=int((3 if r < 1 else 2)**np.ceil(r /2)) ).boundary},
            # resolution=np.ceil(r /MINRADIUS) +1 ).boundary}, # or maybe... "+1"??
            resolution=np.ceil(r /MINRADIUS) +2 ).boundary}, # or maybe... "+2"??
        crs=f'EPSG:{WGEPSG}') , r, p)) ), CENTS, all_radii, all_rain))
# # the above approach (in theory) is much? faster than the list.comprehension below
#     rain_ring = [pd.concat( gpd.GeoDataFrame({'rain':p, 'geometry':gpd.points_from_xy(
#         x=[c[0]], y=[c[1]] ).buffer(r *1e3, np.ceil(r /MINRADIUS) +2 ).boundary},
#         crs=f'EPSG:{WGEPSG}') for p,r in zip(p,r) ) for c,r,p in zip(CENTS, all_radii, all_rain)]

    return rain_ring


#~ RASTERIZE SHPs & INTERPOLATE RAINFALL (between rings) ~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def RASTERIZE(ALL_RINGS, OUTER_RING):
# burn the ALL_RINGS
    tmp = gdal.Rasterize('', ALL_RINGS.to_json(), xRes=X_RES, yRes=Y_RES, allTouched=True,
        attribute='rain', noData=0, outputType=gdal.GDT_Float64, targetAlignedPixels=True,
        outputBounds=[llim, blim, rlim, tlim], outputSRS=f'EPSG:{WGEPSG}', format='MEM')
        #, width=int(abs(rlim-llim)/X_RES), height=int(abs(tlim-blim)/X_RES) )
    fall = tmp.ReadAsArray()
    tmp = None
    #gdal.Unlink('the_tmpfile.tif')
# burn the mask
    tmp = gdal.Rasterize('', OUTER_RING.to_json(), xRes=X_RES, yRes=Y_RES, allTouched=True,
        burnValues=1, noData=0, outputType=gdal.GDT_Int16, targetAlignedPixels=True,
        outputBounds=[llim, blim, rlim, tlim], outputSRS=f'EPSG:{WGEPSG}', format='MEM')
        #, width=int(abs(rlim-llim)/X_RES), height=int(abs(tlim-blim)/X_RES) )
    mask = tmp.ReadAsArray()
    tmp = None
# re-touching the mask...to do a proper interpolation
    mask[np.where(fall!=0)] = 0
# everything that is 1 is interpolated
    fill.fillnodata(np.ma.array(fall, mask=mask), mask=None, max_search_distance=4.0, smoothing_iterations=2)
    return fall


#~ COMPUTE STATS OVER A DEM.RASTER (GIVEN A SHP.POLYGON) ~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def ZTRATIFICATION( Z_OUT ):
    # global qants, ztats
    if Z_CUTS:
# calculate zonal statistics
        # test = zonal_stats(SHP_FILE, './data_WG/dem/WGdem_wgs84.tif', stats='count min mean max median')
        # IF YOUR DEM IS IN WGS84... RE-PROJECT THE POLYGONS TO 4326 (WGS84)
        ztats = zonal_stats(vectors=Z_OUT.to_crs(epsg=4326).geometry, raster=DEM_FILE, stats=Z_STAT)
        # # OTHERWISE, A FASTER APPROACH IS HAVING THE DEM/RASTER IN THE LOCAL CRS
        # # ...i.e., DEM_FILE=='./data_WG/dem/WGdem_26912.tif'
        # ztats = zonal_stats(vectors=Z_OUT.geometry, raster=DEM_FILE, stats=Z_STAT)
# to pandas
        ztats = pd.DataFrame( ztats )
# column 'E' classifies all Z's according to the CUTS
        ztats['E'] = pd.cut(ztats[ Z_STAT ], bins=cut_bin, labels=cut_lab, include_lowest=True)
        ztats.sort_values(by='E', inplace=True)
# storm centres/counts grouped by BAND
# https://stackoverflow.com/a/20461206/5885810  (index to column)
        qants = ztats.groupby(by='E').count().reset_index(level=0)
    else:
# https://stackoverflow.com/a/17840195/5885810  (1-row pandas)
        qants = pd.DataFrame( {'E':'', Z_STAT:len(Z_OUT)}, index=[0] )
        # ztats = pd.DataFrame( {'E':np.repeat('',len(Z_OUT))} )
        ztats = pd.Series( range(len(Z_OUT)) )      # 5x-FASTER! than the line above
    return qants, ztats

#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#- RASTER MANIPULATION ------------------------------------------------- (END) #
#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


#~ REMOVE (ACCORDINGLY) DURATIONS OUTSIDE [MIN_DUR, MAX_DUR].range ~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def CHOP( DUR_S, DATES, STORM_MATRIX ):
# find the indexes outside the limits
    outdur = np.concatenate((np.where(DUR_S<MIN_DUR).__getitem__(0) if MIN_DUR else np.empty(0),
        np.where(DUR_S>MAX_DUR).__getitem__(0) if MAX_DUR else np.empty(0))).astype('int')
    d_bool = ~np.in1d(range(len(STORM_MATRIX)), outdur)
# update 'vectors'
    return DUR_S[ d_bool ], DATES[ d_bool ], [item for i, item in enumerate(STORM_MATRIX) if d_bool[i]]


#~ STORING LAYERS OF RAINFALL TO COMPUTE AGGREGATES ~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def RAIN_CUBE( STORM_MATRIX, DATES, MRAIN, ragg ):#ragg=CUM_S
    data = xr.DataArray(data=STORM_MATRIX, dims=['time','row','col'])
    data.coords['time'] = DATES
    # data.coords['time'] = pd.to_datetime(DATES, unit='s',
    #     origin=DATE_ORIGIN.replace(tzinfo=None))#.tz_localize(tz=TIME_ZONE)
# unless you end up with more than 256.gauges per pixel... leave it as 'u1'
    data.coords['mask'] = (('row','col'), CATCHMENT_MASK.astype('u1'))
# step-cumulative rainfall
    # dagg = data.where(data.mask!=0, np.nan).cumsum(dim='time', skipna=False)
    dagg = data.cumsum(dim='time', skipna=False).where(data.mask!=0, np.nan)

# which aggretation(s) surpass MRAIN?
# the line below is enough for the SIMULATION.scenario
    suma = dagg.median(dim=('row','col'), skipna=True) + ragg
# # the line below allows SIMULATION & VALIDATION!!... basically, rainfall.depths are
# # duplicated/replicated by the 'MASK.frequency', and then the MEDIAN is computed.
# # https://stackoverflow.com/a/57889354/5885810  (apply xr.apply_ufunc)
# # https://stackoverflow.com/a/35349142/5885810  (weighted median)
#     suma = xr.apply_ufunc(lambda x: np.nanmedian(np.repeat(np.ravel(x), np.ravel(data.mask)), skipna=False)\
#         ,dagg ,input_core_dims=[['row','col']], vectorize=True, dask='allowed') + ragg

    xtra = np.where( suma >= MRAIN ).__getitem__(0)
# if two consecutive aggretated fields are the same -> no storm fell within AOI
    void = np.where(data.where(data.mask!=0, np.nan).sum(dim=('row','col'), skipna=True) == 0).__getitem__(0)
# REMOVING VOID FIELDS
    drop = np.union1d(void, xtra[1:])
    data = data.drop_isel( time = drop )
    # data.where(data.mask!=0, np.nan).cumsum(dim='time', skipna=False).median(dim=('row','col'))
# finds whether MRAIN is reached (or not) in this (a given) iteration
    ends = np.delete(suma, drop).values.__getitem__(-1)
    # # do we want to output aggregated fields?
    # dagg = dagg.drop_isel( time = drop )
    # return ends, data, drop, dagg
    return ends, data, drop


    # # VISUALISATION
    # # import matplotlib.pyplot as plt
    # # from matplotlib.patches import Circle
    # fig, ax = plt.subplots(figsize=(7,7), dpi=150)
    # ax.set_aspect('equal')
    # # plt.imshow(fall, interpolation='none', aspect='equal', origin='upper',# alpha=0,
    # plt.imshow(STORM_MATRIX[0], interpolation='none', aspect='equal', origin='upper',
    #            cmap='gist_ncar_r', extent=(llim[0], rlim[0], blim[0], tlim[0]))
    # # Now, loop through coord arrays, and create a circle at each x,y pair
    # posx = 0#32#9
    # for rr in all_radii[posx] *1e3:
    #     circ = Circle((CENTS[posx][0], CENTS[posx][1]), rr, alpha=1, facecolor='None',
    #         edgecolor=npr.choice(['xkcd:lime green','xkcd:gold','xkcd:electric pink','xkcd:azure']))
    #     ax.add_patch(circ)
    # # plt.show()
    # plt.savefig('tmp_ras_22.jpg', bbox_inches='tight',pad_inches=0.02, facecolor=fig.get_facecolor())
    # plt.close()
    # plt.clf()

    # # plt.imshow(da.mask, interpolation='none', aspect='equal', origin='upper',
    # # plt.imshow(data[19,:,:].data, interpolation='none', aspect='equal', origin='upper',
    # plt.imshow(CATCHMENT_MASK, interpolation='none', aspect='equal', origin='upper',
    #            cmap='gist_ncar_r', extent=(llim[0], rlim[0], blim[0], tlim[0]))


#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#- NC.FILE CREATION -------------------------------------------------- (START) #
#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#~ TO SCALE 'DOWN' FLOATS TO INTEGERS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def NCBYTES_RAIN( INTEGER ):
    global SCL, ADD
    MINIMUM = 0                     # -->          0  (0 because it's unsigned)
    MAXIMUM = +(2**( INTEGER *8 ))  # -->      65536  (largest unsigned integer of 16 Bits)
    # MAXIMUM = +(2**( 32 ))        # --> 4294967296  (largest unsigned integer of 32 Bits)
# if you want to work with signed integers (e.g.)...
    # MINIMUM = -(2**(16-1))        # -->     -32768  (smallest  signed integer of 16 Bits)
    # MAXIMUM = +(2**(16-1)-1)      # -->     +32767  (largest   signed integer of 16 Bits)

# # run your own (customized) tests
# temp = 3.14159
# seed = 1133
# epsn = 0.006
# while temp > epsn:
#     temp = (seed - 0.) / (MAXIMUM - (MINIMUM + 1))
#     seed = seed - 1
# print( (f'starting from 0, you\'d need a max. of {seed+1} to guarantee an epsilon of {epsn}') )
# # starting from 0, you'd need a max. of:  65 to guarantee an epsilon of 0.001
# # starting from 0, you'd need a max. of: 327 to guarantee an epsilon of 0.005
# # starting from 0, you'd need a max. of: 393 to guarantee an epsilon of 0.006
# # starting from 0, you'd need a max. of: 655 to guarantee an epsilon of 0.01
# # starting from 0, you'd need a max. of: 429496 to guarantee an epsilon of 0.0001 (for INTEGER==4)

# NORMALIZING THE RAINFALL SO IT CAN BE STORED AS 16-BIT INTEGER (65,536 -> unsigned)
# https://stackoverflow.com/a/59193141/5885810      (scaling 'integers')
# https://stats.stackexchange.com/a/70808/354951    (normalize data 0-1)
    iMIN = 0.
# 655 (precision==0.01 for 16-Bit Int) seems a reasonable 'resolution'/limit for 'daily' rainfall.
# if you want a larger precision (or your variable is in the 'low' scale,
# ...say Summer Temperatures in Celsius) you must/could lower this limit.
    iMAX = 655.
    # SCL = (iMAX - iMIN) / (MAXIMUM - (MINIMUM + 1))   # if one wants UNsigned INTs
    SCL = (iMAX - iMIN) / (MAXIMUM - (MINIMUM + 0))
    ADD = iMAX - SCL * MAXIMUM


#~ SKELETON OF THE NC (OUPUT) FILE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def NC_FILE_I( nc, nsim ):
    global tag_y, tag_x
# define SUB.GROUP and its dimensions
    sub_grp = nc.createGroup(f'{MODE.lower()}_{"{:02d}".format(nsim+1)}')

    sub_grp.createDimension('y', len(YS))
    sub_grp.createDimension('x', len(XS))
    sub_grp.createDimension('t', None)                  # unlimited
    sub_grp.createDimension('n', NUMSIMYRS)

#- LOCAL.CRS (netcdf definition) -----------------------------------------------
#-------------------------------------------------------------------------------
    """
Customization of these parameters for your local CRS is relatively easy!.
All you have to do is to 'convert' the PROJ4 (string) parameters of your (local)
projection into CF conventions.
# https://cfconventions.org/wkt-proj-4.html
# http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/cf-conventions.html#appendix-grid-mappings
# https://spatialreference.org/
In this case: "+proj=utm +zone=12 +ellps=GRS80 +datum=NAD83 +units=m +no_defs"
The amount (and type) of parameters will vary depending on your local CRS.
The use of PROJ4 is now being discouraged (https://inbo.github.io/tutorials/tutorials/spatial_crs_coding/);
neverthelesss, it perfectly works under this framework to straightforwardly
store data in the local CRS, and at the same time be able to visualize it in
WGS84 (via, e.g., https://www.giss.nasa.gov/tools/panoply/) without the need
to transform (and store) local coordinates into Lat-Lon.

IF FOR SOME REASON YOU'D ULTIMATELY PREFER TO STORE YOUR DATA IN WGS84, COMMENT
OUT ALL THIS SECTION & ACTIVATE THE SECTION "- WGS84.CRS (netcdf definition) -"
    """
    grid = sub_grp.createVariable('crs', 'int')
    grid.grid_mapping_name = 'universal_transverse_mercator'
    grid.utm_zone_number = 12
    grid.horizontal_datum_name = 'NAD83'
    grid.reference_ellipsoid_name = 'GRS80'
    grid.unit = 'm'
    grid.EPSG_code = f'EPSG:{WGEPSG}'                   # "EPSG:26912"
    grid.proj4_params = pp.CRS.from_epsg(WGEPSG).to_proj4()
    grid._CoordinateTransformType = "Projection"
    grid._CoordinateAxisTypes = "GeoY GeoX"

    # # STORING LOCAL COORDINATES
    yy = sub_grp.createVariable('projection_y_coordinate', 'i4', dimensions=('y')
                                ,chunksizes=CHUNK_3D( [ len(YS) ], valSize=4))
    xx = sub_grp.createVariable('projection_x_coordinate', 'i4', dimensions=('x')
                                ,chunksizes=CHUNK_3D( [ len(XS) ], valSize=4))
    yy[:] = YS
    xx[:] = XS
    # yy[:] = np.flipud( np.linspace(3498500,3520500,23) )
    # xx[:] = np.linspace(575500,611500,37)
    yy.coordinates = 'projection_y_coordinate'
    xx.coordinates = 'projection_x_coordinate'
    yy.units = 'meter'
    xx.units = 'meter'
    yy.long_name = 'y coordinate of projection'
    xx.long_name = 'x coordinate of projection'
    yy._CoordinateAxisType = "GeoY"
    xx._CoordinateAxisType = "GeoX"
    yy.grid_mapping = 'crs'
    xx.grid_mapping = 'crs'

#- WGS84.CRS (netcdf definition) -----------------------------------------------
#-------------------------------------------------------------------------------
    # # http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/cf-conventions.html#_trajectories
    # grid = sub_grp.createVariable('crs', 'int')
    # grid.grid_mapping_name = "latitude_longitude"
    # grid.longitude_of_prime_meridian = 0.
    # grid.semi_major_axis = 6378137.
    # grid.inverse_flattening = 298.257223563
    # grid._CoordinateAxisTypes = "Lat Lon"
    # grid.proj4_params = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"  # ADAGUC extension
    # # grid.proj4_params = pp.CRS('EPSG:4326').to_proj4()
    # grid.EPSG_code = "EPSG:4326"

    # # STORING WGS84 COORDINATES
    # lat, lon = pp.Transformer.from_crs(f'EPSG:{WGEPSG}','EPSG:4326').transform(
    #     np.meshgrid(XS,YS).__getitem__(0), np.meshgrid(XS,YS).__getitem__(-1),
    #     zz=None, radians=False)
    # # https://pyproj4.github.io/pyproj/stable/gotchas.html#upgrading-to-pyproj-2-from-pyproj-1
    # yy = sub_grp.createVariable('latitude' , 'f8', dimensions=('y','x')
    #                             ,chunksizes=CHUNK_3D( [ len(YS), len(XS) ], valSize=8))
    # xx = sub_grp.createVariable('longitude', 'f8', dimensions=('y','x')
    #                             ,chunksizes=CHUNK_3D( [ len(YS), len(XS) ], valSize=8))
    # yy[:] = lat
    # xx[:] = lon
    # yy.coordinates = 'latitude'
    # xx.coordinates = 'longitude'
    # yy.units = 'degrees_north'
    # xx.units = 'degrees_east'
    # yy.long_name = 'latitude coordinate'
    # xx.long_name = 'longitude coordinate'
    # yy._CoordinateAxisType = "Lat"
    # xx._CoordinateAxisType = "Lon"
    # yy.grid_mapping = 'crs'
    # xx.grid_mapping = 'crs'

    tag_y = yy.getncattr('coordinates')
    tag_x = xx.getncattr('coordinates')

# store the MASK
    mask_chunk = CHUNK_3D( [ len(YS), len(XS) ], valSize=1)
    ncmask = sub_grp.createVariable('mask', 'i1', dimensions=('y','x')
        ,chunksizes=mask_chunk, zlib=True, complevel=9)#,fill_value=0
    ncmask[:] = CATCHMENT_MASK.astype('i1')
    ncmask.grid_mapping = 'crs'
    ncmask.long_name = 'catchment mask'
    ncmask.description = 'n>0 means catchment or gauge(s) : 0 is void'
    #ncmask.coordinates = f'{yy.getncattr("coordinates")} {xx.getncattr("coordinates")}'
    ncmask.coordinates = f'{tag_y} {tag_x}'

# if ANOTHER/OTHER variable is needed
    ncxtra = sub_grp.createVariable('duration', 'f4', dimensions=('t','n')
        ,zlib=True, complevel=9, fill_value=np.nan)# ,fill_value=np.r_[0].astype('u2'))
    ncxtra.long_name = 'storm duration'
    ncxtra.units = 'minutes'
    ncxtra.precision = f'{1/60}'                        # (1 sec); see last line of 'NC_FILE_II'
    # ncxtra.scale_factor = dur_SCL
    # ncxtra.add_offset = dur_ADD
    iixtra = sub_grp.createVariable('sampled_total', 'f4', dimensions=('n')
        ,zlib=True, complevel=9, fill_value=np.nan)
    iixtra.long_name = 'seasonal total from PDF'
    iixtra.units = 'mm'

    return sub_grp#, yy.getncattr('coordinates'), xx.getncattr('coordinates')


#~ FILLING & CLOSURE OF THE NC (OUPUT) FILE  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def NC_FILE_II( sub_grp, simy, KUBE, XTRA, TOTP ):
# define & fill the TIME.variable/dimension
    nctnam = f'time_{"{:02d}".format(simy+1)}'
    # nctnam = f'time_{"{:03d}".format(simy+1)}'        # if more than 100 years/simul are planned
    sub_grp.createDimension(nctnam, len(KUBE.time))
    chunkt = CHUNK_3D( [ len(KUBE.time) ], valSize=8)   # because 'i8'
    timexx = sub_grp.createVariable(nctnam, 'i8', (nctnam), chunksizes=chunkt)
    timexx[:] = KUBE.time.data
    timexx.long_name = 'starting time'
    timexx.units = 'seconds since ' + DATE_ORIGIN.strftime('%Y-%m-%d %H:%M:%S')#'%Y-%m-%d %H:%M:%S %Z%z'
    timexx.calendar = 'gregorian'#'proleptic_gregorian'
    timexx.coordinates = nctnam
    timexx._CoordinateAxisType = 'Time'

# define & fill the RAINFALL variable
    ncvnam = f'year_{SEED_YEAR+simy}'
    chunkx = CHUNK_3D( [ len(KUBE.time), len(YS), len(XS) ], valSize=INTEGER)
    ncvarx = sub_grp.createVariable(ncvnam, datatype=f'u{INTEGER}'
        ,dimensions=(nctnam,'y','x'), chunksizes=chunkx, zlib=True, complevel=9)#,least_significant_digit=3)
    integer_array = ( (KUBE.data - ADD) / SCL ).astype(f'u{INTEGER}')
# # f'i{INTEGER}' converts 'np.nan' into 0s...so turn those zeros into 'MINIMUM'
#     integer_array[ np.isnan(integer_array) ] = MINIMUM
    ncvarx[:] = integer_array
    ncvarx.units = 'mm'
    ncvarx.precision = 1e-2
    ncvarx.long_name = 'storm rainfall'
    ncvarx.grid_mapping = 'crs'
    #ncvarx._FillValue = np.array(MINIMUM).astype(f'i{INTEGER}')
    ncvarx.scale_factor = SCL
    ncvarx.add_offset = ADD
    ncvarx.coordinates = f'{tag_y} {tag_x}'
    #ncvarx.coordinates = f'{yy.getncattr("coordinates")} {xx.getncattr("coordinates")}'

# define & fill some other XTRA variable (previously set up)
# 'f4' guarantees 1-second (1/60 -minute) precision
    sub_grp.variables['duration'][:,simy] = ((XTRA *60).round(0) /60).astype('f4')
# # https://stackoverflow.com/a/28425782/5885810  (round to the nearest-nth) -> second
#     sub_grp.variables['duration'][:,simy] = list(map(lambda x: round(x /(1/60)) *1/60, ass ))
    sub_grp.variables['sampled_total'][simy] = TOTP.astype('f4')


#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#- NC.FILE CREATION ---------------------------------------------------- (END) #
#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


#%% COLLECTOR

#~ wrapper of all previous 'sub'-routines  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def STORM( NC_NAMES ):
    global cut_lab, cut_bin
# storm minimum radius depends on spatial.resolution (for raster purposes)
# ...it must be used/assigned in KM, as its distribution was 'computed' in KM
    MIN_RADIUS = np.max([X_RES, Y_RES]) /1e3
# define Z_CUTS labelling (if necessary)
    if Z_CUTS:
        cut_lab = [f'Z{x+1}' for x in range(len(Z_CUTS) +1)]
        cut_bin = np.union1d(Z_CUTS, [0, 9999])
# define transformation constants/parameters for INTEGER rainfall-NC-output
    NCBYTES_RAIN( INTEGER )
# read (and check) the PDF-parameters
    READ_PDF_PAR()
    CHECK_PDF()
# define some xtra-basics
    WET_SEASON_DAYS()
    SHP_OPS()

    # # https://gis.stackexchange.com/a/267326/127894     (get EPSG/CRS from raster)
    # from osgeo import osr
    # tIFF = gdal.Open( DEM_FILE )
    # tIFF = gdal.Open( './data_WG/dem/WGdem_WGS84.tif' )
    # tIFF_proj = osr.SpatialReference( wkt=tIFF.GetProjection() ).GetAttrValue('AUTHORITY', 1)
    # tIFF = None

    print('\nRUN PROGRESS')
    print('************')

# FOR EVERY SEASON
    for seas in range( SEASONS ):#seas=0#seas=1
    # ESTABLISH HOW THE DOY-SAMPLING WILL BE DONE
        tod_fun = 'TOD_CIRCULAR' if all( list(map(lambda k:
            DOYEAR[ seas ].keys().__contains__(k), ['p', 'mus', 'kappas'])) ) else 'TOD_DISCRETE'
    # CREATE NC.FILE
        ncid = NC_NAMES[ seas ]
        nc = nc4.Dataset(ncid, 'w', format='NETCDF4')
        nc.created_on = datetime.now(tzlocal()).strftime('%Y-%m-%d %H:%M:%S %Z')#%z
        print(f'\nSEASON: {seas+1}/{SEASONS}')

# FOR EVERY SIMULATION
        for nsim in range( NUMSIMS ):#nsim=0
            print(f'\t{MODE.upper()}: {"{:02d}".format(nsim+1)}/{"{:02d}".format(NUMSIMS)}')#, end='', flush=False)
        # 1ST FILL OF THE NC.FILE
            sub_grp = NC_FILE_I( nc, nsim )

# FOR EVERY YEAR (of the SIMULATION)
            for simy in tqdm( range( NUMSIMYRS ), ncols=50 ):#simy=0
            # sample total monsoonal rainfall to reach & increase/decrease (such) total monsoonal rainfall
                MRAIN = SEASONAL_RAIN( TOTALP, seas ) * (1 + PTOT_SC[ seas ] + ((simy +0) *PTOT_SF[ seas ]))
                # MHELP = SEASONAL_RAIN( TOTALP, seas )
                # MRAIN = ( MHELP * (1 + PTOT_SC[ seas ]) ) * (1 + ((simy +1) *PTOT_SF[ seas ]))
                # print(f'\nnsim:{nsim}  |  simy:{simy}  |  MRAIN:{MRAIN}')
            # initialize some VOID arrays
                r_ain = []
                l_ong = []
            # for the WGc-case we start with (~40 days/month * 5 months (WET-S1)) = 200
            # ...hence, we're assuming that (initially) we have more than 1 storm/day
            # ...(then we continue 'half-fing' the above seed)
# 40*4 (FOR '_SF' RUNS); 40*2 (FOR '_SC' RUNS); 40*1 (FOR 'STANDARD' RUNS)
                NUM_S = 40*2 * M_LEN[ seas ].__getitem__(0)
                CUM_S = 0

# DO IT UNTIL THE TOTAL RAINFALL IS REACHED OR NO MORE STORMS TO COMPUTE
                while CUM_S < MRAIN and NUM_S >= 2:
                # sample random storm centres
                # https://stackoverflow.com/a/69630606/5885810  (rnd pts within shp)
                    CENTS = poisson( BUFFRX.geometry.xs(0), size=NUM_S )

                # # you wanna PLOT the STORM CENTRES??
                #     import matplotlib.pyplot as plt
                #     import cartopy.crs as ccrs
                #     fig = plt.figure(figsize=(10,10), dpi=300)
                #     ax = plt.axes(projection=ccrs.epsg(WGEPSG))
                #     ax.set_aspect(aspect='equal')
                #     for spine in ax.spines.values(): spine.set_edgecolor(None)
                #     fig.tight_layout(pad=0)
                #     BUFFRX.plot(edgecolor='xkcd:amethyst', alpha=1., zorder=2, linewidth=.77,
                #                 ls='dashed', facecolor='None', ax=ax)
                #     plt.scatter(CENTS[:,0], CENTS[:,1], marker='P', s=37, edgecolors='none')
                #     plt.show()

                # # if FORCE_BRUTE was used -> truncation deemed necessary to avoid
                # # ...ULTRA-high intensities [chopping almost the PDF's 1st 3rd]
                    # BETAS = TRUNCATED_SAMPLING( BETPAR[ seas ][''], [-0.008, +0.078], NUM_S )
                    # [BETPAR[ seas ][''].cdf(x) if x else None for x in [-.035, .035]]
                    BETAS = RANDOM_SAMPLING( BETPAR[ seas ][''], NUM_S )
                # sampling maxima radii
                    RADII = TRUNCATED_SAMPLING( RADIUS[ seas ][''], [1* MINRADIUS, None], NUM_S )
                # polygon(s) for maximum radii
                    RINGO = LAST_RING( RADII, CENTS )

                # define pandas to split the Z_bands (or not)
                    qants, ztats = ZTRATIFICATION( pd.concat( RINGO ) )
                # compute copulas given the Z_bands (or not)
                    MAX_I, DUR_S = list(map(np.concatenate, zip(* qants.apply( lambda x:\
                        COPULA_SAMPLING(COPULA, seas, x['E'], x['median']), axis='columns') ) ))
                # sort back the arrays
                    MAX_I, DUR_S = list(map( lambda A: A[ np.argsort( ztats.index ) ], [MAX_I, DUR_S] ))
                # increase/decrease maximum intensites
                    MAX_I = MAX_I * (1 + STORMINESS_SC[ seas ] + ((simy +0) *STORMINESS_SF[ seas ]))
                # sample some dates (to capture intra-seasonality & for NC.storing)
                    DATES = eval(f'{tod_fun}( {NUM_S}, {seas}, {simy} )')
                    # DATES = TOD_CIRCULAR( NUM_S, seas, simy )
                # compute granular rainfall over intermediate rings
                    RINGS = LOTR( RADII, MAX_I, DUR_S, BETAS, CENTS )
                # COLLECTING THE STORMS
                    STORM_MATRIX = list(map(RASTERIZE, RINGS, RINGO))
                # updating/removing long/short storm-durations
                    DUR_S, DATES, STORM_MATRIX = CHOP( DUR_S, DATES, STORM_MATRIX )
                # rainfall aggregation
                    CUM_S, rain, remove = RAIN_CUBE( STORM_MATRIX, DATES, MRAIN, CUM_S )

                    r_ain.append( rain )
                    l_ong.append( np.delete(DUR_S, remove) )

                # 'decreasing the counter'
                    NUM_S = int(NUM_S /2)#/1.5)

            # WARN IF THERE IS NO CONVERGING
                assert not (CUM_S < MRAIN and NUM_S < 2), f'Iteration for {MODE.upper()} '\
                    f'{nsim+1}: YEAR {simy} not converging!!\nTry a larger initial '\
                    'seed (i.e., variable "NUM_S"). If the problem persists, it '\
                    'might be very likely that the catchment (stochastic) '\
                    'parameterization is not adequate.' # the more you increase the slower it gets!!

                q_ain = xr.concat(r_ain, dim='time')    # stack (rainfall) arrays
                idx_s = q_ain.time.argsort().data       # sort indexes

            # here unique time.stamps are guaranteed globally!
                # q_ain = q_ain.assign_coords(time = DUAL_STAMP( q_ain.time.data ) )
                q_ain = q_ain.assign_coords(time = DUAL_STAMP( q_ain.time[idx_s].data ) )

            # LAST FILL OF THE NC.FILE
                NC_FILE_II( sub_grp, simy, q_ain, np.concatenate(l_ong)[ idx_s ], MRAIN )
            #...IF (global) 'q_ain.time' is indeed modified by DUAL_STAMP, 'idx_s' might no completely
            #...coincide with the updated 'q_ain.time'. Still, this is not an actual concern!
        nc.close()


#%%

if __name__ == '__main__':

    from pathlib import Path
    Path( abspath( join(parent_d, OUT_PATH) ) ).mkdir(parents=True, exist_ok=True)
    # define NC.output file.names
    NC_NAMES =  list(map( lambda a,b,c: f'{abspath( join(parent_d, OUT_PATH) )}/{MODE[:3].upper()}_'\
        f'{datetime.now(tzlocal()).strftime("%y%m%dT%H%M")}_S{a+1}_{b.strip()}_{c.strip()}.nc',\
        # range(SEASONS), PTOT_SCENARIO, STORMINESS_SCENARIO ))
        range(SEASONS), ['nada','zero'], ['zero','nada'] ))

    STORM( NC_NAMES )
    # # testing for only ONE Season!
    # STORM( [f'./model_output/{MODE[:3].upper()}_test.nc'] )