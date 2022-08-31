
import numpy as np
import pandas as pd
# from pathlib import Path
# from datetime import datetime
from dateutil.tz import tzlocal
# # https://stackoverflow.com/a/23116937/5885810  (0 divison -> no warnings)
# # https://stackoverflow.com/a/29950752/5885810  (0 divison -> no warnings)
# np.seterr(divide='ignore', invalid='ignore')


import warnings
# https://stackoverflow.com/a/9134842/5885810   (supress warning by message)
warnings.filterwarnings('ignore', message='You will likely lose important '\
    'projection information when converting to a PROJ string from another format')
warnings.filterwarnings('ignore', message='Failed to load cfgrib - most likely '\
    'there is a problem accessing the ecCodes library.')

from scipy import stats
from numpy import random as npr
from statsmodels.distributions.copula.api import GaussianCopula
#from statsmodels.distributions.empirical_distribution import ECDF
from scipy.interpolate import interp1d

import geopandas as gpd
#from matplotlib.path import Path
from osgeo import gdal

from pointpats.random import poisson
#import rasterio
from rasterio import fill
import xarray as xr
# https://stackoverflow.com/a/65562060/5885810  (ecCodes in WOS)

# only necessary if you use Z_CUTS & SIMULATION
from rasterstats import zonal_stats

import pyproj as pp
from pyproj import Transformer

from datetime import timedelta, timezone, datetime
from dateutil.relativedelta import relativedelta
from zoneinfo import ZoneInfo

import vonMisesMixtures as vonmises
""" STORM2.0 ALSO runs WITHOUT this library!! """

import netCDF4 as nc4
from chunking import CHUNK_3D

from tqdm import tqdm
##import re


from check_input import PAR_UPDATE

from parameters import *


# print('\nXXX')
# print(PTOT_SC)
# print(PTOT_SF)
# print('YYY\n')


#%% GREETINGS

"""
STORM - STOchastic Rainfall Model STORM produces realistic regional or watershed
rainfall under various climate scenarios based on empirical-stochastic
selection of historical rainfall characteristics.

Based on Singer, M. B., and Michaelides, K. (2017), Deciphering the expression
of climate change within the Lower Colorado River basin by stochastic simulation
of convective rainfall. [ https://doi.org/10.1088/1748-9326/aa8e50 ]

version name: STORM2.0

Authors:
  Michael Singer 2017
  Manuel F. Rios Gaona 2022
Date created: 2015-6. Last modified 18/05/2022

###-----------------------------------------------------------------------------

MODE is a string in single quotations containing one of these two options:
'Validation' for validating STORM's output against historical observations
'Simulation' for simulating stochastic rainfall under various climate scenarios
MODE affects where STORM generates output--either for a set of
gauge locations for comparison with observational data (Validation) or on
an aribitrary watershed grid (Simulation).

SEASONS is a number specifying whether there are 'one' or 'two' seasons. If there are
two seasons, pdfs will be required specifying rainfall characteristics for
the 2nd season (Ptot, Duration, etc). Also, ptot and storminess scenarios will be required
arguments in the function (see below). Currently, season one runs from May 1 to
Sept 30 and season two runs from Oct 1 to Apr 30, but these dates can be changed.

NUMSIMS is the integer number of x-year (NUMSIMYRS) simulations to be run
as a batch. Each simulation will produce its own output folder that is
timestamped with relevant simulation information

NUMSIMYRS is the integer number of years in each simulation. Recommended value >=30

"""


#%% INPUT PARAMETERS

"""
# STORM2.0 RUNS WITH THE PARAMETERS BELOW (PLACED HERE FOR 'ILUSTRATIVE' PURPOSES).
# THEIR TWEAKING SHOULD EXCLUSIVELY BE DONE IN THE FILE 'check_input.py'.
# 'check_input.py' ALSO OFFERS A MORE DETAILED EXPLANATION ON THEIR MEANING/VALUES.


MODE = 'SImuLAtiON'     # Type of Run (case-insensitive). Either 'SIMULATION' or 'VALIDATION'
SEASONS = 1             # Number of Seasons (per Run)
NUMSIMS = 2             # Number of runs per Season
NUMSIMYRS = 2           # Number of years per run (per Season)

# # PARAMETER = [ S1 ,  S2 ]
PTOT_SC       = [ 0. , - .0]
PTOT_SF       = [+0.0, -0. ]
STORMINESS_SC = [ 0.0, + .0]
STORMINESS_SF = [-0.0,  0.0]

# # if you intend to model just 1 SEASON
# # YOU CAN DO (e.g.):
# PTOT_SC       = [0.15]
# PTOT_SF       = [ 0.0]
# STORMINESS_SC = [-0.1]
# STORMINESS_SF = [ 0.0]
# # OR (e.g.):
PTOT_SC       = [0.15, None]
PTOT_SF       = [ 0.0, None]
STORMINESS_SC = [-0.1, None]
STORMINESS_SF = [ 0.0, None]

PRE_FILE = './model_input/ProbabilityDensityFunctions_TWO.csv'  # output from 'pre_processing.py'
GAG_FILE = './model_input/data_WG/gage_data--gageNetworkWG.csv' # gage (meta-)data (optional*)
# GAG_FILE = None
SHP_FILE = './model_input/shp/WG_Boundary.shp'                  # catchment shape-file in WGS84
DEM_FILE = './model_input/dem/WGdem_wgs84.tif'                  # aoi raster-file (optional**)
# DEM_FILE = './model_input/dem/WGdem_26912.tif'                # aoi raster-file in local CRS (***)
# DEM_FILE = None
OUT_PATH = './model_output'                                     # output folder

Z_CUTS = None           # (or Z_CUTS = []) for INT-DUR copula nodelling regardless altitude
Z_CUTS = [1350, 1500]   # in meters!
Z_STAT = 'median'       # statistic to retrieve from the DEM ['mean' or 'min'|'max'?? not 'count']

WGEPSG    = 26912       # EPSG Code of the local/regular Coordinate Reference System (CRS)
X_RES     = 1000        # in meters! (for the 'regular/local' CRS)
Y_RES     = 1000        # in meters! (for the 'regular/local' CRS)
BUFFER    = 5000        # in meters! -> buffer distance (out of the catchment)
CLOSE_DIS = 0.15        # in km -> small circle emulating the storm centre's point/dot
RINGS_DIS = 2.1         # in km -> distance between (rainfall) rings

### these parameters allow to pin down a time-dimension to the storms
SEED_YEAR      = None                       # for your SIM/VAL to start in the current year
# SEED_YEAR    = 2050                       # for your SIM/VAL to start in 2050
SEASONS_MONTHS = [[6,10], None]             # JUNE through OCTOBER (just ONE season)
# # OR:
# SEASONS_MONTHS = [[10,5], ['jul','sep']]  # OCT[y0] through MAY[y1] (& JULY[y1] through SEP[y1])
# SEASONS_MONTHS = [['may','sep'],[11,12]]  # MAY through SEP (& e.g., NOV trhough DEC)
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



#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#- CONSTRUCT THE PDFs (TO SAMPLE FROM) ------------------------------- (START) #
#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#- READ THE CSV.FILE(s) PRODUCED BY THE preprocessing.py SCRIPT ----------------
#-------------------------------------------------------------------------------
def READ_PDF_PAR():
    global PDFS
# read PDF-parameters
# https://stackoverflow.com/a/58227453/5885810
    PDFS = pd.read_fwf(PRE_FILE, header=None)
    PDFS = PDFS.__getitem__(0).str.split(',', expand=True).set_index(0).astype('f8')

# # read the ECDF so you can back-transform your copula into Int/Dur (or assume your conditional-PDFs)
# #--revise these ECDF...especially COP_INTecdf -> way too large (generates 0's)
#     COP_INTecdf = pd.read_csv('./model_input/CopulasECDFintensity.csv', header='infer', comment='#', sep=',')
#     COP_DURecdf = pd.read_csv('./model_input/CopulasECDFduration.csv', header='infer', comment='#', sep=',')


#- CONSTRUCT PDFs FROM PARAMETERS (stored in 'PDFS') ---------------------------
#-------------------------------------------------------------------------------
def RETRIEVE_PDF( TAG ):#TAG='RADIUS_PDF'#TAG='DATIME_VMF'#TAG='COPULA_RHO'#TAG='MAXINT_PDF'
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
        name = ['']                                     # makes "distros" 'universal'

# https://www.geeksforgeeks.org/python-get-first-element-of-each-sublist/
    first = list(list(zip( *subset.index.str.split('+') )).__getitem__(0))
# https://stackoverflow.com/a/6979121/5885810
# https://stackoverflow.com/a/5252867/5885810
# https://stackoverflow.com/a/46453340/5885810
    sort_id = np.unique( list(map( lambda x: x.replace(TAG, ''), first )) )
# the line below makes 1st-PDFs be chosen by default
    sort_id = sort_id[ np.argsort( sort_id.astype('int') )  ]
# # TIP: USE THE LINE BELOW (REPLACING THE LINE ABOVE) IF YOU PREFER 2nd-PDFs INSTEAD
# # https://stackoverflow.com/a/16486305/5885810
#     sort_id = sort_id[ np.argsort( sort_id.astype('int') )[::-1]  ]
    group = [subset[subset.index.str.contains( f'{TAG}{i}' )].dropna(
        how='all', axis='columns') for i in sort_id]

    if TAG == 'DATIME_VMF':
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
                  f"({','.join( i.astype('str').values.ravel() )})")\
                 for item, i in G.iterrows()] )} for G in group]
    return distros


#- RETRIEVE THE PDFs & EVALUATE THEIR 'CONSISTENCY' AGAINST #SEASONS -----------
#-------------------------------------------------------------------------------
def CHECK_PDF():
# https://stackoverflow.com/a/10852003/5885810
# https://stackoverflow.com/q/423379/5885810
    global DATIME, COPULA, TOTALP, RADIUS, BETPAR, MAXINT, AVGDUR#, Z_CUTS

    try:
        DATIME = RETRIEVE_PDF( 'DATIME_VMF' )
    except IndexError:
        # DATIME = [stats.uniform() for x in range(SEASONS)]
        DATIME = [None for x in range(SEASONS)]
        warnings.warn(f'\nNo DATIME_VMF parameters were found in "{PRE_FILE}".'\
            '\nSTORM2.0 will proceed with TOD (Times Of Day) sampled from a '\
            'UNIFORM distribution. If this is not what you want, please '\
            'update the aforementioned file accordingly.', stacklevel=2)

    TOTALP = RETRIEVE_PDF( 'TOTALP_PDF' )
    RADIUS = RETRIEVE_PDF( 'RADIUS_PDF' )
    BETPAR = RETRIEVE_PDF( 'BETPAR_PDF' )
    MAXINT = RETRIEVE_PDF( 'MAXINT_PDF' )
    AVGDUR = RETRIEVE_PDF( 'AVGDUR_PDF' )
    COPULA = RETRIEVE_PDF( 'COPULA_RHO' )

# evaluate consistency between lists (lengths must be consistent with #SEASONS)
    test = ['DATIME', 'COPULA', 'TOTALP', 'RADIUS', 'BETPAR', 'MAXINT', 'AVGDUR']
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





#- WORKING THE CATCHMENT (& ITS BUFFER) MASK(S) OUT ----------------------------
#-------------------------------------------------------------------------------
def SHP_OPS():
    global llim, rlim, blim, tlim, BUFFRX, CATCHMENT_MASK, XS, YS
# read WG-catchment gauge.data (if necessary)
    if MODE.lower() == 'validation':
        gagnet = pd.read_csv(GAG_FILE, sep=',', header='infer', comment='#')
    # just verify that your own gage.network has 'gage_id', X, Y, (Z optional)
        gagnet = gagnet.loc[(gagnet['within_WG']==1) & (gagnet['with_data']==1)]
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
    llim = np.floor( BUFFRX.bounds.minx /X_RES ) *X_RES #+X_RES/2
    rlim = np.ceil(  BUFFRX.bounds.maxx /X_RES ) *X_RES #-X_RES/2
    blim = np.floor( BUFFRX.bounds.miny /Y_RES ) *Y_RES #+Y_RES/2
    tlim = np.ceil(  BUFFRX.bounds.maxy /Y_RES ) *Y_RES #-Y_RES/2

# # BURN A SHP INTO RASTER & VISUALIZE IT
# tmp_file = 'XTORM_v2_tmp-ras_02.tif'
# tmp = gdal.Rasterize(tmp_file, BUFFRX.to_json(), xRes=X_RES, yRes=Y_RES,
#                      allTouched=True, burnValues=1, noData=0, outputType=gdal.GDT_Int16,
#                      targetAlignedPixels=True, outputBounds=[llim, blim, rlim, tlim],
#                      outputSRS=f'EPSG:{WGEPSG}', format='GTiff')
# var = tmp.ReadAsArray()
# tmp = None

# import matplotlib.pyplot as plt
# plt.imshow(var, interpolation='none')
# plt.show()
# # OR
# from rasterio.plot import show
# tmp_file = 'XTORM_v2_tmp-ras_02.tif'
# srcras = rasterio.open(tmp_file)
# fig, ax = plt.subplots()
# ax = rasterio.plot.show(srcras, extent=[
#     srcras.bounds[0], srcras.bounds[2], srcras.bounds[1], srcras.bounds[3]],
#     ax=ax, cmap='viridis')
# srcras.close()

# BURN THE CATCHMENT SHP INTO RASTER (WITH CATCHMENT-BUFFER EXTENSION)
# https://stackoverflow.com/a/47551616/5885810
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
    # plt.imshow(CATCHMENT_MASK, interpolation='none', aspect='equal', origin='upper', cmap='nipy_spectral_r', extent=(llim[0], rlim[0], blim[0], tlim[0]))
    # plt.show()

# # https://gis.stackexchange.com/questions/344942/raster-is-flipped-when-opened-as-numpy-array
# ds = gdal.Open('XTORM_v2_tmp-ras_02.tif')
# gt = ds.GetGeoTransform()
# if gt[2] != 0.0 or gt[4] != 0.0: print ('file is not stored with north up')

# DEFINE THE COORDINATES OF THE XY.AXES
    XS, YS = list(map( lambda a,b,c: np.arange(a.item() +c/2, b.item() +c/2, c),
                      [llim,blim],[rlim,tlim],[X_RES,Y_RES] ))
# flip YS??
    YS = np.flipud( YS )      # -> important...so rasters are compatible with numpys



    # pd.DataFrame(CATCHMENT_MASK).to_csv('zcatch.csv',sep=' ',header=False,index=False)
    # #CATCHMENT_MASK.sum() == len(gagnet)
    # pd.DataFrame(CATCHMENT_MASK).to_csv('zgatch.csv',sep=' ',header=False,index=False)




#- N-RANDOM SAMPLES FROM 'ANY' GIVEN PDF ---------------------------------------
#-------------------------------------------------------------------------------
def RANDOM_SAMPLING( PDF, N ):
# PDF: scipy distribution_infrastructure (constructed PDF)
# N  : number of (desired) random samples
    xample = PDF.rvs( size=N )
    # # for reproducibility
    # xample = PDF.rvs( size=N, random_state=npr.RandomState(npr.Philox(12345)) )
    # xample = PDF.rvs( size=N, random_state=npr.RandomState(npr.PCG64DXSM(1337)) )
    # xample = PDF.ppf( npr.RandomState(777).random( N ) ) # -> TOTALP for Matlab 'compatibility'
    return xample


#- RETRIEVE TOTAL SEASONAL/MONSOONAL -------------------------------------------
#-------------------------------------------------------------------------------
def SEASONAL_RAIN( PDF, seas, BAND='', N=1 ):#PDF=TOTALP
# sample N values of TOTALP & transform them from ln-space
    total = np.exp( RANDOM_SAMPLING( PDF[ seas ][ BAND ], N ) )
# upscale (sampled) median rainfall to climatic/scaling factors
#--multiplying (elemnent-wise, regardless the COEFF.size??) the "Ptot_ann_global" vector by the SUM/MIX of all coefficients
    return total #* (1 + PTOT_SC[ seas ] + PTOT_SF[ seas ])


#- TRUNCATED N-RANDOM SAMPLES FROM 'ANY' GIVEN PDF -----------------------------
#-------------------------------------------------------------------------------
def TRUNCATED_SAMPLING( PDF, LIMITS, N ):#PDF=RADIUS[seas][''],#LIMITS=[1,None] #PDF=BETPAR[seas]['']
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


# #- SAMPLE FROM A COPULA & "CONDITIONAL" I_MAX-AVG_DUR PDFs ---------------------
# #-------------------------------------------------------------------------------
# def COPULA_SAMPLING( COP, seas, N, BAND='' ):#COP=COPULA;BAND=''
# # create the copula & sample from it
#     IntDur = GaussianCopula(corr=COP[ seas ][ BAND ], k_dim=2).rvs( nobs=N )
#     # # for reproducibility
#     # IntDur = GaussianCopula(corr=COP[ seas ][ BAND ], k_dim=2).rvs(nobs=N, random_state=npr.RandomState(npr.PCG64(20220608)))
#     return MAXINT[ seas ][ BAND ].ppf( IntDur[:, 0] ), AVGDUR[ seas ][ BAND ].ppf( IntDur[:, 1] )

#- SAMPLE FROM A COPULA & "CONDITIONAL" I_MAX-AVG_DUR PDFs ---------------------
#-------------------------------------------------------------------------------
def COPULA_SAMPLING( COP, seas, BAND='', N=1 ):#COP=COPULA;BAND=''
# create the copula & sample from it
# https://stackoverflow.com/a/12575451/5885810  (1D numpy to 2D)
# (-1, 2) -> because 'GaussianCopula' will always give 2-cols (as in BI-variate copula)
# the 'reshape' allows for N=1 sampling
    IntDur = GaussianCopula(corr=COP[ seas ][ BAND ], k_dim=2).rvs( nobs=N ).reshape(-1, 2)
    # # for reproducibility
    # IntDur = GaussianCopula(corr=COP[ seas ][ BAND ], k_dim=2).rvs(nobs=N, random_state=npr.RandomState(npr.PCG64(20220608))).reshape(-1, 2)
    return MAXINT[ seas ][ BAND ].ppf( IntDur[:, 0] ), AVGDUR[ seas ][ BAND ].ppf( IntDur[:, 1] )



#- DEFINE THE DAYS OF THE SEASON (to 'sample' from) ----------------------------
#-------------------------------------------------------------------------------
def WET_SEASON_DAYS():
    global SEED_YEAR, M_LEN, DATE_POOL, DATE_ORIGIN
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
# populate the limits with daily.datetimes
    DATE_POOL = [None if v else \
        pd.date_range(d.__getitem__(0), d.__getitem__(-1), freq='D', tz=TIME_ZONE) \
            for d,v in zip(DATE_POOL, mvoid)]
# convert DATE_ORIGIN into 'datetime' (just to not let this line hanging out all alone)
# https://stackoverflow.com/q/70460247/5885810  (timezone no pytz)
# https://stackoverflow.com/a/65319240/5885810  (replace timezone)
    DATE_ORIGIN = datetime.strptime(DATE_ORIGIN, '%Y-%m-%d').replace(
        tzinfo=ZoneInfo(TIME_ZONE))



#- SAMPLE TIMES.OF.DAY (either NORMAL (default) or CIRCULAR --------------------
#-------------------------------------------------------------------------------
def TOD_SAMPLING( POOL, N, VMF, simy ):#POOL=DATE_POOL[0],VMF=DATIME[0]
# here not only is uniformly sample the Day.Of.Season but also the Time.Of.Day
    sample = interp1d(np.linspace(0, 1, num=len(POOL), endpoint=True),
        range(len(POOL)), kind='linear', axis=0)( npr.uniform(0, 1, N) )
    # # for reproducibility
    #     range(len(POOL)), kind='linear', axis=0)( npr.RandomState(666).uniform(0, 1, N) )
#    dates = POOL[ list(map(int, sample)) ] + relativedelta(years=simy)
    dates = list(map(lambda d: d + relativedelta(years=simy),
                     POOL[ list(map(int, sample)) ]))
    times = ( sample - list(map(int, sample)) ) *24
    if type(VMF) is dict:
# sampling from MIXTURE.of.VON_MISES-FISHER.distribution
        times = vonmises.tools.generate_mixtures(p=VMF['p'], mus=VMF['mus'],
                                                 kappas=VMF['kappas'], sample_size=N)
# from radians to decimal HH.HHHH
        times = (times +np.pi) /(2*np.pi) *24
    # # to check out if the sampling is done correctly
    # plt.hist(times, bins=24)
# SECONDS since DATE_ORIGIN
    stamps = list(map(lambda d,t:
        ((d + timedelta(hours=t)) - DATE_ORIGIN).total_seconds(), dates, times))
# # pasting and formatting
# # https://stackoverflow.com/a/67105429/5885810  (chopping milliseconds)
#     stamps = list(map(lambda d,t: (d + timedelta(hours=t)).isoformat(timespec='seconds'),
#                       dates, times))
    return np.round(stamps, 0).astype('u8')     # i root for .astype('u4') instead
    # return np.sort(stamps)



# OUTER RING/POLYGON
# r[0] is where the maximus radius lays/reside
# *1e3 to go from km to m
def LAST_RING( all_radii, CENTS ):
    ring_last = list(map(lambda c,r: gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(x=[c[0]], y=[c[1]] ).buffer(
            # r[0] *1e3, resolution=int((3 if r[0] < 1 else 2)**np.ceil(r[0] /2)) ),
            r *1e3, resolution=int((3 if r < 1 else 2)**np.ceil(r /2)) ),
        crs=f'EPSG:{WGEPSG}'), CENTS, all_radii))
    return ring_last




def ZTRATIFICATION( Z_OUT ):#Z_OUT=pd.concat( RINGO )
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
        qants = pd.DataFrame( {'E':'', 'median':len(Z_OUT)}, index=[0] )
        # ztats = pd.DataFrame( {'E':np.repeat('',len(Z_OUT))} )
        ztats = pd.Series( range(len(Z_OUT)) )      # 5x-FASTER! than the line above
    return qants, ztats



#- CREATE CIRCULAR SHPs (RINGS & CIRCLE) & ASSING RAINFALL TO C.RINGS ----------
#-------------------------------------------------------------------------------
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
# int((3 if r < 1 else 2)**np.ceil(r /2)) is an artifact to lower the resolution of small circles
# ...a lower resolution in such circles increases the script.speed in the rasterisation process.
    rain_ring = list(map(lambda c,r,p: pd.concat( list(map(lambda r,p: gpd.GeoDataFrame(
        {'rain':p, 'geometry':gpd.points_from_xy( x=[c[0]], y=[c[1]] ).buffer(
            r *1e3, resolution=int((3 if r < 1 else 2)**np.ceil(r /2)) ).boundary},
        crs=f'EPSG:{WGEPSG}') , r, p)) ), CENTS, all_radii, all_rain))
# # the above approach (in theory) is much? faster than the list.comprehension below
#     rain_ring = [pd.concat( gpd.GeoDataFrame({'rain':p, 'geometry':gpd.points_from_xy(
#         x=[c[0]], y=[c[1]] ).buffer(r *1e3, resolution=int((4 if r < 1 else 2)**np.ceil(r)) ).boundary},
#         crs=f'EPSG:{WGEPSG}') for p,r in zip(p,r) ) for c,r,p in zip(CENTS, all_radii, all_rain)]
# # the above is the line.comprehension of the code below
#     rain_ring = []
#     for s in range(len(CENTS)):
#         pts = [gpd.GeoDataFrame({'rain':all_rain[s][r],
#             'geometry':gpd.points_from_xy(x=[CENTS[s,0]], y=[CENTS[s,1]]).buffer(
#                 rtem *1e3, resolution=int((4 if r < 1 else 2)**np.ceil(rtem)) ).boundary},
#             crs=f'EPSG:{WGEPSG}') for r, rtem in enumerate( all_radii[s] )]
#         rain_ring.append( pd.concat(pts) )

    return rain_ring


#- RASTERIZE SHPs & INTERPOLATE RAINFALL (between rings) -----------------------
#-------------------------------------------------------------------------------
def RASTERIZE(ALL_RINGS, OUTER_RING):#posx=32#9#-1 #ALL_RINGS=rain_ring[posx];OUTER_RING=last_ring[posx]
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



# # VISUALISATION
# # import matplotlib.pyplot as plt
# # from matplotlib.patches import Circle
# fig, ax = plt.subplots(figsize=(7,7), dpi=150)
# ax.set_aspect('equal')
# #plt.imshow(fall, interpolation='none', aspect='equal', origin='upper', cmap='gist_ncar_r',# alpha=0,
# plt.imshow(STORM_MATRIX[0], interpolation='none', aspect='equal', origin='upper', cmap='gist_ncar_r',# alpha=0,
#             extent=(llim[0], rlim[0], blim[0], tlim[0]))
# # Now, loop through coord arrays, and create a circle at each x,y pair
# posx = 0#32#9
# for rr in all_radii[posx] *1e3:
#     circ = Circle((CENTS[posx][0], CENTS[posx][1]), rr, alpha=1, facecolor='None',
#                   edgecolor=npr.choice(['xkcd:lime green','xkcd:gold','xkcd:electric pink','xkcd:azure']))
#     ax.add_patch(circ)
# #plt.show()
# plt.savefig('tmp_ras_22.jpg', bbox_inches='tight',pad_inches=0.02, facecolor=fig.get_facecolor())
# plt.close()
# plt.clf()

# # plt.imshow(da.mask, interpolation='none', aspect='equal', origin='upper', cmap='gist_ncar_r', extent=(llim[0], rlim[0], blim[0], tlim[0]))
# # plt.imshow(CATCHMENT_MASK, interpolation='none', aspect='equal', origin='upper', cmap='gist_ncar_r', extent=(llim[0], rlim[0], blim[0], tlim[0]))



def RAIN_CUBE( STORM_MATRIX, DATES, MRAIN, ragg ):#ragg=0
    data = xr.DataArray(data=STORM_MATRIX, dims=['time','row','col'])
    data.coords['time'] = DATES
    #data.coords['time'] = pd.to_datetime(DATES, unit='s', origin=DATE_ORIGIN.replace(tzinfo=None))#.tz_localize(tz=TIME_ZONE)
# unless you end up with more than 256.gauges per pixel... leave it as 'u1'
    data.coords['mask'] = (('row','col'), CATCHMENT_MASK.astype('u1'))
# step-cumulative rainfall
    dagg = data.where(data.mask!=0, np.nan).cumsum(dim='time', skipna=False)
# if two consecutive aggretated fields are the same -> no storm fell within AOI
    void = dagg.sum(dim=('row','col'), skipna=True)
    void = np.where(np.diff(void) == 0).__getitem__(0) +1
# which aggretation(s) surpass MRAIN?
# # the line below is enough for the SIMULATION.scenario
#     suma = dagg.median(dim=('row','col'), skipna=True) + ragg
# the line below allows SIMULATION & VALIDATION!!... basically, rainfall.depths are
# duplicated/replicated by the 'MASK.frequency', and then the MEDIAN is computed.
# https://stackoverflow.com/a/57889354/5885810  (apply xr.apply_ufunc)
# https://stackoverflow.com/a/35349142/5885810  (weighted median)
    suma = xr.apply_ufunc(lambda x: np.median(np.repeat(np.ravel(x), np.ravel(data.mask)))\
        ,dagg ,input_core_dims=[['row','col']], vectorize=True, dask='allowed') + ragg
    xtra = np.where( suma >= MRAIN ).__getitem__(0)
# REMOVING VOID FIELDS
    drop = np.union1d(void, xtra[1:])
    data = data.drop_isel( time = drop )
# finds whether MRAIN is reached (or not) in this (a given) iteration
    #ends = 1 if xtra.size >= 1 else 0
    ends = np.delete(suma, drop).values.__getitem__(-1)
    # # do we want to output aggregated fields?
    # dagg = dagg.drop_isel( time = drop )
    # return ends, data, drop, dagg
    return ends, data, drop


    # plt.imshow(data[19,:,:].data, interpolation='none', aspect='equal', origin='upper', cmap='gist_ncar_r', extent=(llim[0], rlim[0], blim[0], tlim[0]))
    # plt.imshow(CATCHMENT_MASK, interpolation='none', aspect='equal', origin='upper', cmap='gist_ncar_r', extent=(llim[0], rlim[0], blim[0], tlim[0]))



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
                                ,chunksizes=CHUNK_3D( [ len(YS) ], valSize=4))
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
        ,zlib=True, complevel=9)#,fill_value=np.r_[0].astype('u4'))
    ncxtra.long_name = 'storm duration'
    ncxtra.units = 'minute'
    ncxtra.precision = f'{1/60}'# (1 sec)'
    # ncxtra.scale_factor = dur_SCL
    # ncxtra.add_offset = dur_ADD

    return sub_grp#, yy.getncattr('coordinates'), xx.getncattr('coordinates')



def NC_FILE_II( sub_grp, simy, KUBE, XTRA ):#KUBE=q_ain[ idx_s ];XTRA=np.concatenate(l_ong)[ idx_s ]
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



#%% COLLECTOR




def STORM( NC_NAMES ):

    global cut_lab, cut_bin

# storm minimum radius depends on spatial.resolution (for raster purposes)
# ...it must be used/assigned in KM, as its distribution was 'computed' in KM
    MIN_RADIUS = np.max([X_RES, Y_RES]) /1e3


# define transformation constants/parameters for INTEGER rainfall-NC-output
    NCBYTES_RAIN( INTEGER )


    READ_PDF_PAR()
    CHECK_PDF()
    SHP_OPS()
    WET_SEASON_DAYS()

    if Z_CUTS:
        cut_lab = [f'Z{x+1}' for x in range(len(Z_CUTS) +1)]
        cut_bin = np.union1d(Z_CUTS, [0, 9999])


    # # https://gis.stackexchange.com/a/267326/127894     (get EPSG/CRS from raster)
    # from osgeo import osr
    # tIFF = gdal.Open( DEM_FILE )
    # tIFF = gdal.Open( './data_WG/dem/WGdem_WGS84.tif' )
    # tIFF_proj = osr.SpatialReference( wkt=tIFF.GetProjection() ).GetAttrValue('AUTHORITY', 1)
    # tIFF = None


#%%

    print('\nRUN PROGRESS')
    print('************\n')

    for seas in range( SEASONS ):#seas=0


# CRETE NC.FILE
        #ncid = 'zomeFILEname100.nc'
        ncid = NC_NAMES[ seas ]
        nc = nc4.Dataset(ncid, 'w', format='NETCDF4')
        nc.created_on = datetime.now(tzlocal()).strftime('%Y-%m-%d %H:%M:%S %Z')#%z

        print(f'SEASON: {seas+1}/{SEASONS}')

        for nsim in range( NUMSIMS ):#nsim=0

            print(f'\t{MODE.upper()}: {"{:02d}".format(nsim+1)}/{"{:02d}".format(NUMSIMS)}')#, end='', flush=False)
# 1ST FILL OF THE NC.FILE
            #sub_grp, tag_y, tag_x = NC_FILE_I( nc, nsim )
            sub_grp = NC_FILE_I( nc, nsim )


            #for simy in range( NUMSIMYRS ):#simy=0
            for simy in tqdm( range( NUMSIMYRS ), ncols=50 ):

                #print(f'\tSIMUL: {nsim}  ->  YEAR {simy}...')

                MRAIN = SEASONAL_RAIN( TOTALP, seas )

                # # PUT SOME MRAIN = 0 (OUTSIDE THIS LOOP)
                # MRAIN = MRAIN * kron['PTOT'][ seas ] +\
                #     SEASONAL_RAIN( TOTALP, seas ) * (1 + PTOT_SC[ seas ] + PTOT_SF[ seas ])

                MRAIN = SEASONAL_RAIN( TOTALP, seas ) * (1 + PTOT_SC[ seas ] + ((nsim +1) *PTOT_SF[ seas ]))

                #MRAIN = np.array([99])

                r_ain = []
                l_ong = []

                #NUM_S = 150       # number of storms initial seed
# for the WGc-case we start with (~30 days/month * 5 months (wet season duration)) = 150
# ...hence, we're assuming that (initially) we have 1 storm/day (then we continue 'half-fing' this seed)
                NUM_S = 40 * M_LEN[ seas ].__getitem__(0)
                CUM_S = 0

                while CUM_S < MRAIN and NUM_S >= 2:

                    # npr.RandomState(npr.Philox(54321))
                    # # the above generator is NOT equal to all the ones below
                    # # all the below generators "fall back" to the same Initial.State / Seed
                    # npr.RandomState(npr.Philox(npr.seed(54321)))
                    # npr.RandomState(npr.MT19937(npr.seed(54321)))
                    # npr.RandomState(npr.seed(54321))
                    # npr.seed(54321)
                    CENTS = poisson( BUFFRX.geometry.xs(0), size=NUM_S )
                    # https://stackoverflow.com/a/69630606/5885810  (rnd pts within shp)

# # plot
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# fig = plt.figure(figsize=(10,10), dpi=300)
# ax = plt.axes(projection=ccrs.epsg(WGEPSG))
# ax.set_aspect(aspect='equal')
# for spine in ax.spines.values(): spine.set_edgecolor(None)
# fig.tight_layout(pad=0)
# #plt.figure()
# buffrX.plot(edgecolor='xkcd:amethyst', alpha=1., zorder=2, linewidth=.77, ls='dashed', facecolor='None', ax=ax)
# plt.scatter(storm_centre[:,0], storm_centre[:,1], marker='P', s=37, edgecolors='none')
# #plt.show()
# plt.savefig('XTORM_v2_cpoisson.png', bbox_inches='tight',pad_inches=0.00, facecolor=fig.get_facecolor())
# plt.close()
# plt.clf()



                    # # if FORCE_BRUTE was used -> truncation deemed necessary to avoid
                    # # ...ULTRA-high intensities [chopping almost the PDF's 1st 3rd]
                    # BETAS = TRUNCATED_SAMPLING( BETPAR[ seas ][''], [-0.008, +0.078], NUM_S )
                    BETAS = RANDOM_SAMPLING( BETPAR[ seas ][''], NUM_S )
                    # [BETPAR[ seas ][''].cdf(x) if x else None for x in [-.035, .035]]


                    RADII = TRUNCATED_SAMPLING( RADIUS[ seas ][''], [1* MIN_RADIUS, None], NUM_S )
                    # RADII = TRUNCATED_SAMPLING( RADIUS[seas][''], [0, None], NUM_S )



                    #RINGO = pd.concat( LAST_RING( RADII, CENTS ) )
                    RINGO = LAST_RING( RADII, CENTS )

# define pandas to split the Z_bands (or not)
                    qants, ztats = ZTRATIFICATION( pd.concat( RINGO ) )
# compute copulas given the Z_bands (or not)
                    MAX_I, DUR_S = list(map(np.concatenate, zip(* qants.apply( lambda x:\
                        COPULA_SAMPLING(COPULA, seas, x['E'], x['median']), axis='columns') ) ))
# sort back the arrays
                    MAX_I, DUR_S = list(map( lambda A: A[ np.argsort( ztats.index ) ], [MAX_I, DUR_S] ))

                    # MAX_I, DUR_S = COPULA_SAMPLING( COPULA, seas, NUM_S, BAND='' )

                    # IF THERE IS A TRUNCATION IN DURATION... GET RID OF IT HERE
                    # ...AND UPDATE (POSTERIOR & ANTERIOR) LENGTHS OF ARRAYS!!
                    MAX_I = MAX_I * (1 + STORMINESS_SC[ seas ] + ((nsim +1) *STORMINESS_SF[ seas ]))



                    DATES = TOD_SAMPLING( DATE_POOL[ seas ], NUM_S, DATIME[ seas ], simy )

                    # RINGS, RINGO = LOTR( RADII, MAX_I, DUR_S, BETAS, CENTS )
                    RINGS = LOTR( RADII, MAX_I, DUR_S, BETAS, CENTS )
                    # COLLECTING THE STORMS
                    STORM_MATRIX = list(map(RASTERIZE, RINGS, RINGO))

                    CUM_S, rain, remove = RAIN_CUBE( STORM_MATRIX, DATES, MRAIN, CUM_S )

                    r_ain.append( rain )
                    l_ong.append( np.delete(DUR_S, remove) )

                    NUM_S = int(NUM_S /2)
                    #NUM_S = int(NUM_S /1.5)

                assert not (CUM_S < MRAIN and NUM_S < 2), 'Iteration for SIMULATION '\
                    f'{nsim}: YEAR {simy} not converging!\nTry a larger initial '\
                    'seed (i.e., variable "NUM_S"). If the problem persists, it '\
                    'might be very likely that the catchment (stochastic) '\
                    'parameterization is not adequate.' # the more you increase the slower it gets!!

                q_ain = xr.concat(r_ain, dim='time')

                idx_s = q_ain.time.argsort().data       # sort indexes

                NC_FILE_II( sub_grp, simy, q_ain[ idx_s ], np.concatenate(l_ong)[ idx_s ] )#KUBE=q_ain[ idx_s ];XTRA=np.concatenate(l_ong)[ idx_s ]

        nc.close()


#%% CALL THE MODULE(S)

if __name__ == '__main__':
    STORM( ['zomeFILEname7_testMAIN.nc'] )  # testing for only ONE Season!