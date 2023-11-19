#%% SOFT-CORE PARAMETERS

"""
Parameters in this block define the type, (temporal) extend, and a couple of
climatic/meterological conditions of the desired RUN.
The parameters set up here will be the default input of STORM.
You can either modify/tweak them here (thus avoiding passing them again when
running STORM from the command line) or passing/defining them righ from the
command line whe running STORM.
For an 'in-prompt' help (on these parameters) type:
    "python storm.py -h"    (from your CONDA environment or Terminal)
    "%%python storm.py -h"  (from your Python console)
"""

MODE = 'SImuLAtiON'     # Type of Run (case-insensitive). Either 'SIMULATION' or 'VALIDATION'
# MODE = 'valiDaTION'     # Type of Run (case-insensitive). Either 'SIMULATION' or 'VALIDATION'
SEASONS = 2             # Number of Seasons (per Run)
NUMSIMS = 2             # Number of runs per Season
NUMSIMYRS = 3           # Number of years per run (per Season)

"""
PTOT_SC       = Signed scalar specifying the step change in the observed wetness (TOTALP)
PTOT_SF       = Signed scalar specifying the progressive trend in the observed wetness
STORMINESS_SC = Signed scalar specifying the step change in the observed storminess
STORMINESS_SF = Signed scalar specifying the progressive trend in the observed storminess
*** all scalars must be specified between 0 and 1 (i.e., 100%) ***
*** a (un-)signed scalar implies stationary conditions akin to (current) observations ***
"""

# # PARAMETER   = [ S1 ]
# PTOT_SC       = [0.00]
# PTOT_SF       = [ 0.0]
# STORMINESS_SC = [-0.0]
# STORMINESS_SF = [+0.0]

# if you intend to model more than 1 SEASON, YOU CAN DO (e.g.):
# PARAMETER   = [ S1 ,  S2 ]
PTOT_SC       = [ 0. , - .0]
PTOT_SF       = [+0.0, -0. ]
STORMINESS_SC = [ 0.0, + .0]
STORMINESS_SF = [-0.0,  0.0]
# # ...or (e.g.):
# # PARAMETER   = [ S1 ,  S2 ]
# PTOT_SC       = [0.15, None]
# PTOT_SF       = [ 0.0, None]
# STORMINESS_SC = [-0.1, None]
# STORMINESS_SF = [ 0.0, None]


#%% HARD-CORE PARAMETERS

"""
Parameters in this block define the input and output files (paths), and the
spatio-temporal characteristics of the domain over which STORM will run.
Unlike the parameters set up in the previous block, these parameters cannot
be passed from the command line. Therefore, their modification/tweaking must
carried out here.
"""

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
"""
*   GAG_FILE is only required for 'validation' runs
**  DEM_FILE is only required for runs at different altitudes, i.e., Z_CUTS != None
*** Having the DEM_FILE in the local CRS could contribute to a faster run,
    although we didn't find staggering differences in both approaches.
    Still, if the preferred option is a local-CRS DEM, switch ON/OFF the line(s):
    'zonal_stats(vectors=Z_OUT.geometry, raster=DEM_FILE, stats=Z_STAT) in 'ZTRATIFICATION'.
"""

# Z_CUTS = None           # (or Z_CUTS = []) for INT-DUR copula modelling regardless altitude
Z_CUTS = [1350, 1500]   # in meters!
Z_STAT = 'median'       # statistic to retrieve from the DEM ['mean' or 'min'|'max'?? not 'count']
"""
Z_CUTS = [1350, 1500] are from the analysis carried out in the 'pre_processing.py' module.
They imply 3 altitude-bands, namely, [0, 1350), [1350, 1500), [1500, 9999), for which the
Intensity-Duration copulas were established.
Hence, modyfing this variable without a copula (re-)analysis, for the desired/updated
bands, will still yield results!; nevertheless, such results won't be representative
of the parametric functions/statistics found in '.../ProbabilityDensityFunctions.csv'.
"""

WGEPSG    = 26912       # EPSG Code of the local/regular Coordinate Reference System (CRS)
X_RES     = 1000        # in meters! (for the 'regular/local' CRS)
Y_RES     = 1000        # in meters! (for the 'regular/local' CRS)
BUFFER    = 5000        # in meters! -> buffer distance (out of the catchment)
CLOSE_DIS = 0.15        # in km -> small circle emulating the storm centre's point/dot
# storm minimum radius depends on spatial.resolution (for raster purposes);
# it must be used/assigned in KM, as its distribution was 'computed' in KM
MINRADIUS =  max([X_RES, Y_RES]) /1e3
# distance between (rainfall) rings; heavily dependant on X_Y_RES | MINRADIUS
RINGS_DIS =  MINRADIUS *(2) +.1         # in km -> distance between (rainfall) rings; heavily dependant on X_Y_RES
"""
BUFFER extends the catchment boundary (some given distance), thus delimiting the area
for which the storm centres are generated (within).
The extension (bounding-box) of this 'buffer' defines too the limits of the rainfall
fields, namely, the Area Of Interest (aoi).
STORM generates circular storms reaching maximum intensity at their centres, and
decaying towards a maximum radius. Hence, intermediate intensities are calculated
for different radii between the storm's centre and its maximum radius. RINGS_DIS is the
separation between these different radii; whereas CLOSE_DIS just simulates the storm's
centre out of a very small circle.
Once the spatial domain of the storm is populated by 'rings-of-rainfall', STORM fills
the voids in between by linear interpolation.
"""

MIN_DUR = 2             # in minutes!
MAX_DUR = 60*24*5       # in minutes! -> 5 days (in this case)
# # OR:
# MIN_DUR = []          # use 'void' arrays if you want NO.CONSTRAINT on storm-duration
# MAX_DUR = []          # ... in either (or both) MIN_/MAX_DUR parameters/constants
"""
MAX_DUR and MIN_DUR constraints the storm-duration of the sampled pairs
from the intenstity-duration copula.
"""

### these parameters allow to pin down a time-dimension to the storms
# SEED_YEAR  = None                         # for your SIM/VAL to start in the current year
SEED_YEAR    = 2000                         # for your SIM/VAL to start in 2050
### bear in mind the 'SEASONS' variable!... (when toying with 'SEASONS_MONTHS')
# SEASONS_MONTHS = [[6,10], None]             # JUNE through OCTOBER (just ONE season)
# # OR:
# SEASONS_MONTHS = [[10,5], ['jul','sep']]  # OCT[y0] through MAY[y1] (& JULY[y1] through SEP[y1])
SEASONS_MONTHS = [['may','sep'],[11,1 ]]  # MAY through SEP (& e.g., NOV trhough DEC)
TIME_ZONE      = 'US/Arizona'               # Local Time Zone (see links below for more names)
# # OR:
# TIME_ZONE    = 'UTC'
# # https://stackoverflow.com/a/64861179/5885810    (zoneinfo list)
# # https://pynative.com/list-all-timezones-in-python/#h-get-list-of-all-timezones-name
# # https://www.timeanddate.com/time/map/
# # https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
DATE_ORIGIN    = '1950-01-01'               # to store dates as INT
"""
SEASONS_MONTHS = [[6,10], None] (equal to Z_CUTS) corresponds to the period (and season!)
for which all the parametric functions found in '.../ProbabilityDensityFunctions.csv' were
computed. Hence, when you modify this variable (for whatever your needs), please carry out
all the respective (stochastic) analyses correspond for the period you want to model.
"""

### only touch this parameter if you really know what you're doing ;)
INTEGER = 2     # number of (unsigned) Bytes (2, 4, 6 or 8) to store the RAINFALL variable (into)