import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from datetime import datetime, timedelta
from scipy import optimize, stats
from scipy.spatial import distance
from fitter import Fitter
from statsmodels.distributions.copula.api import GaussianCopula
#from statsmodels.distributions.empirical_distribution import ECDF
from parameters import Z_CUTS, WGEPSG

""" STORM2.0 ALSO runs WITHOUT this library!!! """
import vonMisesMixtures as vonmises

#~ INSTALLING THE vonMisesMixtures PACKAGE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
"""
# https://framagit.org/fraschelle/mixture-of-von-mises-distributions
first try this(in the miniconda/anacoda prompt):

    pip install vonMisesMixtures

...but as of July 21, 2022, it didn't work (not available in PIP no more?)
otherwise, proceed as the above link suggests:

    git clone https://framagit.org/fraschelle/mixture-of-von-mises-distributions.git
    cd mixture-of-von-mises-distributions/
    pip install .

do the 'cloning' in your library environment, i.e.,
path-to-miniconda3//envs/prll/lib/python3.10/site-packages  (linux)
path-to-miniconda3\\envs\py39\Lib\site-packages             (windows)
that's it you're all set now!
"""
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# https://stackoverflow.com/a/20627316/5885810
pd.options.mode.chained_assignment = None  # default='warn'
tqdm.pandas(ncols=50)#, desc="progress-bar")


#%% PARAMETERS

#~ PATHS TO INPUT/OUTPUT FILES & GLOBAL VARIABLES DEFINITION ~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# INPUT FILES
AGGRE_DATA = './model_input/data_WG/gage_data--1953Aug-1999Dec_aggregateh--ANALOG.csv'
EVENT_DATA = './model_input/data_WG/gage_data--1953Aug18-1999Dec29_eventh--ANALOG.csv'
GAUGE_META = './model_input/data_WG/gage_data--gageNetworkWG--ANALOG.csv'
# AGGRE_DATA = './model_input/data_WG/gage_data--1953Aug-2022Dec_aggregateh.csv'
# EVENT_DATA = './model_input/data_WG/gage_data--1953Aug18-2022Dec14_eventh.csv'
# GAUGE_META = './model_input/data_WG/gage_data--gageNetworkWG.csv'

# OUPUT FILES
OUPUT_FILE = './model_input/ProbabilityDensityFunctions_TWO--ANALOG.csv'
# OUPUT_FILE = './model_input/ProbabilityDensityFunctions_TWO.csv'

# if you want to redefine Z_CUTS here
Z_CUTS = [1350, 1500]
# # for no INT-DUR copula at different altitudes
# Z_CUTS = []
# # Z_CUTS = None

# global variables
# distros commonly attribute to precipitation
PTOT_PDFS = ['argus','betaprime','burr','burr12','chi','chi2','exponweib','exponpow'
             ,'gausshyper','gengamma','genhalflogistic','gumbel_l','gumbel_r','invgauss'
             ,'invweibull','johnsonsb','johnsonsu','ksone','kstwobign','loggamma'
             ,'maxwell','nakagami','ncx2','norm','powerlognorm','rayleigh','rice'
             ,'tukeylambda','weibull_min','weibull_max']

# these distros work for either AVG.DUR and/or BETA (BRUTE_FORCE)
RSKW_PDFS = ['alpha','betaprime','burr','f','fisk','gamma','geninvgauss','invgamma'
             ,'invweibull','johnsonsb','johnsonsu','ksone','lognorm','mielke'
             ,'moyal','norm','powerlognorm','rayleigh','rice','wald','weibull_min']

IMAX_PDFS = ['truncexpon','pareto','loguniform','lomax','halfgennorm','genpareto'
             ,'expon']

# these heavily?-"gaussian" distributions work for the BETA (FORCE-BRUTE)
NORM_PDFS = ['cauchy','cosine','exponnorm','gumbel_r','gumbel_l','hypsecant'
             ,'laplace','logistic','moyal','norm','powernorm']

# discrete 'sensical' distributions to try on DOY.data
DISC_PDFS = ['nbinom','nhypergeom','betabinom']#,'poisson']
# the boundaries for parameter.estimation
disc_bnds = [((0,366),(0,1)), ((0,366*2),(0,1e3),(0,100)), ((0,366*2),(0,100),(0,100))]#, ((0,366),)]
param_sel = 2                                       # 1 for "nllf"  |  2 for "BIC"
"""
The "POISSON" distribution gave the best stats (either NegLogLik & BIC);
nevertheless, it was left out as it doesn't generate values close to the season.limits.
Hence we prefer the use of the other three functions, and the posterior
    truncation (within the season.limits) of the sampled values
"""

# use 'VMF' is the DOY (Day of Year) is fitted via Circular.Statistics (comment it out OTHERWISE!)
USE_DOY = 'VMF'

# to censor gauge.data until the network reaches "uniformity"
TRIM_YEAR = 1963


#%% FUNCTIONS' DEFINITION

#~ CREATE THE OUTPUT FILE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def OUTPUT_PATH():
    folder = '/'.join( OUPUT_FILE.split('/')[:-1] )
    try:
        with open(OUPUT_FILE, 'w'):
# create an empty file (https://www.geeksforgeeks.org/create-an-empty-file-using-python/)
            pass
    except FileNotFoundError:
        print(f"for some reason the folder '{folder}' does not exist!!.\n"\
              'please, create such a folder... and come back!.')


#~ READ GAUGE META-DATA ('core_n' PREREQUISITE) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def READ_META_DATA():
    try:
        global meta_gage
        meta_gage = pd.read_table(GAUGE_META, sep=',', header='infer', comment="#")
# https://stackoverflow.com/a/10588342/5885810
# https://stackoverflow.com/a/10588651/5885810
    except FileNotFoundError:
        print('please assign the correct path to the CSV file containing gauge '\
            'metadata.\nthe current path (GAUGE_META variable) is either '\
            'incorrect or the file does not exist.')
    return meta_gage


#~ READ EVENT-BASED DATA ('core_n' PREREQUISITE) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def READ_DATA_EVENT():
    try:
        one = pd.read_csv(EVENT_DATA, sep=',', header='infer', comment="#",
                          infer_datetime_format=True,
                          dtype={'gage':'category','S':'category'})
    except FileNotFoundError:
        print('please assign the correct path to the CSV file containing STORM data.'\
            '\nthe current path (EVENT_DATA variable) is either incorrect or the'\
            ' file does not exist.')
# create INTENSITY column
    one['int_mmh'] = one.Depth.multiply( 1/one.Duration * 60 )
# [OPTIONAL] transform HOUR into "negative" radians (for Circular Statistics analyses)
    one['hRAD'] = one.hour /24 * 2*np.pi -np.pi
# [OPTIONAL] transform DOY  into "negative" radians (for Circular Statistics analyses)
    one['dRAD'] = (( one.doy -1 + one.hour/24 ) /one.doy.max()) * 2*np.pi -np.pi
# [VERY OPTIONAL] compute #STORMS_PER_DAY
    one['ns_day'] = ( one.loc[:,['gage','year','doy','hour']].groupby(
        by=['gage','year','doy']) ).transform('count')
    return one


#~ GENERIC PDF-FITTING & EXPORTING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def FIT_PDF( DATA, N, NAME, DISTROS ):
# DATA   : either a DICT (with pdf-parameters) or some data to fit
# N      : (either 1 or 2) suffix relating the season for which the pdf is fitted
# NAME   : label (to distinguish among pdfs in the OUPUT_FILE)
# DISTROS: list of pdfs o fit from (void and/or not read whea a DICT is passed)
    if type(DATA) is dict:
        PDFFIT = DATA
    else:
#-FITTING FOR...----------------------------------------------------------------
        best_fit = Fitter(DATA, distributions=DISTROS)
# store the fits
        best_fit.fit()
        # best_fit.summary()
# SELECT THE PARAMETERS FOR THE BEST.FIT (i prefer 'BIC')
        # PDFFIT = best_fit.get_best(method = 'sumsquare_error')
        PDFFIT = best_fit.get_best(method = 'bic')

# # SOME ALTERNATIVES for TOTAL MONSOONAL PRECIPITATION  -> 'gumbel_l' (best.fit)
# best_fit.fitted_param['burr'] # (54.88021744539034, 0.33025499399886504, -0.03036788401546383, 5.643793648684661)
# best_fit.fitted_param['norm'] # (5.362910217624112, 0.3167606398478469)
# stats.gumbel_l(5.51023370002913, 0.266225467459668).rvs(size=1000000).min() #1.84546
# # SOME ALTERNATIVES for STORM RADII  -> 'johnsonsb' (best.fit)
# best_fit.fitted_param['gamma'] # (4.399625814327896, -0.47511306843943046, 1.3991123114790807)
# stats.johnsonsb(1.619923133299649, 1.4819248129140739, -0.5627404643016756, 23.14802080573913).rvs(size=1000000).min() #-0.249
# # SOME ALTERNATIVES for BETA PARAMETER -> for "a * np.exp(-2 * b * x**2)" (FORCE-BRUTE)  -> hypsecant' (best.fit)
# best_fit.fitted_param['laplace'] # (0.00037356816459121903, 0.023265137458131132)
# # SOME ALTERNATIVES for BETA PARAMETER -> for "a * np.exp(-2 * b**2 * x**2)" (BRUTE-FORCE)
# stats.burr(2.351235686277189,0.8505976651919634, -0.0011370351719608585, 0.08377708092591916).stats()
# # (array(0.1041537), array(0.0247543))
# best_fit.fitted_param['norm'] # (0.007439858630402953, 0.08860432831337971)  -> "a * np.exp(-2 * b * x**2)"

# EXPORTING/APPENDING PDF-PARAMETERS (pdf-agnostic)
    with open(OUPUT_FILE, 'a') as f:
# https://stackoverflow.com/a/27638751/5885810
# https://stackoverflow.com/a/56736691/5885810
# https://stackoverflow.com/a/55481809/5885810
# https://stackoverflow.com/a/3590175/5885810
        f.write( f'{NAME}{N}+{ next(iter(PDFFIT.keys())) },' )
# 'unbreakable' string
        f.write( f"{','.join(map(str,[*PDFFIT.get(next(iter(PDFFIT.keys()))).values()]))}\n" )


#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#- TOTAL MONSOONAL PRECIPITATION ------------------------------------- (START) #
#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#~ READ MONTHLY TOTALS (for PTOT purposes) ('PTOT' PREREQUISITE) ~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def READ_DATA_CUMULATIVE():
    try:
        ten = pd.read_csv(AGGRE_DATA, sep=',', header='infer', comment="#",
                          dtype={'gage':'category','S':'category'})
    except FileNotFoundError:
        print('please assign the correct path to the CSV file containing monthly '\
            'rainfall data.\nthe current path (AGGRE_DATA variable) is either '\
            'incorrect or the file does not exist.')
# counting months with 'actual' rainfall (SEASON included)
    ten_count = ten.groupby(by=['gage','year','S']).agg(
        S_c = pd.NamedAgg(column='S', aggfunc='count'),
        rain_c = pd.NamedAgg(column='rain', aggfunc=lambda x:x[x > 0.].count()) )
# find NAN in the RAIN_C column and transform them into 0's
    ten_count = ten_count.where(~ten_count.rain_c.isna(), other=0).astype(
        {'rain_c':'i8'}, copy=True)
# append the above counts to the read dataset
    ten = ten.set_index(keys=['gage','year','S']).join( ten_count )
# drop the DRY season (if you want focus on the WET.season only)
    ten.drop('D', level=2, axis=0, inplace=True)
# CUMULATIVE SUM ('first'/'last' are 'joke'.functions to avoid the 'unique'.hassle)
    ten_cum = ten.groupby(level=['gage','year','S']).agg(
        {'rain':'sum', 'S_c':'first', 'rain_c':'last'} )
    return ten_cum


#~ TWEAK & SLICE SOME MORE THE MONTHLY DATA (to universally use FIT-PDF ~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def MONSOON( total_rain, SEASON, SMAX, N):
# total_rain: the 'ten_cum'-like dataset
# SEASON    : id-label for which the pdf is computed (e.g., 'W1' or 'W2')
# SMAX      : number of months that make up the season (5 for WGc)
    meds = total_rain.loc[(total_rain.index.get_level_values(level='S') == SEASON)\
                          & (total_rain.rain_c == SMAX)].groupby(
                              level=['year'] ).agg( {'rain':'median'} )
# FITTING THE DISTRIBUTION IN THE LOG(e).SPACE!!
# YOU MUST HAVE AN IDEA OF WHAT YOUR DATA LOOKS LIKE...otherwise:
# plt.hist(np.log(meds), bins=50, density=True) ; plt.show()
    FIT_PDF( np.log(meds), N, 'TOTALP_PDF', PTOT_PDFS )

#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#- TOTAL MONSOONAL PRECIPITATION --------------------------------------- (END) #
#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#- RADIUS (AREA) ----------------------------------------------------- (START) #
#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#~ THE MEAN/MAX/WHATEVER DISTANCE FROM THE CENTROID TO THE GIVEN GAGES ~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def X_DIST(gage_list, WHAT):
# WHAT: statistic to compute/append
    coords = meta_gage.loc[meta_gage.gage.isin(gage_list),['X','Y']]
    heigth = meta_gage.loc[meta_gage.gage.isin(gage_list),['Z']]
    cent = coords.mean(axis='index')
    matrix = distance.cdist(coords, np.expand_dims(cent, axis=0), metric='euclidean')
# https://stackoverflow.com/a/43708523/5885810
    return eval(f'matrix.{WHAT}() /1e3'), heigth.mean().__getitem__(0)


#~ GROUP EVENT-DATA TO ESTIMATE RADII, BETA & I_MAX ('decay' PREREQUISITE) ~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def AREA_DATA( one ):
#-STORM.GROUPING (BY UNIQUE TIME.SIGNATURES...mainly)---------------------------
#-...common data.manipulation to both AREA & BETA analyses----------------------
    area = one.copy(deep=True)
    area.drop(area[area.S=='D'].index, inplace=True)
# the network reached its 'current' size by the early 60's; hence...
# ...to 'fairly' compare extensions within the region, focus on storms after the 1960's
    area.drop(index=area[area.year < TRIM_YEAR].index, axis='index', inplace=True)
# gluing dates & dropping (unnecessary columns)
# https://stackoverflow.com/a/63832539/5885810 (faster approach)
# https://stackoverflow.com/a/3590175/5885810
    area['Date'] = list(map(lambda x,y,z:
        f"{datetime.strptime('-'.join(map(str, [x,y])),'%Y-%j').strftime('%Y-%m-%d')}" \
            f"T{timedelta(seconds=z*3600)}" , area.year, area.doy, area.hour))
# dates2datetime & keeping it tight
    area['Date'] = pd.to_datetime( area.Date )
    area = area.set_index('Date').sort_index(ascending=True)
    area = area[['gage','S','Duration','Depth','int_mmh']]
# group by YEAR, MONTH, DAY & TIME, so you can jam storms happening at the very same.time
    area['ndx'] = area.groupby(by=[area.S, area.index]).ngroup()
# # create the compressed/grouped dataset...for STANDALONE STORMS
# https://stackoverflow.com/a/45044756/5885810
    print('compressing & classifying storm data...', end='')
    core = area.groupby(by=['ndx']).agg(
        ngag = pd.NamedAgg(column='gage', aggfunc=list)
        ,nint = pd.NamedAgg(column='int_mmh', aggfunc=list)
        ,numb = pd.NamedAgg(column='gage', aggfunc='count')
        ,davg = pd.NamedAgg(column='Duration',aggfunc='mean')
        ,S = pd.NamedAgg(column='S', aggfunc=lambda x: x.unique().__getitem__(0))
        )
    #[118587 rows x ...] STORMS
    print(' done!')
# reduce to only HAVING.AT.LEAST.2.GAGES (minimum amount to compute a 'credible' radius)
    core_n = core[core.numb >= 2].copy(deep=True)
    #[ 46365 rows x ...] STORMS
# estimate some STORM.RADII
    print('radii estimation...')
    core_n['cmax'], core_n['zavg'] = zip( *core_n.ngag.progress_map(
        lambda x: X_DIST(x, 'max') ))
# https://stackoverflow.com/a/48134659/5885810
    return core_n

#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#- RADIUS (AREA) ------------------------------------------------------- (END) #
#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#- BETA (& I_MAX) ---------------------------------------------------- (START) #
#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#~ MODEL TO FIT "a * np.exp(-2 * b**2 * x**2)" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def BRUTE_FORCE(x, a, b):
    return a * np.exp(-2 * b**2 * x**2)


#~ MODEL TO FIT "a * np.exp(-2 * b * x**2)" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def FORCE_BRUTE(x, a, b):
    return a * np.exp(-2 * b * x**2)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html


#~ RE-ARRANGES RADII & RAIN FROM CLOSER TO FURTHER FROM THE CENTROID ~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def RECESS(gage_list, rain_list):
# gage_list: rain gauges sharing a unique time-stamp
# rain_list: rainfall instensities from the afore-mentioned gauges
    # WGEPSG = 26912 # Walnut-Gulch catchment CRS
    coords = meta_gage.loc[meta_gage.gage.isin( gage_list ),['X','Y']]
    points = gpd.GeoDataFrame( {'rain':rain_list}, geometry=gpd.points_from_xy(
        coords['X'], coords['Y'] ), crs=f'EPSG:{WGEPSG}' )
# for some reason there are OVERLAPED??.GAGES!!
    points.drop_duplicates(subset=['geometry'], keep='first', inplace=True)
# https://stackoverflow.com/a/70088741/5885810
# https://stackoverflow.com/a/64755616/5885810
# https://stackoverflow.com/a/31566601/5885810
    r_km = points.geometry.apply(lambda g: points.dissolve().centroid.distance( g )) / 1e3
# (in km... so BETA is also retrieved in km^-1 or km^-2)
    r_km.sort_values(by=[0], inplace=True)
# https://stackoverflow.com/a/31566644/5885810
    return(list(map(gage_list.__getitem__, r_km.index)), r_km.values.ravel().tolist(),
           list(map(rain_list.__getitem__, r_km.index)) )


#~ DO THE FITTING (& SOME 'CLEANSING') & STORE THE I_MAX ~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def RECESS_DATA( core_n, SEASON ):
# minimum nuber of points/gages (from a centroid) to fit a model
    qmin_gage = 4
# selected dataset
    decay = core_n.loc[(core_n.numb >= qmin_gage) & (core_n.S == SEASON)].copy(deep=True)
    #[11039 rows x ...] STORMS

    print(f'rearranging gauges2centroid for season {SEASON}...')
# compute RECESS (to relate radii & raint...from closest to furthest)
    #recessort = [RECESS(*x) for x in tqdm( tuple(zip(decay.ngag, decay.nint)) )]
# the line below is a bit faster! (& it has tqdm implemented!)...
# https://stackoverflow.com/a/63832539/5885810
# https://stackoverflow.com/a/69818437/5885810
    recessort = list(map(RECESS, tqdm( decay.ngag, ncols=50 ), decay.nint))
# re-vamp the DECAY-pandas
    decay['nint'] = pd.Series(list(zip(*recessort))[ 0], index=decay.index)
    decay['nint'] = pd.Series(list(zip(*recessort))[-1], index=decay.index)
    decay['nrad'] = pd.Series(list(zip(*recessort))[ 1], index=decay.index)

    print(f'Beta & I_max fitting for season {SEASON}...')
# FIT THE MODEL TO A MINIMUM OF 5g+ (~0.5 percentile of DECAY, if 'W1' only)----
# ...stats.percentileofscore(decay.numb, 5) # == 54.339161155901806
# ...initial tests showed that the above model struggles with only 4.gauges
# use the first raint as initial guess (and 0.1 for the BETA)
# BOUND the fit to 3 times the largest observed raint (and a min/max BETA of [-3,3])
# 0.1 ~= decay.nint.map(max).min() *.9 (otherwise try lower values, i.e., 0.07, until it runs)
    mint = .07 # 0.1
    fit_FB = [optimize.curve_fit(
        BRUTE_FORCE
        # FORCE_BRUTE
        , decay.loc[k,'nrad'], decay.loc[k,'nint']
        , p0=(np.r_[mint, decay.loc[k,'nint'].__getitem__(0)].max(), .1)
        # , bounds=([mint, -3], [decay.nint.map(max).max() *3, 3]) )\
        , bounds=([mint, 0], [decay.nint.map(max).max() *3, 3]) )\
            for k in tqdm( decay[decay.numb > qmin_gage].index, ncols=50 )]
#-DO THE CLEANING
# x[1][0,1] (or x[1][1,0]) stores the co-variance of IMAX-BETA
    cov_FB = np.asarray([x[1][0,1] for x in fit_FB])
# (positive) covariances between [0, 5] are the ones that 'make sense'
    var_ok = np.where((cov_FB >= 0) & (cov_FB <=  5)).__getitem__(0)
# capture the 'OK' parameters
    pars = np.asarray( list(zip( *fit_FB ))[0] )[ var_ok ]

# DO THE COPULA FOR THIS 'NEW' iMAX-DATASET-------------------------------------
    max_copula = decay.loc[decay.numb > qmin_gage, ['zavg','numb','davg']].iloc[ var_ok ]
    max_copula['max_int'] = pars[:,0]
    return pars, max_copula

#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#- BETA (& I_MAX) ------------------------------------------------------ (END) #
#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#- INT-DUR COPULA.ANALYSIS (by ELEV) --------------------------------- (START) #
#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def COPULA( max_copula, N ):
    copula = max_copula.copy(deep=True)
# CLASSIFY the different BAND-ELEVATIONS
    if Z_CUTS:
        cut_lab = [f'Z{x+1}' for x in range(len(Z_CUTS) +1)]
        #cut_bin = np.r_[0, Z_CUTS, 9999]
        cut_bin = np.union1d(Z_CUTS, [0, 9999])
# column 'E' classifies all Z's according to the CUTS
        copula['E'] = pd.cut(copula.zavg, bins=cut_bin, labels=cut_lab,
                             include_lowest=True)
# change column names
    copula.rename(columns={'max_int':'Intensity','davg':'Duration'}, inplace=True)
# we first compute/xport int/dur PDFs...regardless Z-BANDS
    FIT_PDF( copula.Intensity, N, 'MAXINT_PDF', IMAX_PDFS )
    FIT_PDF( copula.Duration , N, 'AVGDUR_PDF', RSKW_PDFS )
# CALLING the GAUSSIAN.COPULA LIBRARY
    GCopula = GaussianCopula()
    cop_var = ['Intensity', 'Duration']
# - copulas scale independent?? -
    #rhos = pd.Series({'': GCopula.fit_corr_param( copula[ cop_var ].multiply([1, 1/60], axis='columns') )})
    rhos = pd.Series({'': GCopula.fit_corr_param( copula[ cop_var ] )})
    if Z_CUTS:
        rhos = pd.concat( [rhos, copula.groupby(by=['E']).apply(
            #lambda x: GCopula.fit_corr_param( x.loc[:, cop_var] ))], axis='index' )
            lambda x: GCopula.fit_corr_param( x[cop_var] ))], axis='index' )
    # #--these parameters are to be exported to the PDFs.file
    # rhos # -> I_MAX     rhos  Intensity (NO i_max)
    #       -0.291182           -0.378778
    # _Z1   -0.270638     _Z1   -0.365285
    # _Z2   -0.282422     _Z2   -0.380517
    # _Z3   -0.402781     _Z3   -0.389479
# here we 'parasite' the IF loop to compute/xport Z-BAND (int/dur) PDFs
        copula.groupby(by=['E']).apply( lambda x: FIT_PDF( x['Intensity'],
            f'{N}+{x.E.unique().__getitem__(0)}', 'MAXINT_PDF', IMAX_PDFS ))
        copula.groupby(by=['E']).apply( lambda x: FIT_PDF( x['Duration'] ,
            f'{N}+{x.E.unique().__getitem__(0)}', 'AVGDUR_PDF', RSKW_PDFS ))
# EXPORTING/APPENDING PDF-PARAMETERS (pdf-agnostic)
    with open(OUPUT_FILE, 'a') as f:
        [f.write( f'COPULA_RHO{N}+{x},{rhos[x]}\n' ) for x in rhos.index]

# #-ONLY.USE.ALL.THIS.BELOW...IF.YOU'RE.NOT.USING IMAX.&.ADUR PDFs!!
# # CREATES ALL COPULA MODELS
#     ecdfs = [[ECDF( copula.loc[:,x] )] for x in cop_var]
#     if Z_CUTS:
#         for x in cop_var:
# # 'copula.E.cat.categories' should be equal to 'CUT_LAB'
#             for y in copula.E.cat.categories:
#                 ecdfs[ cop_var.index(x) ].append( ECDF( copula.loc[copula.E==y, x] ) )
# # COMPUTES THE INTENSITY/DURATION ECDFs FOR ALL.DATA
#     for x in cop_var:
#         minmax = copula[x].agg(['min', 'max'])
#         geospa = np.geomspace(start=minmax.loc['min'], stop=minmax.loc['max']/2,
#                               num=211, endpoint=True)
# # https://stackoverflow.com/a/51169549/5885810
#         geospa = np.r_[0, geospa, 10**np.ceil(np.log10(minmax.loc['max']))]
#         ecdist = ecdfs[ cop_var.index(x) ][0]( geospa )
#         dframe = pd.DataFrame(np.c_[geospa, ecdist], columns=[x, 'ecdf'])
# # DOING THE Z-RELATED ECDFs
#         if Z_CUTS:
#             for y in copula.E.cat.categories:
#                 dframe[f'ecdf{y}'] =\
#                     ecdfs[ cop_var.index(x) ][ rhos.index.get_loc(y) ]( geospa )
# # find duplicated ECDFs
# # keeping the first.repeated value...might contribute to lower/smaller duration/intensities
# # the complete opposite might also be true. with 'FIRST' the removing-0-problem is by-passed
#         dframe_true = dframe.apply(lambda x: x.duplicated(keep='first'), axis='index')
# # convert duplicates into NaN & then try-and-drop 'void' rows
#         dframe.where(~dframe_true, np.nan, inplace=True)
#         dframe.dropna(axis='index', how='all', subset=dframe.columns[1:], inplace=True)
# # EXPORTING THE ECDF tables
#         folder = '/'.join( OUPUT_FILE.split('/')[:-1] )
#         out = f'{folder}/CopulasECDF{N}_{x.lower()}.csv'
#         with open(out, 'w') as f:
#             f.write(f'# {x.upper()} in {"minutes" if x=="Duration" else "mm/h"}\n')
#             f.write( '# ECDF+ (Empirical Cumulative Distribution '\
#                     'Function) regardless Elevation\n' )
#             if Z_CUTS:
#                 for pos, y in enumerate(copula.E.cat.categories):
#                     f.write(f'# ECDF+{y}: ECDF for Elevations between '\
#                             f'{cut_bin[pos]} and {cut_bin[pos+1]} m.a.s.l.\n')
#         dframe.to_csv(out, sep=',', index=False, mode='a', header=True)

#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#- INT-DUR COPULA.ANALYSIS (by ELEV) ----------------------------------- (END) #
#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#- FIT MIXTURE.of.VON.MISES.(-FISHER) to TIMES_OF_DAY ---------------- (START) #
#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#-CIRCULAR.STATS in PYTHON via 'vonMisesMixtures'-------------------------------

def TOD( radians, NMIX, N, TAG ):
# radians: hour of the day in RADians (pi == 12:00; 2*pi == 00:00/24:00)
# NMIX   : number of von Mises mixtures
# N      : (either 1 or 2) suffix relating the season for which the pdf is fitted
# TAG    : the output tag to label the output with (either 'DATIME' or 'DOYEAR')

    '''
WARNING!: this component is computationally intensive, and heavily depends on
the amount of points/data one is trying to fit.
the line ' mixture = vonmises.mixture_pdfit(radians, n=NMIX, threshold=1e-8) '
takes ~14.5 minutes to compute/fit (no parallel) ~230K (radian) values for a
fitting of 'just' 3 mixtures in a machine with the following specs:
Processor       -> 12th Gen Intel(R) Core(TM) i9-12900K, 3.20 GHz, 16 Core(s), 24 Logical Processor(s)
Installed RAM   -> 32.0 GB (31.7 GB usable)
~15 minutes is about 3x as long as the total spent by all other components!!.
we found that 3 mixtures is a very optimal fit for the Walnut Gulch catchment.
thus, the more mixtures one wants to add the more (exponential) time one needs.
therefore it is very advisable to:
a) run it just once, for 2 or more mixtures, and then then pass the resulting
    np.array (exported into 'OUTPUT_FILE') to the MIXTURE variable for/in future
    iterations.
b) fit just 1 mixture... if high-accuracy is not a high-demand... which it's the
    case for STORM2.0 with regard to this component.
c) comment out this component (in this script).
    STORM2.0 will assign a (uniform) random value for TOD by default. bear in mind
    that these TODs will not be statistically representative/accurate...but at
    least you'll avoid the hassle of dealing with the 'vonMisesMixtures' package.
    '''

# 3 is a fix dimension given that a vMF-pdf is made of mu, kappa, & proportionality
    if type( radians ).__name__ == 'ndarray' and radians.shape.__getitem__(0) == 3:
        MIXTURE = radians
    else:
# https://fraschelle.frama.io/mixture-of-von-mises-distributions/BasicUsage.html
        MIXTURE = vonmises.mixture_pdfit (radians, n=NMIX, threshold=1e-8 )

# # RESULTS FOR 3-MIXTURES FOR WALNUT GULCH CATCHMENT (ANALOG SET ONLY) **optimum**
#         MIXTURE = np.array(
#             [[4.3229866008934936e-01, 3.1433986726443774e-01, 2.5336147264620779e-01],
#              [2.5362023640410842e+00, 1.7002920476200680e+00, 6.8943802569831281e-01],
#              [4.7209161669312405e-01, 3.1990662021486749e+00, 6.4466332373429092e+00]] )
#         pprint( MIXTURE, 24 )
#         Probs. vonMises(mu=xx.x-h, kappa=xx.xxxx)
#         -----------------------------------------
#         0.4323 vonMises(mu=21.6876, kappa=0.4721) +
#         0.3143 vonMises(mu=18.4946, kappa=3.1991) +
#         0.2534 vonMises(mu=14.6335, kappa=6.4466)
# # RESULTS FOR 3-MIXTURES (ANALOG SET ONLY) FROM "pre_processing_circular.R"
#         MIXTURE = np.array(
#             [[3.12420687e-01, 4.33473238e-01, 2.54106076e-01],
#              [1.70161361e+00, 2.53188866e+00, 6.90215812e-01],
#              [3.21905433e+00, 4.73468500e-01, 6.43691294e+00]] )
#         pprint( MIXTURE, 24 )
#         Probs. vonMises(mu=xx.x-h, kappa=xx.xxxx)
#         -----------------------------------------
#         0.3124 vonMises(mu=18.4997, kappa=3.2191) +
#         0.4335 vonMises(mu=21.6711, kappa=0.4735) +
#         0.2541 vonMises(mu=14.6364, kappa=6.4369)

    with open(OUPUT_FILE, 'a') as f:
        [ f.write( f"{TAG}_VMF{N}+m{x+1},{','.join(map(str, [*xtem]))}\n" )\
            for x, xtem in enumerate(MIXTURE.T) ]

#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#- FIT MIXTURE.of.VON.MISES.(-FISHER) to TIMES_OF_DAY ------------------ (END) #
#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#- DAY OF THE YEAR (DOY) --------------------------------------------- (START) #
#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#~ DISCRETE PDF.FITTING for DOY ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def FIT_DIST( data, dist, bnds ):
# data: day of the year (integer)
# dist: PMF.name to fit
# bnds: tuple of boundaries (min,max) for parameter.estimation
    fitted = eval(f'stats.fit(stats.{dist}, data, {bnds})')
    # ks = stats.kstest(data, dist, fitted.params, 99)
# https://stanfordphd.com/BIC.html
    bic = np.nan if (np.isinf(fitted.nllf()) or np.isnan(fitted.nllf())) else\
        -2 *fitted.nllf() + len(bnds)*np.log( len(data) )
    # fitted.plot()
    # plt.show()

# # SOME ALTERNATIVES for DISCRETE DOY.PMF  -> 'poisson' (best.fit)
# # ('pmf.name', 'loglik', 'BIC', 'par1', 'par2', etc...)
# ('nbinom', 651821.7053205305, -1303619.7993912024, 64.0, 0.22140611133237353, 0.0)
# ('nhypergeom', 652970.3199918666, -1305905.2231089454, 617.0, 535.0, 35.0, 0.0)
# ('betabinom', 652419.352958415, -1304803.2890420423, 726.0, 42.402959107643966, 94.38504575868984, 0.0)
# ('poisson', 783036.4089066491, -1566061.012188369, 225.0608359100977, 0.0)

# # TESTS TO EVALUATE RELATION BETWEEN BIC & Log-Likelihood
# import statsmodels.api as sm
# data = one.loc[one.S=='W', 'doy']
# binc = sm.NegativeBinomial(data, np.ones(len(data)), loglike_method='nb1').fit(disp=True)
# poic = sm.Poisson(data, np.ones(len(data))).fit(disp=True)
# -2 *binc.llf + len(binc.params)*np.log( len(data) ) # bIC
# binc.summary2()
# """
#                      Results: NegativeBinomial
# ===================================================================
# Model:              NegativeBinomial Pseudo R-squared: 0.000
# Dependent Variable: doy              AIC:              1303618.6963
# Date:               2023-02-19 16:24 BIC:              1303638.3075
# No. Observations:   134004           Log-Likelihood:   -6.5181e+05
# Df Model:           0                LL-Null:          -6.5181e+05
# Df Residuals:       134003           LLR p-value:      nan
# Converged:          1.0000           Scale:            1.0000
# ---------------------------------------------------------------------
#             Coef.    Std.Err.       z        P>|z|    [0.025   0.975]
# ---------------------------------------------------------------------
# const       5.4164     0.0004   14141.9782   0.0000   5.4156   5.4171
# alpha       3.4240     0.0171     200.3376   0.0000   3.3905   3.4575
# ===================================================================
# """

# returns: [distro.name, neg.log.likelihood, BIC, estimated.parameters]
    return (dist, fitted.nllf(), bic, *fitted.params)


#~ function.to.nicely.print.parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#-https://fraschelle.frama.io/mixture-of-von-mises-distributions/BasicUsage.html
def pprint( params, SCALAR ):
    # SCALAR = 365
    s = "Probs. vonMises(mu=xx.x-h, kappa=xx.xxxx)"
    print(s)
    print("-"*len(s))
    n = params.shape[1]
    s = ""
    for line in range(n):
        tup = list( tuple(params[:,line]) )
        tup[1] = (tup[1] +np.pi) /(2*np.pi) *SCALAR
    # use the 2.lines below (instead) if you used that weird "Nrand" column!
    #    tup[1] = tup[1] if tup[1]>=0 else np.pi -tup[1]
    #    tup[1] = tup[1] *1/(2*np.pi) *24
        s += "{:.4f} vonMises(mu={:.4f}, kappa={:.4f})".format(*tuple(tup))
        if line!=n-1:
            s += " + \n"
    print( s )


#~ CALLS 'FIT_DIST' & SELECTS THE BEST FIT for DOY ~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def DAYOFYEAR( data, N ):
# fit the discrete PMF
    dfit = list(map(lambda y,z: FIT_DIST( data, y,z ), tqdm( DISC_PDFS, ncols=50 ), disc_bnds ))
# selecting the best.fit based on BIC (or NegLogLik)
    fitd = list(zip(*dfit))[ param_sel ]
    fitd = dfit[ fitd.index(np.min( fitd )) ]
# https://stackoverflow.com/a/20540948/5885810
    dtwo = dict(zip( list(map(lambda x: f'p{x}', range(len( fitd[3:] )))), fitd[3:] ))
# create the whole 'binom'.dict and call FIT_PDF to write it down
    ddic = eval( f'dict({ fitd[0] }={ dtwo })' )
# export the PDF
    FIT_PDF( ddic, N, "DOYEAR_PMF", [] )

# # fit the continuous VMF
# # RESULTS FOR 5-MIXTURES FOR WALNUT GULCH CATCHMENT (ANALOG SET ONLY) **optimum**
#         MIXTURE = np.array(
#             [[7.1793832553957826e-02, 8.5709143292117873e-02, 5.4631313068818396e-02, 9.8271993676566555e-02, 6.8959371740853792e-01],
#              [1.5448202344935775e+00, 2.0950532997037094e-01, 1.9015008192343976e+00, 1.1608274364741620e+00, 5.5023815988073743e-01],
#              [9.1340520493766306e+01, 6.2316491294279110e+01, 1.1464696492692285e+02, 4.8672386254158290e+01, 6.9364503709825778e+00]] )
#         pprint( MIXTURE, 365 )
#         Probs. vonMises(mu=xx.x-h, kappa=xx.xxxx)
#         -----------------------------------------
#         0.0718 vonMises(mu=272.2410, kappa= 91.3405) +
#         0.0857 vonMises(mu=194.6705, kappa= 62.3165) +
#         0.0546 vonMises(mu=292.9611, kappa=114.6470) +
#         0.0983 vonMises(mu=249.9343, kappa= 48.6724) +
#         0.6896 vonMises(mu=214.4642, kappa=  6.9365)
# # RESULTS FOR 5-MIXTURES (ANALOG SET ONLY) FROM "pre_processing_circular.R"
#         MIXTURE = np.array(
#             [[4.56582417e-02, 5.27487992e-02, 1.27401049e-02, 4.43134401e-01, 4.45718453e-01],
#              [1.56155245e+00, 1.90222888e+00,-4.22426346e-01, 3.10674800e-01, 9.47535760e-01],
#              [1.31680596e+02, 1.15425898e+02, 2.88989175e+02, 1.66106095e+01, 9.32419377e+00]] )
#         pprint( MIXTURE, 365 )
#         Probs. vonMises(mu=xx.x-h, kappa=xx.xxxx)
#         -----------------------------------------
#         0.0457 vonMises(mu=273.2130, kappa=131.6806) +
#         0.0527 vonMises(mu=293.0034, kappa=115.4259) +
#         0.0127 vonMises(mu=157.9606, kappa=288.9892) +
#         0.4431 vonMises(mu=200.5476, kappa=16.6106) +
#         0.4457 vonMises(mu=237.5438, kappa=9.3242)

#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#- DAY OF THE YEAR (DOY) ----------------------------------------------- (END) #
#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


#~ put all the previous functions to work ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def MAIN():

    '''
WARNING!: Walnut Gulch data does NOT come with season 2, i.e., there is no 'W2'
category in column(s) ['S'].
the examples below are only meant as a guidance for your own datasets...if they
happen to be bi-seasonal.
    '''

    OUTPUT_PATH()

    print('\nreading datsets...', end='', flush=True)
    ten_cum = READ_DATA_CUMULATIVE()
    one = READ_DATA_EVENT()
    # # RUN THIS (ONCE) ONLY TO COMPUTE COMPUTE THIS EXERCISE in cRan
    # ((one.loc[one.S=='W',['dRAD']] +np.pi) /(2*np.pi) * one.doy.max()).to_csv('pre_processing--ddoy_ANALOG_W.csv', header=False, index=False, mode='w')
    # ((one.loc[one.S=='W',['hRAD']] +np.pi) /(2*np.pi) *24).to_csv('pre_processing--dour_ANALOG_W.csv', header=False, index=False, mode='w')

#-GAGE.GEOPOSITIONS (and their CROSS_DISTANCE) MUST BE DEFINED GLOBALLY
    READ_META_DATA()
    print(' done!')

    print('computing seasonal totals...', end='', flush=True)
# if you need to compute the SEASONAL-PDF from your data
    MONSOON( ten_cum, SEASON='W', SMAX=5, N=1 )
    # MONSOON( ten_cum, SEASON='W2', SMAX=3, N=2 )
# if the PDFs are passed manually, the names of the PDF-parameters are irrelevant...
# ...as long as they're not identical! (e.g., 'loc','scale' == 'mean','var')
    FIT_PDF( {'norm': {'loc':5.362910217624112, 'scale':0.3167606398478469}}, 2,\
            'TOTALP_PDF', [] )
    print(' done!')

    print('\ncomputing storm areas:')
    core_n = AREA_DATA( one )
# pass elements with at least 3.gauges (or more)... for a given season
    print('fitting pdfs...', end='')#, flush=False)
    FIT_PDF( core_n.loc[(core_n.numb>=3) & (core_n.S=='W'), 'cmax'], 1, 'RADIUS_PDF', RSKW_PDFS )
    # FIT_PDF( core_n.loc[(core_n.numb>=3) & (core_n.S=='W2'), 'cmax'], 2, 'RADIUS_PDF', RSKW_PDFS )
    FIT_PDF( {'gamma': {'a':4.399625814327896, 'loc':-0.47511306843943046,\
                         'scale':1.3991123114790807}} , 2, 'RADIUS_PDF', [] )
    print(' done!')

    print('\ncomputing decorrelation parameters:')
    pars, max_copula = RECESS_DATA( core_n, 'W' )
    # pars_two, max_copula_two = RECESS_DATA( core_n, 'W2' )
    FIT_PDF( pars[:,1], 1, 'BETPAR_PDF', NORM_PDFS )
    # FIT_PDF( pars_two[:,1], 2, 'BETPAR_PDF', NORM_PDFS )
    FIT_PDF( {'burr': {'c':2.351235686277189, 'd':0.8505976651919634,\
                       'loc':-0.0011370351719608585, 'scale':0.08377708092591916}}\
            , 2, 'BETPAR_PDF', [] )

# #-ONLY.USE.THESE IMAX.&.ADUR PDFs IF.YOU'RE.NOT.USING ECDFs.FROM.COPULAS!!
#     print('\ncomputing maximum intesity & average duration...', end='', flush=False)
#     FIT_PDF( pars[:,0], 1, 'MAXINT_PDF', IMAX_PDFS )
#     # FIT_PDF( pars_two[:,0], 2, 'MAXINT_PDF', IMAX_PDFS )
#     FIT_PDF( {'lomax': {'c':5.432548872708142, 'loc':0.06999999933105766,\
#                         'scale':26.87198199648127}}, 2, 'MAXINT_PDF', [] )
#     FIT_PDF( max_copula.davg, 1, 'AVGDUR_PDF', RSKW_PDFS )
#     # FIT_PDF( max_copula_two.davg, 2, 'AVGDUR_PDF', RSKW_PDFS )
#     FIT_PDF( {'wald': {'loc':-2.553691071708469,'scale':112.48428969036745}}, 2,\
#             'AVGDUR_PDF', [] )
#     print(' done!')

    print('\ncomputing I_max & D_avg & Int-Dur copulas...', end='', flush=True)
    COPULA( max_copula, 1 )
    # COPULA( max_copula_two, 2 )
    COPULA( max_copula, 2 )
    print(' done!')

# if "vonMisesMixtures" couldn't be installed... you CAN comment.off this whole TOD
    print('computing VMF for Time-Of-Day...', end='', flush=True)
    mixture_for_tod = np.array( # -> the one from R
        [[2.49156475242312e-01, 3.24520271490722e-01, 4.26323253266966e-01],
         [3.83327280990811e+00, 4.84895620149114e+00, 5.69886684675002e+00],
         [6.39091201120178e+00, 3.06431018164439e+00, 4.69994281181170e-01]] )
    mixture_for_tod[1,:] = mixture_for_tod[1,:] -np.pi
    # mixture_for_tod = np.array( # -> the one from PYTHON
    #     [[2.468419551426534e-01, 3.315907829508499e-01, 4.215672619064808e-01],
    #      [6.893273101866058e-01, 1.703495074238532e+00, 2.575690215589140e+00],
    #      [6.418276632248997e+00, 3.000253688430609e+00, 4.649344363713587e-01]] )
    TOD( mixture_for_tod, NMIX=mixture_for_tod.shape.__getitem__(-1), N=1, TAG='DATIME' )
    # TOD( one.loc[one.S=='W', 'hRAD'], NMIX=3, N=1, TAG='DATIME' )
    # TOD( one.loc[one.S=='W2','hRAD'], NMIX=1, N=2, TAG='DATIME' )
    TOD( one.loc[one.S=='W', 'hRAD'], NMIX=1, N=2, TAG='DATIME' )
    print(' done!')

# if "vonMisesMixtures" couldn't be installed... you CAN still work with DAYOFYEAR()
    if "USE_DOY" in globals():
        print('computing VMF for Day-Of-Year...', end='', flush=True)
        # mixture_for_doy = np.array( # -> the one from R (3-mix)
        #     [[4.04905655044381e-02, 3.06199138239520e-01, 6.53310296256042e-01],
        #      [5.06206592560194e+00, 4.35421320240035e+00, 3.56594990301999e+00],
        #      [1.22898067787712e+02, 9.26957303636785e+00, 9.34243664965950e+00]] )
        # mixture_for_doy[1,:] = mixture_for_doy[1,:] -np.pi
        mixture_for_doy = np.array( # -> the one from R (5-mix)
            [[4.44824317699613e-02, 1.17663927485551e-02, 4.65681610396558e-01, 4.2495217043963e-01, 5.31173946453013e-02],
             [4.71114482356166e+00, 2.72067414568930e+00, 3.46668770085889e+00, 4.1033358440160e+00, 5.04882607523782e+00],
             [1.29046671192187e+02, 2.87172811856793e+02, 1.55871683696581e+01, 9.4627962328381e+00, 1.04719296092994e+02]] )
        mixture_for_doy[1,:] = mixture_for_doy[1,:] -np.pi
        # mixture_for_doy = np.array( # -> the one from PYTHON
        #     [[5.446271224933757e-02, 8.954689198507723e-02, 7.054990890263783e-02, 8.732366808837940e-02, 6.981168187745506e-01],
        #      [1.907409236571684e+00, 2.286006656480273e-01, 1.551363989955984e+00, 1.172647712414930e+00, 5.586619737368196e-01],
        #      [1.053227046311876e+02, 5.197011694272198e+01, 8.719112504916305e+01, 5.291419173073581e+01, 6.828147043994408e+00]] )
        TOD( mixture_for_doy, NMIX=mixture_for_doy.shape.__getitem__(-1), N=1, TAG='DOYEAR' )
        # TOD( one.loc[one.S=='W', 'dRAD'], NMIX=5, N=1, TAG='DOYEAR' )
        # TOD( one.loc[one.S=='W2','dRAD'], NMIX=1, N=2, TAG='DOYEAR' )
        TOD( one.loc[one.S=='W', 'dRAD'], NMIX=1, N=2, TAG='DOYEAR' )
        print(' done!')
    else:
        print('computing PDF/VMF for Day-Of-Year:')
        DAYOFYEAR( one.loc[one.S=='W', 'doy'], N=1 )
        # DAYOFYEAR( one.loc[one.S=='W2', 'doy'], N=2 )
        FIT_PDF( {'nbinom': {'n':64.0, 'p':0.22140611133237353, 'loc':0}} , 2, 'DOYEAR_PMF', [] )

    print('\n¡PRE-PROCESS succesfully preprocessed, have FUN!.')


#%%

if __name__ == '__main__':
    MAIN()