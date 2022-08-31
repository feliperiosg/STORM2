import vonMisesMixtures as vonmises
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from datetime import datetime, timedelta
from scipy import optimize
from scipy.spatial import distance
from fitter import Fitter
from statsmodels.distributions.copula.api import GaussianCopula
#from statsmodels.distributions.empirical_distribution import ECDF
from parameters import Z_CUTS, WGEPSG

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
# input files
AGGRE_DATA = './model_input/data_WG/gage_data--1953Aug-2022Jan_Jun-Oct_aggregateh.csv'
EVENT_DATA = './model_input/data_WG/gage_data--1953Aug18-2022Jan01_Jun-Oct_eventh.csv'
GAUGE_META = './model_input/data_WG/gage_data--gageNetworkWG.csv'
# output files
OUPUT_FILE = './model_input/ProbabilityDensityFunctions_TWO.csv'
# # if you want to redefine Z_CUTS here
# Z_CUTS = [1350, 1500]
# # for no INT-DUR copula at different altitudes
# Z_CUTS = []
# Z_CUTS = None

# global variables
PTOT_PDFS = ['argus','betaprime','burr','burr12','chi','chi2','exponweib','exponpow'
             ,'gausshyper','gengamma','genhalflogistic','gumbel_l','gumbel_r','invgauss'
             ,'invweibull','johnsonsb','johnsonsu','ksone','kstwobign','loggamma'
             ,'maxwell','nakagami','ncx2','norm','powerlognorm','rayleigh','rice'
             ,'tukeylambda','weibull_min','weibull_max']

# these distros work for either AVG.DUR and/or BETA (BRUTE_FORCE)
RSKW_PDFS = ['alpha','betaprime','burr','f','fisk','gamma','geninvgauss','gilbrat'
             ,'invgamma','invweibull','johnsonsb','johnsonsu','ksone','lognorm','mielke'
             ,'moyal','norm','powerlognorm','rayleigh','rice','wald','weibull_min']

IMAX_PDFS = ['truncexpon','pareto','loguniform','lomax','halfgennorm','genpareto'
             ,'expon']

# these heavily?-"gaussian" distributions work for the BETA (FORCE-BRUTE)
NORM_PDFS = ['cauchy','cosine','exponnorm','gumbel_r','gumbel_l','hypsecant'
             ,'laplace','logistic','moyal','norm','powernorm']

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
        #best_fit.summary()
# SELECT THE PARAMETERS FOR THE BEST.FIT (i prefer 'BIC')
        #PDFFIT = best_fit.get_best(method = 'sumsquare_error')
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

def TOD( radians, NMIX, N ):
# radians: hour of the day in RADians (pi == 12:00; 2*pi == 00:00/24:00)
# NMIX   : number of vonMises-Fisher mixtures
# N      : (either 1 or 2) suffix relating the season for which the pdf is fitted

    '''
WARNING!: this component is computationally intensive, and heavily depends on
the amount of points/data one is trying to fit.
the line ' mixture = vonmises.mixture_pdfit(radians, n=NMIX, threshold=1e-8) '
takes ~14.5 minutes to compute/fit (no parallel) ~230K (radian) values for a
fitting of 'just' 3 mixtures (~1 second for 1 mixture)... in a machine with the following specs:
Processor       -> 12th Gen Intel(R) Core(TM) i9-12900K, 3.20 GHz, 16 Core(s), 24 Logical Processor(s)
Installed RAM   -> 32.0 GB (31.7 GB usable)
~15 minutes is about 3x as long as the total spent by all other components!!.
we found that 3 mixtures is a very optimal fit for the Walnut Gulch catchment.
thus, the more mixtures one wants to add the more (exponential) time one needs.
therefore it is very advisable to:
a) run it just once (offline if possible), for 2 or more mixtures, and then
    then pass the resulting np.array (with the parameters) to the MIXTURE variable
    for/in future iterations.
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
        MIXTURE = vonmises.mixture_pdfit(radians, n=NMIX, threshold=1e-8)

# # RESULTS FOR 1-MIXTURE FOR WALNUT GULCH CATCHMENT
#         MIXTURE = np.array(
#             [[ 1.        ],
#               [1.44096952],
#               [1.02412649]]
#             )
#             Probs. vonMises(mu=xx.x-h, kappa=xx.xxxx)
#             -----------------------------------------
#             1.0000 vonMises(mu=17.5041, kappa=1.0241)

# # RESULTS FOR 3-MIXTURES FOR WALNUT GULCH CATCHMENT (very first run)
#         MIXTURE = np.array(
#             [[  0.30571647,  0.21751198,  0.47677155],
#               [-2.8554979 ,  0.63499782,  1.711601  ],
#               [ 0.36807874,  6.61593317,  2.26518188]]
#             )
#             Probs. vonMises(mu=xx.x-h, kappa=xx.xxxx)
#             -----------------------------------------
#             0.3057 vonMises(mu= 1.0928, kappa=0.3681) +
#             0.2175 vonMises(mu=14.4255, kappa=6.6159) +
#             0.4768 vonMises(mu=18.5378, kappa=2.2652)

# # RESULTS FOR 3-MIXTURES FOR WALNUT GULCH CATCHMENT (most recent run)
#         MIXTURE = np.array(
#             [[  0.33029448,  0.22250998,  0.44719555],
#               [-3.04486903,  0.64182193,  1.71450856],
#               [ 0.35123045,  6.55350374,  2.3901821 ]]
#             )

    with open(OUPUT_FILE, 'a') as f:
        [ f.write( f"DATIME_VMF{N}+m{x+1},{','.join(map(str, [*xtem]))}\n" )\
            for x, xtem in enumerate(MIXTURE.T) ]

#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#- FIT MIXTURE.of.VON.MISES.(-FISHER) to TIMES_OF_DAY ------------------ (END) #
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
#-GAGE.GEOPOSITIONS (and their CROSS_DISTANCE) MUST BE DEFINED GLOBALLY
    READ_META_DATA()
    print(' done!')

    print('computing seasonal totals...', end='', flush=True)
# if you need to compute the SEASONAL-PDF from your data
    MONSOON( ten_cum, SEASON='W', SMAX=5, N=1 )
    #MONSOON( ten_cum, SEASON='W2', SMAX=3, N=2 )
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
    #FIT_PDF( core_n.loc[(core_n.numb>=3) & (core_n.S=='W2'), 'cmax'], 2, 'RADIUS_PDF', RSKW_PDFS )
    FIT_PDF( {'gamma': {'a':4.399625814327896, 'loc':-0.47511306843943046,\
                         'scale':1.3991123114790807}} , 2, 'RADIUS_PDF', [] )
    print(' done!')

    print('\ncomputing decorrelation parameters:')
    pars, max_copula = RECESS_DATA( core_n, 'W' )
    #pars_two, max_copula_two = RECESS_DATA( core_n, 'W2' )
    FIT_PDF( pars[:,1], 1, 'BETPAR_PDF', NORM_PDFS )
    #FIT_PDF( pars_two[:,1], 2, 'BETPAR_PDF', NORM_PDFS )
    FIT_PDF( {'burr': {'c':2.351235686277189, 'd':0.8505976651919634,\
                       'loc':-0.0011370351719608585, 'scale':0.08377708092591916}}\
            , 2, 'BETPAR_PDF', [] )

# #-ONLY.USE.THESE IMAX.&.ADUR PDFs IF.YOU'RE.NOT.USING ECDFs.FROM.COPULAS!!
#     print('\ncomputing maximum intesity & average duration...', end='', flush=False)
#     FIT_PDF( pars[:,0], 1, 'MAXINT_PDF', IMAX_PDFS )
#     #FIT_PDF( pars_two[:,0], 2, 'MAXINT_PDF', IMAX_PDFS )
#     FIT_PDF( {'lomax': {'c':5.432548872708142, 'loc':0.06999999933105766,\
#                         'scale':26.87198199648127}}, 2, 'MAXINT_PDF', [] )
#     FIT_PDF( max_copula.davg, 1, 'AVGDUR_PDF', RSKW_PDFS )
#     #FIT_PDF( max_copula_two.davg, 2, 'AVGDUR_PDF', RSKW_PDFS )
#     FIT_PDF( {'wald': {'loc':-2.553691071708469,'scale':112.48428969036745}}, 2,\
#             'AVGDUR_PDF', [] )
#     print(' done!')

    print('\ncomputing I_max & D_avg & Int-Dur copulas...', end='', flush=True)
    COPULA( max_copula, 1 )
    #COPULA( max_copula_two, 2 )
    COPULA( max_copula, 2 )
    print(' done!')

    print('computing starting times...', end='', flush=True)
    #TOD( one.loc[one.S=='W', 'hRAD'], NMIX=3, N=1 )
    mixture_for_WGc = np.array(
        [[  0.33029448,  0.22250998,  0.44719555],
          [-3.04486903,  0.64182193,  1.71450856],
          [ 0.35123045,  6.55350374,  2.3901821 ]]
        )
    TOD( mixture_for_WGc, NMIX=3, N=1 )
    #TOD( one.loc[one.S=='W2', 'hRAD'], NMIX=3, N=2 )
    TOD( one.loc[one.S=='W', 'hRAD'], NMIX=1, N=2 )
    print(' done!')

    print('\n¡PRE-PROCESS succesfully preprocessed, have FUN!.')


#%%

if __name__ == '__main__':
    MAIN()