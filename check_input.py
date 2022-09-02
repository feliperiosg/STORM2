import numpy as np
from pandas import DataFrame
from pathlib import Path
from datetime import datetime
from dateutil.tz import tzlocal
from parameters import *

# https://stackoverflow.com/a/23116937/5885810  (0 divison -> no warnings)
# https://stackoverflow.com/a/29950752/5885810  (0 divison -> no warnings)
np.seterr(divide='ignore', invalid='ignore')


#%% FUNCTIONS' DEFINITION

#~ replace FILE.PARAMETERS with those read from the command line ~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def PAR_UPDATE( args ):
    for x in list(vars( args ).keys()):
# https://stackoverflow.com/a/2083375/5885810   (exec global... weird)
        exec(f'globals()["{x}"] = args.{x}')
    # print([PTOT_SC, PTOT_SF])


#~ SO WELCOME() CAN EVENTUALLY BE USED WITHOUT PREVIOUSLY RUNNING ASSERT() ~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#wet_hash = pd.DataFrame({'Var2':['PTOT', 'STORMINESS'],
wet_hash = DataFrame({'Var2':['PTOT', 'STORMINESS'],
                      'Var3':['Total Rainfall', 'Rain Intensity']})


#~ BLURTS OUT WARNINGS IF YOUR 'SOFT-CORE' PARAMETERS 'SMELL FUNNY' ~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def ASSERT():
# checks the MODE & SEASONS (input) parameters
    assert MODE.upper() in ('VALIDATION', 'SIMULATION'), 'MODE not valid!\n'\
        'Please try either "validation" or "simulation" (case-insensitive).'
    assert SEASONS in (1, 2), 'SEASONS not valid!\nIt must be either 1 or 2.'

# checks the _STEPCHANGE & _SCALING_FACTOR (input) parameters
    for j in wet_hash.Var2:#j='PTOT'
        SC = f'{j}_SC'
        SF = f'{j}_SF'
    # do STEPCHANGE & SCALING_FACTOR have the same lengths??
        assert len(eval(SC)) == len(eval(SF)), f'Imcompatible Sizes!\nPlease, '\
            f'ensure that both {SC} and {SF} have the same length/size.'
    # does each dimension/season have the same length among them??
        assert np.in1d([np.unique( tuple([np.asarray(x).size for x in eval(x)]) ).size\
            for x in [SC, SF]] ,1).all(),\
                'Imcompatible Sizes!\nPlease, ensure that the parameteres (either'\
                f' in {SC} or {SF}) have the same length/size for each season.'
    # is each dimension/season as long as 1 or NUMSIMYRS??
        for i in range(SEASONS):#i=0
            assert 1 in tuple(map( lambda x: np.asarray(eval(f'{x}[{i}]')).size, [SC, SF] )) or\
                NUMSIMYRS in tuple(map( lambda x: np.asarray(eval(f'{x}[{i}]')).size, [SC, SF] )),\
                    f'Imcompatible Sizes!\nBoth {SC} and {SF} must have '\
                    'lengths of either 1 (one parameter for all NUMSIMYRS)'\
                    f' or {NUMSIMYRS} (one parameter for each NUMSIMYRS).'
# https://stackoverflow.com/a/25050572/5885810  (1st items of a list of lists)
# https://stackoverflow.com/a/31154423/5885810  (find None in tuples)
# https://stackoverflow.com/q/12116491/5885810  (0's to None)
        # the list comprehension has to be done as Python 'reads' 0s as None
            assert not any( map( lambda x: x is None,\
                list( zip(*map( eval, [SC, SF] )) ).__getitem__(i) ) ), 'Missing Values!'\
                    '\nThere are missing values (i.e. None) in some (or all) of the'\
                    f' {", ".join([SC, SF])} variables for Season {i+1}.\nPlease,'\
                    ' ensure that the aforementioned variables contain numerical '\
                    f'values, if you are indeed planning to model Season {i+1}.'
        some_sum = np.nansum((
            np.asarray(eval(SC), dtype='f8') / np.asarray(eval(SC), dtype='f8'),
            2* np.asarray(eval(SF), dtype='f8') / np.asarray(eval(SF), dtype='f8')
            ), axis=0, dtype=np.int32)
    # the integer 3 helps to distinguish between '...stepchange' & '...scaling_factor'
        assert 3 not in some_sum, f'{j.upper()}_SCENARIO not valid!\n'\
            f'Please, ensure that either {SC} or {SF} (or both!) are '\
            f'set to 0 (zero), for any given season of the {MODE.lower()}.'

# check that the DEM exists (if you're using Z_CUTS)
    assertdem = f'NO DEM_FILE!\n'\
        'You chose to model rainfall at/for different altitudes, i.e., Z_CUTS != [].'\
        f' Nevertheless, the path to the DEM ({DEM_FILE}) was not correctly set up '\
        'or the file does not exist.\nPlease ensure that the DEM file exists in the'\
        ' correct path.\nConversely, you can also switch off the Z_CUTS variable'\
        ' (i.e., Z_CUTS == [] or None) if you aim to model rainfall regardless altitude.'
    if Z_CUTS:
# https://stackoverflow.com/a/82852/5885810     (files exists?)
# https://stackoverflow.com/a/40183030/5885810  (assertion error)
# https://stackoverflow.com/a/6095782/5885810   (multiple excepts)
        try:
            Path( DEM_FILE ).resolve( strict=True )
        except (FileNotFoundError, TypeError):
            raise AssertionError( assertdem )

# check that the GAG exists (if you're doing VALIDATION)
    assertgag = f'NO GAUGE_FILE!\n'\
        f"You are after a 'validation run', i.e., MODE == {MODE}. Nevertheless, the"\
        f' path to the Gauge file ({GAG_FILE}) was not correctly set up or the file'\
        ' does not exist.\nPlease ensure that the Gauge file exists in the correct path.'
    if MODE.lower() == 'validation':
        try:
            Path( GAG_FILE ).resolve( strict=True )
        except (FileNotFoundError, TypeError):
            raise AssertionError( assertgag )

# check that the SHP exists (always!)
    try:
        Path( SHP_FILE ).resolve( strict=True )
    except (FileNotFoundError, TypeError):
        raise AssertionError( f'NO SHP_FILE!\n'\
            f'The path to the SHP file ({SHP_FILE}) was not correctly set up or the file'\
            ' does not exist.\nPlease ensure that the SHP file exists in the correct path.' )

# check that the SEASON_MONTHS exists and/or are correctly set up
    assertext = f'You chose to model {SEASONS} season(s) but there is/are either'\
        ' missing or not correctly allocated seasonal period(s) in the variable '\
        'SEASONS_MONTHS, which defines the date-times of the wet season(s).\n'\
        'Please update the aforementioned variable accordingly.'
    if SEASONS == 1:
        assert list(map(lambda x: not None in x, zip(SEASONS_MONTHS))).__getitem__(0), assertext
    else:
        assert not None in SEASONS_MONTHS, assertext


#~ TRANSFORMS THE NUMERICAL INPUT OF xxx_SC/SF INTO 'READABLE' LABELS ~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def INFER_SCENARIO( stepchange, scaling_factor, tab_x, tab_sign ):
    #stepchange=PTOT_SC; scaling_factor=PTOT_SF; tab_x=tab_ptot
# convert input into numpy
    stepchange = np.asarray(stepchange, dtype='f8')
    scaling_factor = np.asarray(scaling_factor, dtype='f8')
# establish whether is a 0, 1, 2, or 3
    sum_vec = np.nansum((stepchange /stepchange, 2* scaling_factor /scaling_factor),
                        axis=0, dtype=np.int32)
# compute signs
    sign_ar = np.sign( np.sign( stepchange ) + np.sign( scaling_factor ) )
# find the variables and their signs int the corresponding 'tables'
# ...the 'for' loop is necessary as neither .isin nor .reindex can be used (negative or repeated values)
# https://stackoverflow.com/a/51327154/5885810      (.reindex instead of .isin)
    # str_vec = list(map( lambda A,B: f"{A}{B}", tab_x.loc[sum_vec,'Var1'].values,
    #     tab_sign.loc[ [np.where(tab_sign.Var2==x).__getitem__(0).__getitem__(0)\
    #         for x in sign_ar[~np.isnan(sign_ar)]]  ].Var1.values ))
    str_vec = list(map( lambda A,B: f"{A}{B}", tab_x.loc[sum_vec,'Var1'].values,
        np.concatenate([tab_sign.loc[tab_sign.Var2.isin([x]), 'Var1'].values\
            for x in sign_ar[~np.isnan(sign_ar)]]) ))
# # the above line 'works' as there few values to look for. otherwise, faster methods are:
# #tab_sign.loc[np.take(np.argsort(tab_sign.Var2), np.searchsorted(tab_sign.Var2[np.argsort(tab_sign.Var2)], sign_ptot)), 'Var1'].values
# #tab_sign.loc[np.argsort(tab_sign.Var2)[np.searchsorted(tab_sign.Var2[np.argsort(tab_sign.Var2)], sign_ptot)], 'Var1'].values
# # https://stackoverflow.com/a/8251668/5885810     (find elements in pd)
    return str_vec


#~ RE-STATES THE 'SOFT-CORE' PARAMETERS (some legacy of STORM1.0 ~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def WELCOME():
    global PTOT_SCENARIO, STORMINESS_SCENARIO
# tables to correlate signs & scenarios
    tab_sign = DataFrame({'Var1':['', '+', '-'], 'Var2':[0, 1, -1]})
    tab_ptot = DataFrame({'Var1':['ptotC', 'ptotS', 'ptotT', 'n/a'], 'Var2':[0, 1, 2, 3]})
    tab_storm = DataFrame({'Var1':['stormsC', 'stormsS', 'stormsT', 'n/a'],
                            'Var2':[0, 1, 2, 3]})
    """
  'ptotC' == Stationary conditions / Control Climate
  'ptotS' == Step Change (increase/decrese) in the observed wetness
  'ptotT' == Progressive Trend (positive/negative) in the observed wetness
'stormsC' == Stationary conditions / Control Climate
'stormsS' == Step Change (increase/decrese) in the observed storminess'
'stormsT' == Progressive Trend (positive/negative) in the observed storminess'
    'n/a' == scenario NOT DEFINED (as both xxx_SC & xxx_SF differ from 0)
    """
# infer scenarios
    PTOT_SCENARIO = INFER_SCENARIO(PTOT_SC, PTOT_SF, tab_ptot, tab_sign)
    STORMINESS_SCENARIO = INFER_SCENARIO(STORMINESS_SC, STORMINESS_SF, tab_storm, tab_sign)
# create OUT_PATH folder (if it doen'st exist already)
# https://stackoverflow.com/a/50110841/5885810  (create folder if exisn't)
    Path( OUT_PATH ).mkdir(parents=True, exist_ok=True)
# define NC.output file.names
    NC_NAMES =  list(map( lambda a,b,c: f'{OUT_PATH}/{MODE[:3].upper()}_'\
        f'{datetime.now(tzlocal()).strftime("%y%m%dT%H%M")}_S{a+1}_{b.strip()}_'\
        f'{c.strip()}.nc', range(SEASONS), PTOT_SCENARIO, STORMINESS_SCENARIO ))
# print the CORE INFO
    form_out = '".02f"'
    print('\nRUN SETTINGS')
    print('************\n')
    print(f'Type of Model Run: {MODE.upper()}')
    print(f"Number of Seasons: {['ONE', 'TWO'][SEASONS-1]}")
    print(f'Number of {MODE.lower().capitalize()}s: {NUMSIMS}')
    print(f'Years per {MODE.lower().capitalize()} : {NUMSIMYRS}')
    for j in wet_hash.Var2:
        # var = 'SC' if eval(f'{j}_SCENARIO').__getitem__(-2)=="S" else 'SF' # NOT ENTIRELY ACCURATE!!
        print(f'{wet_hash[wet_hash.Var2.isin([ j ])].Var3.iloc[ 0 ]} scenarios '\
              f'({" | ".join( [f"S{x+1}" for x in range(SEASONS)] )}):  '\
# 8 because 'stormsT+' is the maximum length of these strings
               f'{ " | ".join( map(eval, [f"{j}_SCENARIO[{x}].center(8," ")" for x in range(SEASONS)]) )}')
#               f'{" | ".join([" ".join(map(eval, [ f"{j}_SCENARIO[{x}].center(8," ")", f"format(abs( {j}_{var}[{x}] ), {form_out})" ] )) for x in range(SEASONS)])}')
# # https://stackoverflow.com/a/25559140/5885810  (string no sign)

    print('\nOutput paths:')
# https://www.delftstack.com/howto/python/python-pad-string-with-spaces/
# https://stackoverflow.com/a/45120812/5885810  (print/create string padding)
    print(*[( k.ljust(max(map(len, NC_NAMES)),' ') ).rjust(max(map(len, NC_NAMES))+4,' ')\
            for k in NC_NAMES], sep='\n')

    return NC_NAMES


#%%

if __name__ == '__main__':
    ASSERT()
    WELCOME()