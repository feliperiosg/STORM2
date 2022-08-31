from parameters import *



def ARG_UPDATE( args ):
    for x in list(vars( args ).keys()):
        # # use this if you're not inside a (local) function
        # exec(f'if args.{x}:\n\t{x} = args.{x}')
# https://stackoverflow.com/a/2083375/5885810   (exec global... weird)
        exec(f'if args.{x}:\n\tglobals()["{x}"] = args.{x}')
    return args


# https://stackoverflow.com/a/48295546/5885810  (None in argparse)
def none_too( v ):
    return None if v.lower() == 'none' else float( v )


# https://www.geeksforgeeks.org/command-line-arguments-in-python/
# https://docs.python.org/3/library/argparse.html#the-add-argument-method
def PARCE( parser ):
    #parser = argparse.ArgumentParser(description = 'STOchastic Rainfall Modelling [STORM]')#, prog='STORM')
# add the defaults
# https://stackoverflow.com/a/27616814/5885810  (parser case insensitive)
    parser.add_argument('-m', '--MODE', type=str.lower, default=MODE, choices=['simulation', 'validation'],
                        help='Type of Run (case-insensitive) (default: %(default)s)')
    parser.add_argument('-w', '--SEASONS', type=int, default=SEASONS, choices=[1,2],
                        help='Number of Seasons (per Run) (default: %(default)s)')
    parser.add_argument('-n', '--NUMSIMS', type=int, default=NUMSIMS,
                        help='Number of runs per Season (default: %(default)s)')
    parser.add_argument('-y', '--NUMSIMYRS', type=int, default=NUMSIMYRS,
                        help='Number of years per run (per Season) (default: %(default)s)')
    parser.add_argument('-ps', '--PTOT_SC', default=PTOT_SC, type=none_too, nargs='+',#type=float
                        help='Signed scalar (per Season) specifying the step change in the observed wetness (default: %(default)s)')
    parser.add_argument('-pf', '--PTOT_SF', default=PTOT_SF, type=none_too, nargs='+',
                        help='Signed scalar (per Season) specifying the progressive trend in the observed wetness')
    parser.add_argument('-ss', '--STORMINESS_SC', default=STORMINESS_SC, type=none_too, nargs='+',
                        help='Signed scalar (per Season) specifying the step change in the observed storminess')
    parser.add_argument('-sf', '--STORMINESS_SF', default=STORMINESS_SF, type=none_too, nargs='+',
                        help='Signed scalar (per Season) specifying the progressive trend in the observed storminess')
    parser.add_argument('--version', action='version', version='STORM 2.0')#'%(prog)s 2.0' )
# Read arguments from command line
    args = parser.parse_args()
    # print(args)
# REDEFINE variables by their names, instead of relative to 'args'
    updated_args = ARG_UPDATE( args )


    # print(f'\nvar MODE: {MODE}')
    # print(f'var SEASONS: {SEASONS}')
    # print(f'var NUMSIMS: {NUMSIMS}')
    # print(f'var NUMSIMYRS: {NUMSIMYRS}')
    # print(f'var PTOT_SC: {PTOT_SC}')
    # print(f'var PTOT_SF: {PTOT_SF}')
    # print( NUMSIMS + NUMSIMYRS *7 )

    return updated_args


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description = 'STOchastic Rainfall Modelling [STORM]')
    PARCE( parser )