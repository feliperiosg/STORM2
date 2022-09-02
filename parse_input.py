from parameters import *


#%% FUNCTIONS' DEFINITION

#~ replace FILE.PARAMETERS with those read from the command line ~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def ARG_UPDATE( args ):
    for x in list(vars( args ).keys()):
        # # use this if you're not inside a (local) function
        # exec(f'if args.{x}:\n\t{x} = args.{x}')
# https://stackoverflow.com/a/2083375/5885810   (exec global... weird)
        exec(f'if args.{x}:\n\tglobals()["{x}"] = args.{x}')
    return args


# so one can input NONE from the command line
# https://stackoverflow.com/a/48295546/5885810  (None in argparse)
def none_too( v ):
    return None if v.lower() == 'none' else float( v )


#~ read parameters from the command line ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# https://www.geeksforgeeks.org/command-line-arguments-in-python/
# https://docs.python.org/3/library/argparse.html#the-add-argument-method
def PARCE( parser ):
# add the defaults
# https://stackoverflow.com/a/27616814/5885810  (parser case insensitive)
    parser.add_argument('-m', '--MODE', type=str.lower, default=MODE,
        choices=['simulation', 'validation'],
        help='Type of Run (case-insensitive) (default: %(default)s)')
    parser.add_argument('-w', '--SEASONS', type=int, default=SEASONS, choices=[1,2],
        help='Number of Seasons (per Run) (default: %(default)s)')
    parser.add_argument('-n', '--NUMSIMS', type=int, default=NUMSIMS,
        help='Number of runs per Season (default: %(default)s)')
    parser.add_argument('-y', '--NUMSIMYRS', type=int, default=NUMSIMYRS,
        help='Number of years per run (per Season) (default: %(default)s)')
    parser.add_argument('-ps', '--PTOT_SC', default=PTOT_SC, type=none_too, nargs='+',#type=float
        help='Relative change in the seasonal rain equally applied to every '\
            'simulated year. (one signed scalar per Season).')
    parser.add_argument('-pf', '--PTOT_SF', default=PTOT_SF, type=none_too, nargs='+',
        help='Relative change in the seasonal rain progressively applied to '\
            'every simulated year. (one signed scalar per Season).')
    parser.add_argument('-ss', '--STORMINESS_SC', default=STORMINESS_SC, type=none_too,
        nargs='+', help='Relative change in the observed intensity equally applied '\
            'to every simulated year. (one signed scalar per Season).')
    parser.add_argument('-sf', '--STORMINESS_SF', default=STORMINESS_SF, type=none_too,
        nargs='+', help='Relative change in the observed intensity progressively '\
            'applied to every simulated year. (one signed scalar per Season).')
    parser.add_argument('--version', action='version', version='STORM 2.0')#'%(prog)s 2.0')
# Read arguments from command line
    args = parser.parse_args()
    # print(args)
# REDEFINE variables by their names, instead of relative to 'args'
    updated_args = ARG_UPDATE( args )

    return updated_args


#%%

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description = 'STOchastic Rainstorm Model [STORM v2.0]')
    PARCE( parser )