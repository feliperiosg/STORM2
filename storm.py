#~ executes the PARSER (first), so no need to upload all.libs to ask.4.help ~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def READ():
    import argparse
    import parse_input
    global argsup
    parser = argparse.ArgumentParser(description = 'STOchastic Rainfall Modelling [STORM]')
# updated args
    argsup = parse_input.PARCE( parser )
    # print( argsup )


#~ checks the validity of the SOFT-CORE PARAMETERS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def TEST():
    from check_input import ASSERT, WELCOME, PAR_UPDATE
    global NC_NAMES
    PAR_UPDATE( argsup )
    ASSERT()
    NC_NAMES = WELCOME()


#~ all the heavy work is done in 'rainfall.py' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def RAIN():
    from rainfall import STORM, PAR_UPDATE
    PAR_UPDATE( argsup )
    STORM( NC_NAMES )


if __name__ == '__main__':
    READ()
    TEST()
    RAIN()