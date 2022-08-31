


def READ():
    import argparse
    import parse_input
    global argsup
    parser = argparse.ArgumentParser(description = 'STOchastic Rainfall Modelling [STORM]')#, prog='STORM')
# updated args
    argsup = parse_input.PARCE( parser )
    #print( argsup )


def TEST():
    from check_input import ASSERT, WELCOME, PAR_UPDATE
    global NC_NAMES
    PAR_UPDATE( argsup )
    ASSERT()
    NC_NAMES = WELCOME()

def RAIN():
    from rainfall import STORM, PAR_UPDATE
    # print('\nalpha')
    PAR_UPDATE( argsup )
    # print('\nbeta')
    STORM( NC_NAMES )

if __name__ == '__main__':
    READ()
    TEST()
    RAIN()