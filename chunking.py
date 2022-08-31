# original script:
# https://bitbucket.csiro.au/projects/CMAR_RS/repos/netcdf-tools/browse/chunking/chunk_shape_3D.py?at=97c17c836371b07d9d08c9ffa05b3a8db623e0f1

# CHUNK_3D: Tries to work out best chunking shape
# Author: Russ Rew, Unidata

# updated version on: August 06.2018 -> for the script to run in Python3x
# updated version on: August 08.2022 -> brooming off & fixing bugs (STORM2.0)
# updates made by: Manuel F.


import numpy as np
from operator import mul
from functools import reduce


def BINLIST(n, width=0):
    """
    Returns list of bits that represent a non-negative integer
    n      -> non-negative integer
    width  -> number of bits in returned zero-filled list (default 0)
    """
    return list(map( int, list( bin(n)[2:].zfill(width) ) ))


def PERTURB_SHAPE(shape, onbits):
    """
    Returns shape perturbed by adding 1 to elements corresponding to 1 bits in onbits
    shape  -> list of variable dimension sizes
    onbits -> non-negative integer less than 2**len(shape)
    """
    return list(map( sum, zip( shape, BINLIST(onbits, len(shape)) ) ))


def CHUNK_3D(vorShape, valSize=8, chunkSize=4096):
    #varShape=[108, 4612800, 6];valSize=2;chunkSize=512
    #varShape=[1, 5, 27];valSize=8
    """
    FIRST OF ALL: find the Physical Sector Size (of your partition)
    chunkSize == [4096 Bytes] -> WOS (in this case)
    """
# https://unix.stackexchange.com/a/2669     (find sector size)
# https://unix.stackexchange.com/a/179013   (optimize sector size)
# https://stackoverflow.com/a/9465556/5885810   (sector size WOS)
# https://stackoverflow.com/a/36872873/5885810  (sector size WOS)
    """
    varShape  -> length 3 list of variable dimension sizes
    chunkSize -> maximum chunksize desired, in bytes    (default 4096)
    valSize   -> size/type of data values,  in bytes    (default 8   )
        e.g., a 64-bit float/integer is represented as 'f8'/'i8' (byte 'notation')
    Returns integer lengths of a chunk-shape that provides balanced access of 1D,
    and 2D subsets of a netCDF or HDF5 variable var with shape (V1, V2, V3).
    'Good shape' for chunks means that the number of chunks accessed to read are
    approximately equal, and the size of each chunk (uncompressed) is no more
    than chunkSize, which is often a disk block (Physical Sector) size.
    """

# https://stackoverflow.com/a/42387839/5885810  (why not np.r_)
    #varShape = np.r_[ vorShape ].astype('u8')
    varShape = np.asarray( vorShape ).astype('f8')
# this is a special case of n-dimensional function chunk_shape
    rank = len(varShape)
    chunkVals = chunkSize /valSize                      # ideal number of values in a chunk
    numChunks = (varShape.cumprod()[-1]) /chunkVals     # ideal number of chunks
    maxpos = varShape.argmax()
# https://stackoverflow.com/a/47670187/5885810          # swapping
    varShape[0], varShape[maxpos] = varShape[maxpos], varShape[0]
    """
    radical   -> is strongly influenced by the dimension of the array.
    for more details check:
        https://www.unidata.ucar.edu/blogs/developer/en/entry/chunking_data_choosing_shapes
    here I only explore up to 3D...
        for larger dims, I have no clue what the 'radical' should be.
    """
    radical = rank+1 if rank==3 else rank
    axisChunks = numChunks**(1 /radical)                # ideal number of chunks along each 2D axis

    if varShape.__getitem__(0) /axisChunks**(rank -1) < 1.0:
        chunkDim = 1.0                                  # each chunk shape.dim must be at least 1
        axisChunks = axisChunks /np.sqrt(varShape.__getitem__(0) /axisChunks**(rank -1))
    else:
        chunkDim = varShape.__getitem__(0) //axisChunks**(rank -1)\
            if rank!=1 else varShape.__getitem__(0) //axisChunks**(rank)
        chunkDim = min(chunkDim, chunkVals)             # for large dims and low space

    cFloor = []
    cFloor.append( chunkDim )
# factor to increase other dims if some must be increased to 1.0
    prod = 1.0

    if cFloor[0] == chunkVals:
        for i in range(1, rank): cFloor.append(1.0)
    else:
        for i in range(1, rank):
            if varShape.__getitem__(i) /axisChunks < 1.0:
                #prod *= axisChunks /varShape.__getitem__(i)
                prod = prod *axisChunks /varShape.__getitem__(i)
# # seems NOT.TO.DEVIATE too much from the original... but try: CHUNK_3D([3100,500,7],valSize=2)
#         for i in range(1, rank):
#             if varShape.__getitem__(i) /axisChunks < 1.0:
                chunkDim = 1.0
            else:
                chunkDim = (prod *varShape.__getitem__(i)) //axisChunks
            cFloor.append( chunkDim )

    if rank==1:
        cFloor[0] = min(cFloor.__getitem__(0), varShape.__getitem__(0))
    """
    cFloor is typically too small (numVals(cFloor) < chunkSize). Adding 1 to each
    shape.dim results in chunks that are too large, (numVals(cCeil) > chunkSize).
    Want to just add 1 to some of the axes to get as close as possible to
    chunkSize without exceeding it.
    Here we use brute force, compute numVals(cCand) for all 2**rank candidates
    and return the one closest to chunkSize without exceeding it.
    """
    bestChunkSize = 0
    cBest = cFloor

    for i in range(2**rank):
        cCand = PERTURB_SHAPE(cFloor, i) if rank!=1 else cFloor
# Returns number of values in chunk of specified shape, given by a list of dimension lengths
        thisChunkSize = valSize *reduce(mul, cCand)
        #thisChunkSize = valSize *reduce(mul, cCand) if cCand else valSize
        if bestChunkSize < thisChunkSize <= chunkSize:
            bestChunkSize = thisChunkSize
            cBest = list( cCand )                       # make a copy of best candidate

    fin = np.fromiter(map(int, cBest), dtype='int')
    fin[maxpos], fin[0] = fin[0], fin[maxpos]           # swapping

# if INPUT is smaller than CHUNK OUTPUT -> returns INPUT!!
    return vorShape if np.less_equal(vorShape, fin).all() == True\
        else fin.astype('u8').tolist()