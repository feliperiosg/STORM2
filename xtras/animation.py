import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from cartopy.feature import ShapelyFeature
from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
#import imageio.v2 as imageio
from PIL import Image
from tqdm import tqdm
from pathlib import Path

# https://treyhunner.com/2018/12/why-you-should-be-using-pathlib/
ROOT_DIR = Path(__file__).resolve().parent.parent


#%% INPUT PARAMETERS

SHP_FILE = './model_input/shp/WG_Boundary.shp'
nc_file  = './model_output/SIM_220901T1142_S1_ptotC_stormsC.nc'
ncgroup  = 'simulation_02'
png_tag  = 'animation_i'


#%% FUNCTIONS' DEFINITION

#~ opens the NC.FILE and creates the aggregated fields ~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def COLLECT():
    global da, suma, EPSG, catchmnt, x_coord, y_coord, x_res, y_res, lim_da, limsum, DATE
# read nc.data
    ds = xr.open_dataset( ROOT_DIR.joinpath( nc_file ), group=ncgroup,
                         mask_and_scale=True, decode_coords="coordinates")
    EPSG = ds.crs.attrs['EPSG_code']
# https://stackoverflow.com/a/64127140/5885810  (string partial match)
    rainvar = list(filter(lambda x: x.startswith( 'year_' ), list(ds.keys())))
    # ['year_2022', 'year_2023', 'year_2024']
# we're interested just in "year_2022"
    da = ds[ rainvar.__getitem__(0) ]
# we're interested in "time_01"
    dim_da = list(da.coords.keys())
# we do cumulative maps (along the time-dimension)
    suma = da.where(ds.mask!=0, np.nan).cumsum(dim=dim_da.__getitem__(-1), skipna=False)

    # da[:,10,10]
    # np.nancumsum(da[:,10,10])
    # suma[:,10,10]

# extract coordinates
    y_coord = da[dim_da[0]].data
    x_coord = da[dim_da[1]].data
    y_res = abs(np.unique(np.diff( y_coord ))).__getitem__(0)
    x_res = abs(np.unique(np.diff( x_coord ))).__getitem__(0)
# find limits (for color bar)
    max_da = da.max().data
    maxsum = suma.max().data
    lim_da = [0, np.ceil(max_da/100) *100]
    limsum = [0, np.ceil(maxsum/50) *50]

# dates
    DATE = list(map(lambda x: x.item().strftime('%Y.%m.%d %H:%M:%S'),
                    da[dim_da.__getitem__(-1)].data.astype('datetime64[s]')))
# read SHP.file
    catchmnt = ShapelyFeature(Reader( ROOT_DIR.joinpath( SHP_FILE ) ).geometries(),
                              ccrs.PlateCarree(), facecolor='none')


# from pyproj import Transformer
# transformer = Transformer.from_crs(EPSG, 'EPSG:4326')
# lons, lats = transformer.transform( x_tick, np.repeat(y_tick[0], len(x_tick)))


#~ exponential/scientific format (for colorbar labels) ~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def eFORMAT(x, pos):
    a, b = '{:.0e}'.format(x).split('e')
    b = int(b)
    return r'{}$\mathrm{{E}}^{{{}}}$'.format(a,b)


#~ pimps up the plot ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def LAY_OUT(NUM, ZZ, LIMS, TIKS, KOLOR, LOG, LAT, LAB):
    ax = plt.subplot(NUM, projection=ccrs.Projection(EPSG))
    ax.set_aspect(aspect='equal')
    ax.spines['geo'].set_visible( False )
    ax.xlabels_top = ax.ylabels_right = False  #;  gl.xlines = False
# add some grid
    if LAT == 1:
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.4,
            color='xkcd:aqua green', linestyle='dotted', alpha=1.)
        gl.right_labels = False
        gl.bottom_labels = False
        gl.xlocator = mticker.FixedLocator( np.linspace(-110.2, -109.9, 4) )
        gl.ylocator = mticker.FixedLocator( np.linspace(31.65, 31.8, 4) )
        gl.xformatter = LongitudeFormatter( direction_label='inout', degree_symbol='°',
            number_format='.1f', transform_precision=1e-08, dms=False, auto_hide=False )
        gl.yformatter = LatitudeFormatter( direction_label='inout', degree_symbol='°',
            number_format='.2f', transform_precision=1e-08, dms=False, auto_hide=False )
        gl.xlabel_style = {'size':9, 'color':'xkcd:army green', 'weight':'light', 'ha':'left'} #'visible':True
        gl.ylabel_style = {'size':9, 'color':'xkcd:army green', 'weight':'light', 'va':'top'}
        gl.xpadding = -0.1
        gl.ypadding = -0.1
    ax.set_extent([x_coord[0]-x_res/2, x_coord[-1]+x_res/2,
                   y_coord[0]-y_res/2, y_coord[-1]+y_res/2], crs=ccrs.CRS(EPSG))
    ax.add_feature(catchmnt, edgecolor='xkcd:barbie pink', linewidth=0.9, zorder=10)
# add time.stamp
    ax.text(0.01, 0.97, LAB, color='xkcd:electric pink', fontsize=12, fontweight='normal',
        horizontalalignment='left', va='center', clip_on=True, transform=ax.transAxes)
# use a logarithmic scale
    if LOG==1:
        normo = mpl.colors.SymLogNorm(linthresh=0.01, linscale=0.075, vmin=LIMS[0], vmax=LIMS[1])
        formo = mpl.ticker.FuncFormatter( eFORMAT )
    else:
        normo = mpl.colors.Normalize(vmin=LIMS[0], vmax=LIMS[1])
        formo = f'%.{0}f'
    cs = ax.pcolormesh(x_coord, y_coord, ZZ, norm=normo, cmap=KOLOR,
                       transform=ccrs.Projection(EPSG))
# add colorbar
    cbar = plt.colorbar(cs, shrink=2/3, aspect=27, orientation='vertical', pad=.025,
                        ticks=TIKS, extend='max', extendfrac=0.02, format=formo, ax=ax) #,spacing='proportional'
    # cbar.set_clim(0.1,1)                      # to.limit.the.MAP.COLORS
# https://stackoverflow.com/a/27672236/5885810  (no border color bar)
    cbar.outline.set_visible( False )
    cbar.ax.yaxis.set_label_position('left')    # cbar.ax.xaxis.set_label_position('top')
    cbar.ax.tick_params(labelsize=12, direction='inout', color='xkcd:blood orange',
                        pad=1, width=0.39, bottom='False')
    return cbar


#~ defines the layout and calls the 'plotting'.function (LAY_OUT) ~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def PLOT( IDX ):#IDX=91#IDX=62
    fig = plt.figure(figsize=(10*1.1, 10), dpi=150)
    fig.tight_layout(pad=0)
    ax1 = LAY_OUT(NUM=211, ZZ=da[IDX,:], LIMS=lim_da, TIKS=[0,.02,.1,.2,.5,2,5,10,50,100,300],
                  KOLOR=plt.cm.nipy_spectral_r, LOG=1, LAT=0, LAB=DATE[IDX])
    ax1.set_label('storm-event rainfall  [mm]', fontsize=12)
    ax2 = LAY_OUT(NUM=212, ZZ=suma[IDX,:], LIMS=limsum, TIKS=np.linspace(limsum[0], limsum[-1], 7),
                  KOLOR=plt.cm.gist_ncar_r, LOG=0, LAT=1, LAB='')
    ax2.set_label('seasonal aggregated rainfall  [mm]', fontsize=12)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.1, hspace=+0.02)
    plt.savefig(ROOT_DIR.joinpath( f'./xtras/{png_tag}{"{:03d}".format(IDX)}.png' ),
    # plt.savefig(ROOT_DIR.joinpath( f'./xtras/{png_tag}{"{:03d}".format(91)}.png' ),
                bbox_inches='tight' ,pad_inches=0.01)
    plt.close()
    plt.clf()


#~ creates the GIF ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def GIF():
# GIF resizing.factor
    DOWNSCALE = 2/3
# locate output.GIF
    outpng = nc_file.split('/').__getitem__(-1).replace('.nc',
        f'--{ncgroup}--{DATE.__getitem__(0)[:4]}.gif')
    outpng = ROOT_DIR.joinpath( f'./xtras/{outpng}' )
# collect the images to compse the GIF
    images = list( ROOT_DIR.joinpath( f'./xtras/' ).glob(f'{png_tag}*') )
# https://stackoverflow.com/a/57751793/5885810  (GIF using Pillow)
# https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
# https://note.nkmk.me/en/python-pillow-image-resize/
    gif = []
    for f in images:
        img = Image.open( f )
# COMMENT.OUT THE LINE BELOW -> FOR PRESERVING 'ORIGINAL' GIF RESOLUTION
        # img = img.resize(
        #     tuple(map( int, np.round((img.size[0] *DOWNSCALE, img.size[1] *DOWNSCALE),0) ))
        #     # ,Image.Resampling.LANCZOS
        #     )
        gif.append( img )
    gif[0].save(fp=outpng, format='GIF', append_images=gif[1:],
                save_all=True, duration=360, loop=0)
# deleting pns
    list(map(Path.unlink, images))


#%%

if __name__ == '__main__':
    COLLECT()
    print('\nplotting pngs...')
    # PLOT( 91 )
    [PLOT( x ) for x in tqdm( range(len( DATE )), ncols=50 )]
    print('making GIF...', end='', flush=True)
    GIF()
    print(' done!')