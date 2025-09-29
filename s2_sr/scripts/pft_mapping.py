import numpy as np
import pandas as pd
import os
import glob
import rioxarray as rxr
import xarray as xr
import pickle
import logging
from osgeo import gdal
from datetime import datetime
import sys
from shapely.geometry import box

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y-%H%M%S")

#########################################################################
# Parameters
#########################################################################

# general params
BASE = '/mnt/poseidon/remotesensing/arctic/data'
OUT_DIR = f'{BASE}/rasters/model_results_tiled_toolik_09-22-2025'
DATA_DIR = f'{BASE}/rasters'
REF_RAST = f'{DATA_DIR}/esa_sentinel/s2_sr/tmp/out/S2_B02_AOI.tif'
MODEL = f'{BASE}/training/Test_06/results/03'

# gdalwarp params
DST_NODATA = 255
XRES = 0.000179663056824
YRES = 0.000179663056824
EPSG = 'EPSG:4326'
bbox = box(-150, 67, -148, 69)

# specify pkl file names and map to associated PFT
pft_file_map = {
    "bryophyte": {
        "model": "bryophyte_30m_parent_sources5_IQR3.pkl",
        "outfile_suffix": "30M-5P-IQR3",
    },
    "lichen": {
        "model": "lichen_55m_parent_sources3_IQR2.5.pkl",
        "outfile_suffix": "55M-3P-IQR2.5",
    },
    "deciduous shrub": {
        "model": "deciduous shrub_30m_child_sources3_IQR2.pkl",
        "outfile_suffix": "30M-3C-IQR2",
    },
    "evergreen shrub": {
        "model": "evergreen shrub_30m_parent_sources4_IQR2.5.pkl",
        "outfile_suffix": "30M-4P-IQR2.5",
    },
    "forb": {
        "model": "forb_55m_child_sources5_IQR1.5.pkl",
        "outfile_suffix": "55M-5C-IQR1.5",
    },
    "graminoid": {
        "model": "graminoid_30m_parent_sources3_IQR2.pkl",
        "outfile_suffix": "30M-3P-IQR2",
    },
    "non-vascular": {
        "model": "non-vascular_30m_parent_sources2_IQR1.5.pkl",
        "outfile_suffix": "30M-2P-IQR1.5",
    },
    "litter": {
        "model": "litter_30m_parent_sources2_IQR3.pkl",
        "outfile_suffix": "30M-2P-IQR3",
    },
}

# Sensor-specific Params
S2_DIR = f'{DATA_DIR}/esa_sentinel/s2_sr/tmp/out/S2_*.tif'
S1_DIR = f'{DATA_DIR}/esa_sentinel/s2_sr/tmp/out/S1_*.tif'
DEM_DIR = f'{DATA_DIR}/esa_sentinel/s2_sr/tmp/out/DEM_*.tif'

# logging configuration
os.makedirs(OUT_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    filename=f'{OUT_DIR}/std_{dt_string}.log',
    filemode='w',
    format='%(asctime)s >>> %(message)s',
    datefmt='%d-%b-%y %H:%M:%S'
)

#########################################################################
# Definitions
#########################################################################

# function to stack sensor bands for one gridcell
# will need to loop through each sensor and gridcell
def stack_bands(band_paths, ref_rast, scale_factor=None):
    
    """
    Creates an xarray with each band recorded as a variable.
    bands_paths    : [list] list of file paths to each band
    ref_rast       : [xr.Dataset] raster used as the model resolution/crs
    scale_factor   : [float or int] number multiplied to rescale data
    Returns an xr.Dataset with x,y,band dimensions for one gridcell with 
    each band as a data variable that matches the resolution/scale of the
    reference raster.
    """

    raster_bands = []
    for band_path in band_paths:

        # get file name from file path
        print(band_path)
        filename = os.path.basename(band_path)
        b_name = filename.split('_')[-2]  # assumes format <source>_<band>_AOI.tif
        
        # open raster in xarray
        raster = rxr.open_rasterio(band_path)
        raster.name = b_name

        # set nodata values to nan and rescale
        raster = raster.rio.reproject_match(ref_rast)
        raster = raster.where(raster != 0, -9999)
        if scale_factor is not None:
            raster = raster * scale_factor

        raster_bands.append(raster)

    merged = xr.merge(raster_bands)
    merged = merged.dropna(dim='band', how='any')
    return merged
    
# function that creates new veg idx data variables for an xr
def calc_veg_idx_s2(xrd):
    
    """
    Creates new data attributes for an s2_sr xr.Dataset with bands
    B2, B3, B4, B5, B6, B8, B8A, B11, and B12. Second step after 
    stack_bands. S2_sr data must be scaled from 0 to 1; can set
    scale factor in stack_bands function if necessary.
    xrd : [xr.Dataset] s2_sr xarray dataset
    Returns: xarray dataset with new vegetation indices
    """
    
    xrd = xrd.assign(ndwi1 = lambda x: (x.nir - x.swir1)/(x.nir + x.swir2))
    xrd = xrd.assign(ndwi2 = lambda x: (x.nir - x.swir2)/(x.nir + x.swir2))
    xrd = xrd.assign(msavi = lambda x: (2*x.nir + 1 - ((2*x.nir + 1)**2 - 8*(x.nir - x.red))**0.5) * 0.5)
    xrd = xrd.assign(vari = lambda x: (x.green - x.red)/(x.green + x.red - x.blue))
    xrd = xrd.assign(rvi = lambda x: x.nir/x.red)
    xrd = xrd.assign(osavi = lambda x: 1.16 * (x.nir - x.red)/(x.nir + x.red + 0.16))
    xrd = xrd.assign(tgi = lambda x: (120 * (x.red - x.blue) - 190 * (x.red - x.green))*0.5)
    xrd = xrd.assign(gli = lambda x: (2 * x.green - x.red - x.blue)/(2 * x.green + x.red + x.blue))
    xrd = xrd.assign(ngrdi = lambda x: (x.green - x.red)/(x.green + x.red))
    xrd = xrd.assign(ci_g = lambda x: x.nir/x.green - 1)
    xrd = xrd.assign(gNDVI = lambda x: (x.nir - x.green)/(x.nir + x.green))
    xrd = xrd.assign(cvi = lambda x: (x.nir * x.red)/(x.green ** 2))
    xrd = xrd.assign(mtvi2 = lambda x: 1.5*(1.2*(x.nir - x.green) - 2.5*(x.red - x.green))/(((2*x.nir + 1)**2 - (6*x.nir - 5*(x.red**0.5))-0.5)**0.5))
    xrd = xrd.assign(brightness = lambda x: 0.3037 * x.blue +0.2793 * x.green +0.4743 * x.red +0.5585 * x.nir +0.5082 * x.swir1 + 0.1863 * x.swir2)
    xrd = xrd.assign(greenness = lambda x: 0.7243 * x.nir +0.0840 * x.swir1 - 0.2848 * x.blue - 0.2435 * x.green - 0.5436 * x.red - 0.1800 * x.swir2)
    xrd = xrd.assign(wetness = lambda x: 0.1509 * x.blue+0.1973* x.green+0.3279*x.red+0.3406*x.nir-0.7112*x.swir1 - 0.4572*x.swir2)
    xrd = xrd.assign(tcari = lambda x: 3 * ((x.redEdge1 - x.red)-0.2 * (x.redEdge1 - x.green)*(x.redEdge1/x.red)))
    xrd = xrd.assign(tci = lambda x: 1.2 * (x.redEdge1 - x.green)- 1.5 * (x.red - x.green)*((x.redEdge1/x.red)**0.5))
    xrd = xrd.assign(nari = lambda x: (1/x.green - 1/x.redEdge1)/(1/x.green + 1/x.redEdge1))

    return xrd
    
#########################################################################
# Begin modeling
#########################################################################

# use an existing raster to ensure all rasters are aligned
reference_raster = rxr.open_rasterio(REF_RAST)
reference_raster = reference_raster.where(reference_raster != 0, -9999)
reference_raster = reference_raster * .0001


#########################################################################
# Sentinel 2 (nodata = -9999)
#########################################################################

# create 20-m xarray raster
s2_band_rasters = sorted(glob.glob(S2_DIR))
s2_stacked_raster = stack_bands( 
    s2_band_rasters,
    reference_raster,
    scale_factor = .0001,
)

# rename bands to something legible
s2_stacked_raster = s2_stacked_raster.rename({
    'B02':'blue', 
    'B03':'green',
    'B04':'red', 
    'B05':'redEdge1', 
    'B06':'redEdge2', 
    'B07':'redEdge3', 
    'B8A':'redEdge4', 
    'B08':'nir',
    'B11':'swir1',
    'B12':'swir2'})

# calculate vegetation indices
s2_stacked_raster_veg = calc_veg_idx_s2(s2_stacked_raster)


#########################################################################
# Sentinel 1 (nodata = -9999)
#########################################################################

# create 20-m xarray raster
s1_band_rasters = sorted(glob.glob(S1_DIR))
rescale_bands = ['VV', 'VH']
s1_stacked_raster = stack_bands(
    s1_band_rasters,
    reference_raster,
)
s1_stacked_raster = s1_stacked_raster.where(s1_stacked_raster[rescale_bands] != 0, -9999)


#########################################################################
# Arctic DEM (nodata = -9999)
#########################################################################

# create 20-m xarray raster
dem_band_rasters = sorted(glob.glob(DEM_DIR))
dem_stacked_raster = stack_bands(
    dem_band_rasters, 
    reference_raster,
)


#########################################################################
# Combine into one xarray
#########################################################################

# make sure pandas df features are in the right order
stacked_raster = xr.merge([s2_stacked_raster_veg, 
                            s1_stacked_raster, 
                            dem_stacked_raster])

# get coordinate information from raster as df
df = stacked_raster.to_dataframe()
coords = df.reset_index()
coords = coords[['x', 'y']]

# get raster data as df
df = df.droplevel([1, 2]).reset_index(drop=True)
df = df.iloc[:,1:]
df = df.replace(-9999, np.nan)

# find any bands that were divided by 0 and produced an inf value
bad_idx_list = df[np.isinf(df.values)].index.tolist()
df.drop(index=bad_idx_list, inplace=True)
coords.drop(index=bad_idx_list, inplace=True)

# remove straggling nans
nan_idx_list = df[np.isnan(df.values)].index.tolist()
df.drop(index=nan_idx_list, inplace=True)
coords.drop(index=nan_idx_list, inplace=True)

#########################################################################
# Apply model
#########################################################################

for pft, pft_info in pft_file_map.items():

    # extract info from pft_info
    model = pft_info['model']
    outfile_suffix = pft_info['outfile_suffix']

    # define output file name
    pft_slug = pft.replace(" ", "_")
    outfile = f"{OUT_DIR}/{pft_slug}_{outfile_suffix}.tif"
    
    try:

        # get pickle path
        model_file_path = os.path.join(MODEL, "tunedModel_" + model)
        if not os.path.isfile(model_file_path):
            print(f"Model file not found: {model_file_path}. Exiting.")
            sys.exit(1)

        # load the pickled model
        with open(model_file_path, "rb") as f:
            model = pickle.load(f)

        # reorder df and predict
        col_order = model.feature_names_in_.tolist()
        df2 = df2[col_order]
        fcover = model.predict(df2)  # fcover is 1 x n
        fcover = (fcover * 100).astype(int)
        
    except Exception as e:
        print(f"Modeling failed for '{pft}': {e}")
        continue

    
    #########################################################################
    # Export modeled tif
    #########################################################################

    # set up df for xarray
    results = coords.copy()
    results['fcover'] = fcover
    results['band'] = 1
    
    # export xarray as tif
    try:
        results_xr = xr.Dataset.from_dataframe(results.set_index(['band', 'y', 'x']))
        xr_band = results_xr.isel(band=0).rio.write_crs('EPSG:4326')
        xr_band.rio.to_raster(outfile)
        print(f"Exported {outfile}")
        
    except Exception as e:
        print(f"Exception found while exporting '{pft}': {e}")
        continue
        
    # set crs
    opts = gdal.WarpOptions(
        format='GTiff', 
        dstSRS=EPSG, 
        dstNodata=DST_NODATA,
        outputBounds=bbox.bounds,
        outputType=gdal.GDT_Int16,
        resampleAlg='bilinear',
        xRes=XRES,
        yRes=YRES,
        targetAlignedPixels=True,
    )
    gdal.Warp(outfile, outfile, options=opts)