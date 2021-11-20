import os, sys
import subprocess
from decimal import Decimal

from netCDF4 import Dataset
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import django

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'djangoapp'))
os.environ["DJANGO_SETTINGS_MODULE"] = "config.settings"
os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"
django.setup()
from lib import utils


# find north america lake shapes
na_lakes = gpd.read_file(os.path.join(utils.DATA_DIR, 'na_lakes/hydrography_p_lakes_v2/hydrography_p_lakes_v2.shp'))
na_lakes = na_lakes.to_crs(epsg=4326)

# load any sst file for metadata
sst_water_meta = utils.get_sst_water_gridpoints()
df = pd.DataFrame(sst_water_meta)
sst_water_meta = gpd.GeoDataFrame(df, crs='epsg:4326', geometry=[ Point(xy) for xy in zip(df.lon, df.lat) ])

"""
Lakes
	Lake Simcoe [Near City of Orillia] (Canada)
	Lake St. Clair (MI)
	Lake Champlain (VT/NY)
	Sebago Lake (ME)
	Lake Okeechobee (FL)
	Lake Pontchartrain (LA)
coastal lagoon	Intracoastal Waterway Padre Island (TX) -- not included
	Salton Sea (CA)
	Lake Tahoe (NV/CA)
	Great Salt Lake (UT)
	Utah Lake (UT)
	Bear Lake (UT/ID)
	Lake Sakakawea (ND)
	Lake Winnebago (WI)
	Lower Red Lake (MN)
	Lake of the Woods (MN)
	Lake Manitoba (Canada)
	Lake Winnipeg (Canada)
	Lake Nipigon (Canada)
"""

# uidents for lakes (must be found manually)
lake_uidents = {
    "Lake Simcoe": 555502, 
    "Lake St. Clair": [556602, 194902],
	"Lake Champlain": 555002,
	"Sebago Lake": 314102,
	"Lake Okeechobee": 250102,
	"Lake Pontchartrain": 322602,
	"Salton Sea": 343302,
	"Lake Tahoe": 312402,
	"Great Salt Lake": 315502,
	"Utah Lake": 311702,
	"Bear Lake": 306202,
	"Lake Sakakawea": 293002,
	"Lake Winnebago": 310502,
	"Lower Red Lake": 292802,
	"Lake of the Woods": 536802,
	"Lake Manitobaz": 137402,
	"Lake Winnipeg": 111302,
	"Lake Nipigon": 535302
    }

tif_dir = os.path.join(utils.DATA_DIR, 'sst_extract', 'tif')
nc4_dir = os.path.join(utils.DATA_DIR, 'sst_extract', 'nc4')
csv_dir = os.path.join(utils.DATA_DIR, 'sst_extract', 'csv')

# create csv dir for results
if not os.path.exists(csv_dir):
    os.makedirs(csv_dir)

# convert geotiff to netcdf4 if needed
if not os.path.exists(nc4_dir):
    os.makedirs(nc4_dir)

    for tif_filename in os.listdir(tif_dir):
        tif_filepath = os.path.join(tif_dir, tif_filename)
        nc4_filepath = os.path.join(nc4_dir, tif_filename.split(".")[0]+".nc")

        # retrieve geotiff and convert to netCDF4
        convert_cmd = ["gdal_translate", tif_filepath, nc4_filepath, "-ot", "Float32", "-of", "netcdf", "-co", "COMPRESS=LZW", "-co", "TILED=yes", "-scale", "0", "255", "25.232", "117.032"]
        result = subprocess.Popen(convert_cmd, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
        out, rc = result.communicate()[0], result.returncode

        if rc != 0:
            print(f"Failed to convert {tif_filename} to netcdf")
            continue

# load lakes
avgs = {
    "Lake Name": list(sorted(lake_uidents.keys()))
}
for name, uident in sorted(lake_uidents.items()):
    # get lake meta from na_lakes df
    if isinstance(uident, list):
        lake_meta = na_lakes[na_lakes['UIDENT'].isin(uident)]
    else:
        lake_meta = na_lakes[na_lakes['UIDENT'] == uident]
    lake_bounds = lake_meta['geometry'].unary_union

    # get sst indices
    if 'lake' in sst_water_meta.columns:
        sst_water_meta.drop(labels=['lake'], axis="columns", inplace=True)
    sst_water_meta['lake'] = sst_water_meta.within(lake_bounds)

    # extract gridpoints
    sst_gridpoints = list(map(str, sst_water_meta[sst_water_meta['lake']]['grid_idx']))
    sst_lon = list(map(str, sst_water_meta[sst_water_meta['lake']]['lon']))
    sst_lat = list(map(str, sst_water_meta[sst_water_meta['lake']]['lat']))

    # for each nc4 file, extract data
    day_avg_cache = []
    for nc4_filename in sorted(os.listdir(nc4_dir)):
        nc4_filepath = os.path.join(nc4_dir, nc4_filename)

        # load dataset
        try:
            sst_output = Dataset(nc4_filepath, "r", format="NETCDF4")
            
            # get temps
            water_temps = np.squeeze(sst_output.variables['Band1'][:].data)
            lake_water_temps = np.array([ water_temps[tuple(eval(point))] for point in sst_gridpoints ])

            # load lake points as Decimal
            lake_water_temps = [ Decimal(f"{temp.item():.2f}") for temp in lake_water_temps ]

            sst_output.close()

        except Exception as e:
            print(f"Failed to read data from {nc4_filepath}")
            print(e)
            continue

        lake_data_df = pd.DataFrame({
            'sst_gridpoint': sst_gridpoints,
            'lon': sst_lon,
            'lat': sst_lat,
            'water_temp': lake_water_temps
        })

        # replace 25.23 with np.nan
        # -3.76 C == 25.232 F is the lower bound for tiff pixel conversion to netcdf temp
        # it's basicall a zero value for that pixel in the image, which is taken to be invalid,
        # given it is below freezing.
        lake_data_df['water_temp'] = lake_data_df['water_temp'].replace(Decimal('25.23'), np.nan)

        lake_csv_dir = os.path.join(csv_dir, name.replace(" ", ""))
        # first time, create csv dir for lake data
        if not os.path.exists(lake_csv_dir):
            os.makedirs(lake_csv_dir)

        # write data to csv file
        lake_csv_filename = name.replace(" ", "") + "_" + nc4_filename.split(".")[0] + ".csv"
        lake_csv_filepath = os.path.join(lake_csv_dir, lake_csv_filename)
        lake_data_df.to_csv(lake_csv_filepath, index=False)

        # extend cache with this hour's preds
        day_avg_cache.extend(list(lake_data_df['water_temp']))

        # calculate average
        if "1800" in nc4_filename:
            # avg all of day's preds
            lake_avg = pd.Series(day_avg_cache).dropna().mean()
            lake_avg = Decimal(f"{lake_avg:.2f}")
        
            # add average to dict
            col_name = nc4_filename.split('_')[0] # date
            if col_name not in avgs:
                avgs[col_name] = [ lake_avg ]
            else:
                avgs[col_name].append(lake_avg)
            
            # reset cache
            day_avg_cache = []

avgs_df = pd.DataFrame.from_dict(avgs)
avgs_df.to_csv(os.path.join(csv_dir, "lake_averages.csv"), index=False)