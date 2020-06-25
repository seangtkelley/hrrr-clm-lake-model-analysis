import os, sys
from datetime import date, datetime, timedelta
import requests
import subprocess

from netCDF4 import Dataset
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import contextily as ctx
from pyproj import Transformer
from tqdm import tqdm
import magic


DATA_DIR = os.path.join('.', 'data')


def get_state_bounds(abbr, ):
    # get bounds for state and convert to epsg:3857

    usa_states = gpd.read_file('./data/states_21basic/states.shp')

    state_bounds = usa_states[usa_states.STATE_ABBR == abbr].total_bounds.reshape((2, 2))

    transformer = Transformer.from_crs("epsg:4326", "epsg:3857")

    state_bounds[0] = transformer.transform(state_bounds[0,1]-0.1, state_bounds[0,0]-0.1)
    state_bounds[1] = transformer.transform(state_bounds[1,1]+0.1, state_bounds[1,0]+0.1)
    
    return state_bounds


def get_lake_gridpoints_with_meta():
    metadata = Dataset("./data/hrrrv4.geo_em.d01.nc", "r", format="NETCDF4")

    land_use_points = np.squeeze(metadata.variables['LU_INDEX'][:].data)

    lake_points = []
    for i in range(len(land_use_points)):
        if len(np.where(land_use_points[i] == 21)[0]) != 0:
            col_idxs = np.where(land_use_points[i] == 21)[0]
            lake_points.extend( [ [i, col_idx] for col_idx in col_idxs ] )

    longs = np.squeeze(metadata.variables['CLONG'][:].data)
    lats = np.squeeze(metadata.variables['CLAT'][:].data)
    depths = np.squeeze(metadata.variables['LAKE_DEPTH'][:].data)

    lake_longs = [ longs[tuple(point)] for point in lake_points ]
    lake_lats = [ lats[tuple(point)] for point in lake_points ]
    lake_depths = [ depths[tuple(point)] for point in lake_points ]

    metadata.close()
    return {
        'points': lake_points,
        'longs': lake_longs,
        'lats': lake_lats,
        'depths': lake_depths
    }


def download_url_prog(url, filepath):
    req = requests.get(url, stream=True)
    file_size = int(req.headers['Content-Length'])
    chunk_size = 1024  # 1 MB
    num_bars = int(file_size / chunk_size)

    with open(filepath, 'wb') as fp:
        for chunk in tqdm(req.iter_content(chunk_size=chunk_size), total=num_bars, unit='KB', desc=filepath, leave=True, file=sys.stdout):
            fp.write(chunk)


def get_hrrrx_lake_output(start_date, end_date, cycle_hours=list(range(24)), pred_hours=list(range(32))):
    
    lakes_gdf = None

    # get lake grid points
    lake_meta = get_lake_gridpoints_with_meta()

    # build dir
    base_dir = os.path.join('hrrrX', 'sfc')

    # file host
    grib2_host = 'https://pando-rgw01.chpc.utah.edu'

    curr_date = start_date
    while curr_date <= end_date:
        date_str = curr_date.strftime("%Y%m%d")
        date_dir = os.path.join(base_dir, date_str)

        if not os.path.isdir(os.path.join(DATA_DIR, date_dir)):
            os.mkdir(os.path.join(DATA_DIR, date_dir))

        for cycle in cycle_hours:
            for pred in pred_hours:
                # build file meta
                filename = f'hrrrX.t{cycle:02}z.wrfsfcf{pred:02}'
                nc4_filepath = os.path.join(DATA_DIR, date_dir, filename+'.nc4' )

                if not os.path.isfile(nc4_filepath):
                    # download grib2
                    url = os.path.join(grib2_host, date_dir, filename+'.grib2')
                    grb2_filepath = os.path.join(DATA_DIR, date_dir, filename+'.grib2')
                    download_url_prog(url, grb2_filepath)

                    if 'xml' in magic.from_file(grb2_filepath, mime=True):
                        break

                    # convert to netCDF4
                    convert_cmd = ["wgrib2", grb2_filepath, "-nc4", "-netcdf", nc4_filepath ]
                    result = subprocess.Popen(convert_cmd, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
                    out, rc = result.communicate()[0], result.returncode

                    if rc != 0:
                        print(f"Failed to convert {filename} to netcdf")
                        break
                
                # load dataset
                hrrr_output = Dataset(nc4_filepath, "r", format="NETCDF4")
                
                # get temps
                water_temps = np.squeeze(hrrr_output.variables['TMP_surface'][:].data)
                air_temps = np.squeeze(hrrr_output.variables['TMP_2maboveground'][:].data)

                lake_water_temps = np.array([ water_temps[tuple(point)] for point in lake_meta['points'] ])
                lake_air_temps = np.array([ air_temps[tuple(point)] for point in lake_meta['points'] ])

                # convert K to C
                lake_water_temps = lake_water_temps - 272.15
                lake_air_temps = lake_air_temps - 272.15

                # load lake points as geodataframe
                df = pd.DataFrame(data={
                    'cycle_hour': [cycle]*len(lake_water_temps),
                    'fcst_datetime': datetime.fromordinal(curr_date.toordinal()) + timedelta(hours=cycle),
                    'pred_hour': [pred]*len(lake_water_temps),
                    'pred_datetime': datetime.fromordinal(curr_date.toordinal()) + timedelta(hours=pred),
                    'idx': [ str(point) for point in lake_meta['points'] ],
                    'depth': lake_meta['depths'],
                    'water_temp': lake_water_temps, 
                    'air_temp': lake_air_temps,
                })

                gdf = gpd.GeoDataFrame(df, crs='epsg:4326', geometry=[ Point(xy) for xy in zip(lake_meta['longs'], lake_meta['lats']) ])
                gdf = gdf.to_crs(epsg=3857)

                if lakes_gdf is None:
                    lakes_gdf = gdf
                else:
                    lakes_gdf = lakes_gdf.append(gdf, ignore_index=True)

                hrrr_output.close()

        curr_date += timedelta(days=1)
    
    return lakes_gdf