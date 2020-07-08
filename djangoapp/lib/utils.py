import os, sys
from datetime import date, datetime, timedelta
import pytz
import requests
import subprocess
import multiprocessing as mp
import time
import pathlib
from decimal import Decimal

from netCDF4 import Dataset
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import contextily as ctx
from pyproj import Transformer
from tqdm import tqdm
import magic
from django.conf import settings

from api import models
from config.settings import PROJECT_BASE_DIR


user = settings.DATABASES['default']['USER']
password = settings.DATABASES['default']['PASSWORD']
database_name = settings.DATABASES['default']['NAME']
database_url = f'postgresql://{user}:{password}@localhost:5432/{database_name}'

DATA_DIR = os.path.join(PROJECT_BASE_DIR, 'data')

# load usa shape file
usa_states = gpd.read_file(os.path.join(DATA_DIR, 'states_21basic', 'states.shp'))


def get_state_bounds(abbr):
    # get bounds for state and convert to epsg:3857

    state_bounds = usa_states[usa_states.STATE_ABBR == abbr].total_bounds.reshape((2, 2))

    transformer = Transformer.from_crs("epsg:4326", "epsg:3857")

    state_bounds[0] = transformer.transform(state_bounds[0,1]-0.1, state_bounds[0,0]-0.1)
    state_bounds[1] = transformer.transform(state_bounds[1,1]+0.1, state_bounds[1,0]+0.1)
    
    return state_bounds


def get_lake_gridpoints_with_meta_from_landuse():
    metadata = Dataset(os.path.join(DATA_DIR, "hrrrv4.geo_em.d01.nc"), "r", format="NETCDF4")

    land_use_points = np.squeeze(metadata.variables['LU_INDEX'][:].data)

    lake_points = []
    for i in range(len(land_use_points)):
        if len(np.where(land_use_points[i] == 21)[0]) != 0:
            col_idxs = np.where(land_use_points[i] == 21)[0]
            lake_points.extend( [ [i, col_idx] for col_idx in col_idxs ] )

    lons = np.squeeze(metadata.variables['CLONG'][:].data)
    lats = np.squeeze(metadata.variables['CLAT'][:].data)
    depths = np.squeeze(metadata.variables['LAKE_DEPTH'][:].data)

    lake_lons = [ lons[tuple(point)] for point in lake_points ]
    lake_lats = [ lats[tuple(point)] for point in lake_points ]
    lake_depths = [ depths[tuple(point)] for point in lake_points ]

    metadata.close()
    return {
        'points': lake_points,
        'geom': [ Point(xy) for xy in zip(lake_lons, lake_lats) ],
        'depths': lake_depths
    }


def intersect(coords):
        mask = coords.intersects(usa_states.unary_union)
        time.sleep(1)
        return mask

def get_lake_gridpoints_with_meta_from_landmask():
    print("Loading lake points from landmask...")

    # load metadata file
    metadata = Dataset(os.path.join(DATA_DIR, "hrrrv4.geo_em.d01.nc"), "r", format="NETCDF4")

    lake_points_filepath = os.path.join(DATA_DIR, 'lake_points_from_landmask.npy')
    if os.path.isfile(lake_points_filepath):
        with open(lake_points_filepath, 'rb') as f:
            lake_points = np.load(f)

        lons = np.squeeze(metadata.variables['CLONG'][:].data)
        lats = np.squeeze(metadata.variables['CLAT'][:].data)

        lake_lons = [ lons[tuple(point)] for point in lake_points ]
        lake_lats = [ lats[tuple(point)] for point in lake_points ]

        lake_geom = [ Point(xy) for xy in zip(lake_lons, lake_lats) ]

    else:

        land_mask = np.squeeze(metadata.variables['LANDMASK'][:].data)
        lons = np.squeeze(metadata.variables['CLONG'][:].data)
        lats = np.squeeze(metadata.variables['CLAT'][:].data)

        # get water points
        water_points = []
        water_geom = []
        for i in range(len(land_mask)):
            if len(np.where(land_mask[i] == 0)[0]) != 0:

                col_idxs = np.where(land_mask[i] == 0)[0]
                col_water_points = [ [i, col_idx] for col_idx in col_idxs ]

                water_lons = [ lons[tuple(point)] for point in col_water_points ]
                water_lats = [ lats[tuple(point)] for point in col_water_points ]

                water_points.append(col_water_points)
                water_geom.append(gpd.GeoSeries([ Point(xy) for xy in zip(water_lons, water_lats) ]))

        # determine which water points lie within inland usa
        inland_water_mask = []
        with mp.Pool(mp.cpu_count()-2) as p:
            with tqdm(total=len(water_geom)) as pbar:
                for i, out in enumerate(p.imap(intersect, water_geom)):
                    inland_water_mask.append(out)
                    pbar.update()

        # filter water points using mask
        lake_points = []
        lake_geom = []
        for i in range(len(inland_water_mask)):
            lake_points.extend( [ water_points[i][j] for j in range(len(inland_water_mask[i])) if inland_water_mask[i][j] ] )
            lake_geom.extend( [ water_geom[i][j] for j in range(len(inland_water_mask[i])) if inland_water_mask[i][j] ] )

        # save to file
        with open(lake_points_filepath, 'wb') as f:
            np.save(f, lake_points)

    # get lake depth
    depths = np.squeeze(metadata.variables['LAKE_DEPTH'][:].data)
    lake_depths = [ depths[tuple(point)] for point in lake_points ]

    metadata.close()

    return {
        'points': lake_points,
        'geom': lake_geom,
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
    lake_meta = get_lake_gridpoints_with_meta_from_landuse()

    # build dir
    base_dir = os.path.join('hrrrX', 'sfc')

    # file host
    grib2_host = 'https://pando-rgw01.chpc.utah.edu'

    fcst_datetimes, pred_datetimes = [], []

    curr_date = start_date
    while curr_date <= end_date:
        date_str = curr_date.strftime("%Y%m%d")
        date_dir = os.path.join(base_dir, date_str)

        if not os.path.isdir(os.path.join(DATA_DIR, date_dir)):
            os.mkdir(os.path.join(DATA_DIR, date_dir))

        for cycle in cycle_hours:
            for pred in pred_hours:

                fcst_datetime = pytz.utc.localize(datetime.fromordinal(curr_date.toordinal()) + timedelta(hours=cycle))
                pred_datetime = pytz.utc.localize(datetime.fromordinal(curr_date.toordinal()) + timedelta(hours=cycle+pred))

                # keep list of all datatimes for retrieval later
                fcst_datetimes.append(fcst_datetime)
                pred_datetimes.append(pred_datetime)

                # check if already in database
                if models.HRRRPred.objects.filter(fcst_datetime=fcst_datetime, pred_datetime=pred_datetime).exists():
                    continue

                print(f"Retrieving {fcst_datetime}")

                # fill meta
                data = {
                    'grid_idx': [ str(point) for point in lake_meta['points'] ],
                    'lon': [ geom.x for geom in lake_meta['geom'] ],
                    'lat': [ geom.y for geom in lake_meta['geom'] ],
                    'fcst_datetime': [fcst_datetime]*len(lake_meta['points']),
                    'pred_datetime': [pred_datetime]*len(lake_meta['points']),
                }

                # build file info
                filename = f'hrrrX.t{cycle:02}z.wrfsfcf{pred:02}'
                nc4_filepath = os.path.join(DATA_DIR, date_dir, filename+'.nc4' )

                # if not exist, attempt to retrieve grib2 and convert
                if not os.path.isfile(nc4_filepath):
                    # download grib2
                    grb2_filepath = os.path.join(DATA_DIR, date_dir, filename+'.grib2')
                    if not os.path.isfile(grb2_filepath) or 'xml' in magic.from_file(grb2_filepath, mime=True):
                        url = os.path.join(grib2_host, date_dir, filename+'.grib2')
                        download_url_prog(url, grb2_filepath)

                        if 'xml' in magic.from_file(grb2_filepath, mime=True):
                            print(f"{filename} not avaiable from {grib2_host}")

                            # fill nans
                            data['water_temp'] = [Decimal('NaN')]*len(lake_meta['points'])

                    # convert to netCDF4
                    convert_cmd = ["wgrib2", grb2_filepath, "-nc4", "-netcdf", nc4_filepath ]
                    result = subprocess.Popen(convert_cmd, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
                    out, rc = result.communicate()[0], result.returncode

                    if rc != 0:
                        print(f"Failed to convert {filename} to netcdf")
                        
                        # fill nans
                        data['water_temp'] = [Decimal('NaN')]*len(lake_meta['points'])

                # ensure nc4 file exists
                if os.path.isfile(nc4_filepath):
                    # load dataset
                    try:
                        hrrr_output = Dataset(nc4_filepath, "r", format="NETCDF4")
                        
                        # get temps
                        water_temps = np.squeeze(hrrr_output.variables['TMP_surface'][:].data)
                        lake_water_temps = np.array([ water_temps[tuple(point)] for point in lake_meta['points'] ])

                        # convert K to C
                        lake_water_temps = lake_water_temps - 272.15

                        # load lake points as geodataframe
                        data['water_temp'] = [ Decimal(temp.item()) for temp in lake_water_temps ]

                        hrrr_output.close()

                    except Exception as e:
                        print(f"Failed to read data from {nc4_filepath}")
                        print(e)

                        # fill nans
                        data['water_temp'] = [Decimal('NaN')]*len(lake_meta['points'])
                    
                # failsafe (no nc4 and no grib2)
                else:
                    # fill nans
                    data['water_temp'] = [Decimal('NaN')]*len(lake_meta['points'])

                # save to db
                for i in range(len(lake_meta['points'])):
                    record = models.HRRRPred(**{ key: values[i] for key, values in data.items() })
                    record.save()

        curr_date += timedelta(days=1)
    
    # load all data from db
    qs = models.HRRRPred.objects.filter(fcst_datetime__in=fcst_datetimes, pred_datetime__in=pred_datetimes).order_by('fcst_datetime', 'pred_datetime')
    df = pd.DataFrame.from_records(qs.values())
    return gpd.GeoDataFrame(df, crs='epsg:4326', geometry=[ Point(xy) for xy in zip(df.lon, df.lat) ])