import os, sys
from datetime import date, datetime, timedelta
import pytz
import requests
import subprocess
import multiprocessing as mp
import time
import pathlib
from decimal import Decimal
import json

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
USA_STATES = gpd.read_file(os.path.join(DATA_DIR, 'states_21basic', 'states.shp'))


#
#   Metadata
#

def load_metadata_into_db():

    # read insitu stations file
    station_meta = pd.read_csv(os.path.join(DATA_DIR, '2020-HRRRxCLM-SPoRT-Insitu-Points-Matchup-V2.csv'), dtype={
        'NA Lakes Shapefile UIDENT': str,
        'NH Lakes Shapefile GNIS_ID': str
    })
    station_meta = station_meta.dropna(how='all')

    # find north america lake shapes
    na_lakes = gpd.read_file(os.path.join(DATA_DIR, 'na_lakes/hydrography_p_lakes_v2/hydrography_p_lakes_v2.shp'))
    na_lakes = na_lakes.to_crs(epsg=4326)

    # find new hampshire lake shapes
    nh_lakes = gpd.read_file(os.path.join(DATA_DIR, 'na_lakes/NH Water Bodies/NH Water Bodies.shp'))
    nh_lakes = nh_lakes.to_crs(epsg=4326)

    # load hrrr metadata
    hrrr_lake_meta = get_hrrr_lake_gridpoints_with_meta_from_landuse()
    df = pd.DataFrame(hrrr_lake_meta)
    hrrr_lake_meta = gpd.GeoDataFrame(df, crs='epsg:4326', geometry=[ Point(xy) for xy in zip(df.lon, df.lat) ])

    # load any sst file for metadata
    sst_water_meta = get_sst_water_gridpoints()
    df = pd.DataFrame(sst_water_meta)
    sst_water_meta = gpd.GeoDataFrame(df, crs='epsg:4326', geometry=[ Point(xy) for xy in zip(df.lon, df.lat) ])

    # load lakes
    for i, station in station_meta[station_meta['In-Situ Obs Avail'].str.contains('Y')].iterrows():
        lake_record = None
        if isinstance(station['NA Lakes Shapefile UIDENT'], str):
            # check if lake already in db
            if models.Lake.objects.filter(uident=station['NA Lakes Shapefile UIDENT']).exists():
                lake_record = models.Lake.objects.filter(uident=station['NA Lakes Shapefile UIDENT']).first()
            else:
                lake_meta = na_lakes[na_lakes['UIDENT'] == int(station['NA Lakes Shapefile UIDENT'])]
                lake_bounds = lake_meta['geometry'].unary_union

                uident = station['NA Lakes Shapefile UIDENT']
                gnis_id = None

        elif isinstance(station['NH Lakes Shapefile GNIS_ID'], str):
            # check if lake already in db
            if models.Lake.objects.filter(gnis_id=station['NH Lakes Shapefile GNIS_ID']).exists():
                lake_record = models.Lake.objects.filter(gnis_id=station['NH Lakes Shapefile GNIS_ID']).first()
            else:
                # TODO: see if nh_lakes['GNIS_ID'] is int
                lake_meta = nh_lakes[nh_lakes['GNIS_ID'] == station['NH Lakes Shapefile GNIS_ID']]
                lake_bounds = lake_meta['geometry'].unary_union

                uident = None
                gnis_id = station['NH Lakes Shapefile GNIS_ID']

        else:
            continue

        print(station['Name of Lake Forecast Location'])

        # create lake record
        if not lake_record:
            # get hrrr indices
            if 'lake' in hrrr_lake_meta.columns:
                hrrr_lake_meta.drop(labels=['lake'], axis="columns", inplace=True)
            hrrr_lake_meta['lake'] = hrrr_lake_meta.within(lake_bounds)

            # get sst indices
            if 'lake' in sst_water_meta.columns:
                sst_water_meta.drop(labels=['lake'], axis="columns", inplace=True)
            sst_water_meta['lake'] = sst_water_meta.within(lake_bounds)

            lake_record = models.Lake(
                name=station['Lake Name'],
                uident=uident,
                gnis_id=gnis_id,
                geojson=str(gpd.GeoSeries([lake_bounds]).__geo_interface__),
                hrrr_gridpoints=';'.join(map(str, hrrr_lake_meta[hrrr_lake_meta['lake']]['grid_idx'])),
                sst_gridpoints=';'.join(map(str, sst_water_meta[sst_water_meta['lake']]['grid_idx']))
            )
            lake_record.save()

        # create station record
        station_record = models.Station(
            name = f"{station['Name of Lake Forecast Location']}, {station['Source of Obs']}",
            url = station['URL or POC'] if 'http' in station['URL or POC'] else "",
            contact_name= station['URL or POC'] if ('http' not in station['URL or POC']) and ('@' not in station['URL or POC']) else "",
            contact_email= station['URL or POC'] if '@' in station['URL or POC'] else "",
            lake = lake_record,
            loc_desc = station['Name of Lake Forecast Location'].split('-')[-1],
            lon = station['Lon (dec)'],
            lat = station['Lat (dec)']
        )
        station_record.save()

#
#   HRRR Data
#

def get_hrrr_lake_gridpoints_with_meta_from_landuse():
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
        'grid_idx': lake_points,
        'lon': lake_lons,
        'lat': lake_lats,
        'depth': lake_depths
    }


def get_us_intersect_mask(coords):
    mask = coords.intersects(USA_STATES.unary_union)
    time.sleep(1)
    return mask

def get_hrrr_lake_gridpoints_with_meta_from_landmask():
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
                for i, out in enumerate(p.imap(get_us_intersect_mask, water_geom)):
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
        'grid_idx': lake_points,
        'lon': [ geom.x for geom in lake_geom ],
        'lat': [ geom.y for geom in lake_geom ],
        'depth': lake_depths
    }


def download_url_prog(url, filepath):
    req = requests.get(url, stream=True)
    file_size = int(req.headers['Content-Length'])
    chunk_size = 1024  # 1 MB
    num_bars = int(file_size / chunk_size)

    with open(filepath, 'wb') as fp:
        for chunk in tqdm(req.iter_content(chunk_size=chunk_size), total=num_bars, unit='KB', desc=filepath, leave=True, file=sys.stdout):
            fp.write(chunk)


def get_hrrrx_output_for_lake(lake, start_date, end_date, cycle_hours=list(range(24)), pred_hours=list(range(32))):

    # create list of lake points
    hrrr_gridpoints = lake.hrrr_gridpoints.split(';')

    # build hrrrX dir
    base_dir = os.path.join('hrrrX', 'sfc')

    # file host
    grib2_host = 'https://pando-rgw01.chpc.utah.edu'

    # keep list of all datatimes for retrieval later
    fcst_datetimes, pred_datetimes = [], []
    
    # fill meta
    lake_meta = get_hrrr_lake_gridpoints_with_meta_from_landuse()
    lake_points_str = list(map(str, lake_meta['grid_idx']))
    data = {
        'lake': [lake]*len(hrrr_gridpoints),
        'grid_idx': hrrr_gridpoints,
        'lon': [ str(lake_meta['lon'][lake_points_str.index(point)]) for point in hrrr_gridpoints ],
        'lat': [ str(lake_meta['lat'][lake_points_str.index(point)]) for point in hrrr_gridpoints ]
    }

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
                fcst_datetimes.append(fcst_datetime)
                pred_datetimes.append(pred_datetime)

                # check if already in database
                if models.HRRRPred.objects.filter(lake=lake, fcst_datetime=fcst_datetime, pred_datetime=pred_datetime).exists():
                    continue

                print(f"Retrieving {fcst_datetime}")

                # fill meta
                data['fcst_datetime'] = [fcst_datetime]*len(hrrr_gridpoints)
                data['pred_datetime'] = [pred_datetime]*len(hrrr_gridpoints)

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
                            data['water_temp'] = [Decimal('NaN')]*len(hrrr_gridpoints)

                    # convert to netCDF4
                    convert_cmd = ["wgrib2", grb2_filepath, "-nc4", "-netcdf", nc4_filepath ]
                    result = subprocess.Popen(convert_cmd, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
                    out, rc = result.communicate()[0], result.returncode

                    if rc != 0:
                        print(f"Failed to convert {filename} to netcdf")
                        
                        # fill nans
                        data['water_temp'] = [Decimal('NaN')]*len(hrrr_gridpoints)

                # ensure nc4 file exists
                if os.path.isfile(nc4_filepath):
                    # load dataset
                    try:
                        hrrr_output = Dataset(nc4_filepath, "r", format="NETCDF4")
                        
                        # get temps
                        water_temps = np.squeeze(hrrr_output.variables['TMP_surface'][:].data)
                        lake_water_temps = np.array([ water_temps[tuple(eval(point))] for point in hrrr_gridpoints ])

                        # convert K to C
                        lake_water_temps = lake_water_temps - 272.15

                        # load lake points as geodataframe
                        data['water_temp'] = [ Decimal(temp.item()) for temp in lake_water_temps ]

                        hrrr_output.close()

                    except Exception as e:
                        print(f"Failed to read data from {nc4_filepath}")
                        print(e)

                        # fill nans
                        data['water_temp'] = [Decimal('NaN')]*len(hrrr_gridpoints)
                    
                # failsafe (no nc4 and no grib2)
                else:
                    # fill nans
                    data['water_temp'] = [Decimal('NaN')]*len(hrrr_gridpoints)

                # save to db
                for i in range(len(hrrr_gridpoints)):
                    record = models.HRRRPred(**{ key: values[i] for key, values in data.items() })
                    record.save()

        curr_date += timedelta(days=1)
    
    # load all data from db
    qs = models.HRRRPred.objects.filter(lake=lake, fcst_datetime__in=fcst_datetimes, pred_datetime__in=pred_datetimes).order_by('fcst_datetime', 'pred_datetime')
    df = pd.DataFrame.from_records(qs.values())
    return gpd.GeoDataFrame(df, crs='epsg:4326', geometry=[ Point(xy) for xy in zip(df.lon, df.lat) ])


#
#   SPoRT SST Analysis
#

def get_sst_water_gridpoints():
    # load any dataset
    sst_metadata = Dataset(os.path.join(DATA_DIR, "sst_nowcoast/202006/20200601_0600_sport_nhemis_sstcomp_2km_unscaled.nc"), "r", format="NETCDF4")

    water_mask = sst_metadata.variables['Band1'][:].mask
    water_points = []
    for i in range(1300, 2800):
        for j in range(2800, 6100):
            if water_mask[i][j] == 0:
                water_points.append([i, j])

    lats = sst_metadata.variables['lat'][:].data
    lons = sst_metadata.variables['lon'][:].data

    sst_metadata.close()
    return {
        'grid_idx': water_points,
        'lon': [ lons[tuple(point)[1]] for point in water_points ],
        'lat': [ lats[tuple(point)[0]] for point in water_points ]
    }


def get_sst_output_for_lake(lake, start_date, end_date, cycle_hours=[6, 18]):

    # create list of lake points
    sst_gridpoints = lake.sst_gridpoints.split(';')

    # build sst dir
    base_dir = os.path.join('sst_nowcoast')

    # keep list of all datatimes for retrieval later
    dts = []

    # fill meta
    lake_meta = get_sst_water_gridpoints()
    lake_points_str = list(map(str, lake_meta['grid_idx']))
    data = {
        'lake': [lake]*len(sst_gridpoints),
        'grid_idx': sst_gridpoints,
        'lon': [ str(lake_meta['lon'][lake_points_str.index(point)]) for point in sst_gridpoints ],
        'lat': [ str(lake_meta['lat'][lake_points_str.index(point)]) for point in sst_gridpoints ]
    }

    curr_date = start_date
    while curr_date <= end_date:
        date_str = curr_date.strftime("%Y%m")
        date_dir = os.path.join(base_dir, date_str)

        if not os.path.isdir(os.path.join(DATA_DIR, date_dir)):
            # folder for month not available, just continue to next month
            print(f"{date_str} data not available.")
            curr_date = curr_date.replace(month=1 if curr_date.month==12 else curr_date.month+1, day=curr_date.day)
            continue

        for cycle in cycle_hours:

            dt = pytz.utc.localize(datetime.fromordinal(curr_date.toordinal()) + timedelta(hours=cycle))
            dts.append(dt)

            # check if already in database
            if models.SSTPred.objects.filter(lake=lake, datetime=dt).exists():
                print(f"{dt} already in db.")
                continue

            print(f"Retrieving {dt}")

            data['datetime'] = [dt]*len(sst_gridpoints)

            # build file info
            filename = f'{curr_date.strftime("%Y%m%d")}_{cycle:02}00_sport_nhemis_sstcomp_2km'
            nc4_filepath = os.path.join(DATA_DIR, date_dir, filename+'.nc' )

            # if not exist, attempt to retrieve grib2 and convert
            if not os.path.isfile(nc4_filepath):
                # find tif file
                tif_filepath = os.path.join(DATA_DIR, date_dir, filename+'.tif')
                if not os.path.isfile(tif_filepath):
                    print(f"{tif_filepath} missing.")

                    # fill nans
                    data['water_temp'] = [Decimal('NaN')]*len(sst_gridpoints)
                else:
                    # convert to netCDF4
                    convert_cmd = ["gdal_translate", tif_filepath, nc4_filepath, "-ot", "Float32", "-of", "netcdf", "-co", "COMPRESS=LZW", "-co", "TILED=yes", "-scale", "0", "255", "25.232", "117.032"]
                    result = subprocess.Popen(convert_cmd, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
                    out, rc = result.communicate()[0], result.returncode

                    if rc != 0:
                        print(f"Failed to convert {filename} to netcdf")
                        
                        # fill nans
                        data['water_temp'] = [Decimal('NaN')]*len(sst_gridpoints)

            # ensure nc4 file exists
            if os.path.isfile(nc4_filepath):
                # load dataset
                try:
                    sst_output = Dataset(nc4_filepath, "r", format="NETCDF4")
                    
                    # get temps
                    water_temps = np.squeeze(sst_output.variables['Band1'][:].data)
                    lake_water_temps = np.array([ water_temps[tuple(eval(point))] for point in sst_gridpoints ])

                    # convert F to C
                    lake_water_temps = (lake_water_temps - 32)*(5/9)

                    # load lake points as geodataframe
                    data['water_temp'] = [ Decimal(temp.item()) for temp in lake_water_temps ]

                    sst_output.close()

                except Exception as e:
                    print(f"Failed to read data from {nc4_filepath}")
                    print(e)

                    # fill nans
                    data['water_temp'] = [Decimal('NaN')]*len(sst_gridpoints)
                
            # failsafe (no nc4 and no grib2)
            else:
                # fill nans
                data['water_temp'] = [Decimal('NaN')]*len(sst_gridpoints)

            # save to db
            for i in range(len(sst_gridpoints)):
                record = models.SSTPred(**{ key: values[i] for key, values in data.items() })
                record.save()

        curr_date += timedelta(days=1)

    # load all data from db
    qs = models.SSTPred.objects.filter(lake=lake, datetime__in=dts).order_by('datetime')
    df = pd.DataFrame.from_records(qs.values())
    return gpd.GeoDataFrame(df, crs='epsg:4326', geometry=[ Point(xy) for xy in zip(df.lon, df.lat) ])


#
#   Observations
#

def load_usgs_seneca_lake_data():

    # file host
    csv_host = 'https://pando-rgw01.chpc.utah.edu'

    filename_base = 'USGSBuoy-20200608-20200620-SenecaLake'

    csv_filepath = os.path.join(DATA_DIR, 'USGSBuoy', filename_base+'.csv')
    if not os.path.isfile(csv_filepath):
        # url = os.path.join(grib2_host, date_dir, filename+'.grib2')
        # download_url_prog(url, grb2_filepath)

        print('Download not implemented for csv.')
        json_filepath = os.path.join(DATA_DIR, filename_base+'.json')
        if os.path.isfile(json_filepath):
            # open json
            with open('../data/insitu/USGSBuoy-20200608-20200620-SenecaLake.json') as f:
                data = json.load(f)
            
            # get station
            if models.Station.objects.filter(name="USGS Buoy, Seneca Lake - Upper").exists():
                station = models.Station.objects.filter(name="USGS Buoy, Seneca Lake - Upper").first()
            else:
                print("Couldn't find Seneca station in db.")
                return

            # save to db
            for i in range(3):
                for j in range(1, len(data[i]['data'])):
                    dt = pytz.utc.localize(datetime.utcfromtimestamp(data[i]['data'][j][0]/1000))
                    if not models.Ob.objects.filter(sensor_name=data[i]['name'], datetime=dt).exists():
                        record = models.Ob(
                            station=station,
                            sensor_name=data[i]['name'],
                            datetime=dt,
                            water_temp=Decimal(f"{data[i]['data'][j][1]:.2f}"),
                            ob_depth=Decimal('NaN')
                        )
                        record.save()
            
        else:
            print('JSON file must be manually downloaded.')


#
#   General Utils
#

def get_state_bounds(abbr):
    # get bounds for state and convert to epsg:3857

    state_bounds = USA_STATES[USA_STATES.STATE_ABBR == abbr].total_bounds.reshape((2, 2))

    transformer = Transformer.from_crs("epsg:4326", "epsg:3857")

    state_bounds[0] = transformer.transform(state_bounds[0,1]-0.1, state_bounds[0,0]-0.1)
    state_bounds[1] = transformer.transform(state_bounds[1,1]+0.1, state_bounds[1,0]+0.1)
    
    return state_bounds


def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))


# https://stackoverflow.com/questions/30112202/how-do-i-find-the-closest-values-in-a-pandas-series-to-an-input-number
def find_neighbours(df, field, value):
    exactmatch = df[df[field] == value]
    if not exactmatch.empty:
        return exactmatch.index
    else:
        lowerneighbour_ind = df[df[field] < value][field].idxmax() if len(df[df[field] < value][field]) > 1 else 0
        upperneighbour_ind = df[df[field] > value][field].idxmin() if len(df[df[field] > value][field]) > 1 else 0
        return [lowerneighbour_ind, upperneighbour_ind]