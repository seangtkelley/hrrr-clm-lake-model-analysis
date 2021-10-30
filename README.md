# HRRRx/CLM Lake Model Analysis

## Quick Start

### Pre-reqs

1. Python 3
2. [wgrib2](https://www.cpc.ncep.noaa.gov/products/wesley/wgrib2/)
3. [GDAL](https://gdal.org/)
4. [NetCDF4](https://www.unidata.ucar.edu/software/netcdf/)
    - Debian: `sudo apt-get install libnetcdf-dev`
6. [PostgreSQL](https://www.postgresql.org/)

### Setup

1. (optional) Create a Python virtual environment: [venv](https://docs.python.org/3/library/venv.html)
    - personal recommendation: [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/)
2. Install pip packages: `pip install -r requirements.txt`
3. Configure Postgres in accordance with `djangoapp/config/setting.py`.
    - Create a database called `hrrr_clm_analysis`
    - Create a user called `hrrr_clm_analysis_user` and password `dev`
    - Give user permissions on database
4. Create `keys.json` with Django secret key: *TODO*: python-decouple + .env ???
    - `echo "{\"django_secret_key\": \"$(python -c 'from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())')\" }" >> djangoapp/keys.json`
5. Run django migrations: `python djangoapp/manage.py migrate`
6. Populate database with lake metadata: `python djangoapp/manage.py load_metadata_into_db`

### View analyses

1. (optional) Activate Python virtual environment
2. `jupyter notebook`
3. Navigate to the `notebooks` folder and click to open

## Introduction

### Lake Models

#### HRRRx

- explain forecast cycle vs prediction datetime

#### SPoRT SST 

- explain why no forecast cycle vs prediction datetime

### In-situ Observations

- provide a few examples of in-situ stations used and how observations are taken

## Technical Methodology

### Data Wranggling

Basic idea:
1. insection between gridpoints and lake shape to find that lake's points
2. save grid idx to db so costly intersection only needs to be performed once

#### Data Format

- file names on site

- NetCDF and Grib2 format
    - grid of pixels
- NetCDF easiest to read

- provide relevant NetCDF fields
    - HRRRx
    - SPoRT SST

#### Building Metadata

##### Lake DB Model

- short explanation of fields

##### Filtering for Lake Points

- landuse mask
    - specific number given for lake (inland water?)
- landmask mask
- simple window

simple window ends up being easiest and efficient enough. should use for HRRRx too

now we need to find the points for specific lakes

##### NA Lakes Shapefile

- explain/show example table of shapefile contents
- explain + code snippet of adding intersection field with geopandas

saved to db to save computation

#### Ingesting Forecast Output

##### Pred DB Models

- short explanation of fields
- mention forecast cycle vs prediction time 

##### Collection

- maybe quick explanation of the automatic searching for files?

## Analysis

- basic idea: compare closest 5 pred points with ob

### HRRRx issues

- explain experimental nature
- still working out bugs
- show graph of problem (i think it was consistent underestimation?)

### SST

#### Section for each important figure created with example

## Discussion

- can take from presentation
- also link to presentation

## Future Work

- more lakes
- more forecast cycles
- etc