# HRRR/CLM Lake Model Analysis

## Requirements

1. Python 3
2. [wgrib2](https://www.cpc.ncep.noaa.gov/products/wesley/wgrib2/)
3. [GDAL](https://gdal.org/)
4. [NetCDF4](https://www.unidata.ucar.edu/software/netcdf/)
    - Debian: `sudo apt-get install libnetcdf-dev`
6. [PostgreSQL](https://www.postgresql.org/) and [PostGIS](https://docs.djangoproject.com/en/3.0/ref/contrib/gis/install/postgis/) (???)

## Setup

*TODO*: Conda instructions

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

## View analyses

1. (optional) Activate Python virtual environment
2. `jupyter notebook`
3. Navigate to the `notebooks` folder and click to open
