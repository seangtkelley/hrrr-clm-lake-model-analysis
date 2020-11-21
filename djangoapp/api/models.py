from django.db import models


class Lake(models.Model):

    # name of lake
    name = models.CharField(max_length=256)
    # id for north america hydrography shapefile
    uident = models.CharField(max_length=32, default=None, null=True)
    # id for new hampshire hydrography shapefile
    gnis_id = models.CharField(max_length=32, default=None, null=True)
    # geometry in geojson espg:4326
    geojson = models.TextField()
    
    # indices of hrrr model points corresponding to lake
    hrrr_gridpoints = models.TextField()
    # indices of sport sst points corresponding to lake
    sst_gridpoints = models.TextField()


class HRRRPred(models.Model):

    # lake point is located in
    lake = models.ForeignKey('Lake', on_delete=models.PROTECT)
    # index of point in HRRR grid
    grid_idx = models.CharField(max_length=16)
    # lon, lat
    lon = models.DecimalField(max_digits=7, decimal_places=4)
    lat = models.DecimalField(max_digits=7, decimal_places=4)
    # HRRR Cycle Date and Hour (UTC)
    fcst_datetime = models.DateTimeField()
    # HRRR Forecast Projection (UTC)
    pred_datetime = models.DateTimeField()
    # deg C
    water_temp = models.DecimalField(max_digits=4, decimal_places=2)


class SSTPred(models.Model):

    # lake point is located in
    lake = models.ForeignKey('Lake', on_delete=models.PROTECT)
    # index of point in SST grid
    grid_idx = models.CharField(max_length=16)
    # lon, lat
    lon = models.DecimalField(max_digits=7, decimal_places=4)
    lat = models.DecimalField(max_digits=7, decimal_places=4)
    # SST Cycle Date and Hour (UTC)
    datetime = models.DateTimeField()
    # deg C
    water_temp = models.DecimalField(max_digits=4, decimal_places=2)


class Station(models.Model):

    # unique string id for lookup
    str_id = models.CharField(max_length=16, default=None, null=True)
    # station or buoy name (e.g. Seneca Lake Water Quality Buoy)
    name = models.CharField(max_length=256)
    # owner (e.g. Hobart and William Smith Colleges, Finger Lake Institute)
    owner = models.CharField(max_length=256, default=None, null=True)
    # person of contact (e.g Dr. John Halfman)
    contact_name = models.CharField(max_length=256, default=None, null=True)
    # person of contact email
    contact_email = models.CharField(max_length=256, default=None, null=True)
    # web access
    url = models.CharField(max_length=256, default=None, null=True)

    # lake station is located in
    lake = models.ForeignKey('Lake', on_delete=models.PROTECT)
    # location description (e.g. mid-lake, offshore of Clark's Point)
    loc_desc = models.CharField(max_length=256, default=None, null=True)
    # lon, lat
    lon = models.DecimalField(max_digits=7, decimal_places=4)
    lat = models.DecimalField(max_digits=7, decimal_places=4)
    # lake surface elevation (m)
    elevation = models.DecimalField(max_digits=4, decimal_places=1, default=None, null=True)
    # water depth at station/buoy location (m)
    water_depth = models.DecimalField(max_digits=4, decimal_places=1, default=None, null=True)
    # type of water temp sensor (e.g. YSI/Xylen EXO2 Water Quality Logger)
    sensor_type = models.CharField(max_length=256, default=None, null=True)
    # frequency of observation (e.g. hourly)
    ob_freq = models.CharField(max_length=256, default=None, null=True)


class Ob(models.Model):
    
    # station relation
    station = models.ForeignKey('Station', on_delete=models.PROTECT)
    # name of sensor at station
    sensor_name = models.CharField(max_length=256, default=None, null=True)
    # date and time of observation (UTC)
    datetime = models.DateTimeField()
    # water temp (deg C)
    water_temp = models.DecimalField(max_digits=4, decimal_places=2)
    # depth of measurement (m)
    ob_depth = models.DecimalField(max_digits=4, decimal_places=1)