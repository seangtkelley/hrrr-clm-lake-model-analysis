from django.db import models


class HRRRPred(models.Model):

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
    water_temp = models.DecimalField(max_digits=5, decimal_places=2)


class Station(models.Model):

    # station or buoy name (e.g. Seneca Lake Water Quality Buoy) 
    name = models.CharField(max_length=256)
    # owner (e.g. Hobart and William Smith Colleges, Finger Lake Institute)
    owner = models.CharField(max_length=256)
    # person of contact (e.g Dr. John Halfman)
    contact_name = models.CharField(max_length=256)
    # person of contact email
    contact_email = models.CharField(max_length=256)
    # lake name (e.g. Seneca Lake)
    lake_name = models.CharField(max_length=256)
    # location description (e.g. mid-lake, offshore of Clark's Point)
    loc_desc = models.CharField(max_length=256)
    # lon, lat
    lon = models.DecimalField(max_digits=7, decimal_places=4)
    lat = models.DecimalField(max_digits=7, decimal_places=4)
    # lake surface elevation (m)
    elevation = models.DecimalField(max_digits=4, decimal_places=1)
    # water depth at station/buoy location (m)
    water_depth = models.DecimalField(max_digits=4, decimal_places=1)
    # type of water temp sensor (e.g. YSI/Xylen EXO2 Water Quality Logger)
    sensor_type = models.CharField(max_length=256)
    # frequency of observation (e.g. hourly)
    ob_freq = models.CharField(max_length=256)


class Ob(models.Model):
    
    # station relation
    station = models.ForeignKey('Station', on_delete=models.PROTECT)
    # date and time of observation (UTC)
    datetime = models.DateTimeField()
    # water temp (deg C)
    water_temp = models.DecimalField(max_digits=5, decimal_places=2)
    # depth of measurement (m)
    ob_depth = models.DecimalField(max_digits=4, decimal_places=1)