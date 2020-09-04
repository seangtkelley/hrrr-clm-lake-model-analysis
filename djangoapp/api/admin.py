from django.contrib import admin
from .models import HRRRPred, SSTPred, Lake, Station, Ob


@admin.register(Lake)
class LakeAdmin(admin.ModelAdmin):
    list_display = ['name']
    ordering = ['name']

@admin.register(HRRRPred)
class HRRRPredAdmin(admin.ModelAdmin):
    list_display = ['lake', 'grid_idx', 'fcst_datetime', 'pred_datetime', 'water_temp']
    ordering = ['-fcst_datetime', '-pred_datetime']

@admin.register(SSTPred)
class SSTPredAdmin(admin.ModelAdmin):
    list_display = ['lake', 'grid_idx', 'datetime', 'water_temp']
    ordering = ['-datetime']

@admin.register(Station)
class StationAdmin(admin.ModelAdmin):
    list_display = ['name']
    ordering = ['name']


@admin.register(Ob)
class ObAdmin(admin.ModelAdmin):
    list_display = ['sensor_name', 'datetime']
    ordering = ['-datetime']