from django.contrib import admin
from .models import HRRRPred, Station, Ob

@admin.register(HRRRPred)
class HRRRPredAdmin(admin.ModelAdmin):
    list_display = ('grid_idx', 'fcst_datetime', 'pred_datetime', 'water_temp')
    ordering = ('-fcst_datetime', '-pred_datetime')


@admin.register(Station)
class StationAdmin(admin.ModelAdmin):
    pass


@admin.register(Ob)
class ObAdmin(admin.ModelAdmin):
    pass