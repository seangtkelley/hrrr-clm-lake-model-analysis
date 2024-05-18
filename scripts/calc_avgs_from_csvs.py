import os, sys
from decimal import Decimal

import numpy as np
import pandas as pd
import django

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'djangoapp'))
os.environ["DJANGO_SETTINGS_MODULE"] = "config.settings"
os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"
django.setup()
from lib import utils


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

csv_dir = os.path.join(utils.DATA_DIR, 'sst_extract', 'csv')

avgs = {
    "Lake Name": list(sorted(lake_uidents.keys()))
}
for name, _ in sorted(lake_uidents.items()):
    lake_csv_dir = os.path.join(csv_dir, name.replace(" ", ""))

    day_avg_cache = []
    for csv_filename in sorted(os.listdir(lake_csv_dir)):
        csv_filepath = os.path.join(lake_csv_dir, csv_filename)

        # load data
        lake_data_df = pd.read_csv(csv_filepath)

        # extend cache with this hour's preds
        day_avg_cache.extend(list(lake_data_df['water_temp']))

        # calculate average
        if "1800" in csv_filename:
            # avg all of day's preds
            lake_avg = pd.Series(day_avg_cache).dropna().mean()

            # truncate float
            lake_avg = Decimal(f"{lake_avg:.2f}")
        
            # add average to dict
            col_name = csv_filename.split('_')[1] # date
            if col_name not in avgs:
                avgs[col_name] = [ lake_avg ]
            else:
                avgs[col_name].append(lake_avg)
            
            # reset cache
            day_avg_cache = []

avgs_df = pd.DataFrame.from_dict(avgs)
avgs_df.to_csv(os.path.join(csv_dir, "lake_averages_from_csvs.csv"), index=False)