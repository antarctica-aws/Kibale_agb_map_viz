"""
Reads the field plot CSV and outputs plots.js for use by index.html.

Outputs agb_map_viz/plots.js — a JS variable PLOT_DATA: array of objects,
one per unique plot location, with 2017 and/or 2020 values.

Run with:
  /home/users/mliang77/.micromamba/envs/gdal-env/bin/python prepare_plots.py
"""
import csv
import json
import os

CSV = os.path.join(os.path.dirname(__file__),
    'pred_icc_agbd_all_plots_years_val_EF2_2sig_no5_628_coarse_bilinear.csv')
OUT = os.path.join(os.path.dirname(__file__), 'plots.js')

YEARS = ('2017', '2020')

with open(CSV, newline='') as f:
    rows = list(csv.DictReader(f))

# Filter to target years only
rows = [r for r in rows if r['measure_year'] in YEARS]

# Group by Plot_ID; coords are stable across years for the same plot
from collections import defaultdict
by_plot = defaultdict(dict)
coords  = {}

for r in rows:
    pid  = r['Plot_ID']
    year = r['measure_year']
    lng  = float(r['coords.x1'])
    lat  = float(r['coords.x2'])
    coords[pid] = (lat, lng)
    try:
        pagbd = float(r['plot_agbd'])
    except (ValueError, KeyError):
        pagbd = None
    try:
        pred  = float(r['pred_agbd'])
    except (ValueError, KeyError):
        pred  = None
    by_plot[pid][year] = {'plot_agbd': pagbd, 'pred_agbd': pred}

plot_data = []
for pid, yr_data in sorted(by_plot.items(), key=lambda x: int(x[0])):
    lat, lng = coords[pid]
    entry = {
        'id':   pid,
        'lat':  round(lat, 7),
        'lng':  round(lng, 7),
    }
    for y in YEARS:
        if y in yr_data:
            entry['y' + y] = {
                'plot_agbd': round(yr_data[y]['plot_agbd'], 3) if yr_data[y]['plot_agbd'] is not None else None,
                'pred_agbd': round(yr_data[y]['pred_agbd'], 3) if yr_data[y]['pred_agbd'] is not None else None,
            }
        else:
            entry['y' + y] = None
    plot_data.append(entry)

js = 'var PLOT_DATA = ' + json.dumps(plot_data, separators=(',', ':')) + ';\n'
with open(OUT, 'w') as f:
    f.write(js)

n17  = sum(1 for p in plot_data if p['y2017'])
n20  = sum(1 for p in plot_data if p['y2020'])
both = sum(1 for p in plot_data if p['y2017'] and p['y2020'])
print(f'plots.js written: {len(plot_data)} plots  ({n17} with 2017, {n20} with 2020, {both} with both)')
