"""
Prepares PNG overlays (clipped to Kibale AOI) for the AGBD comparison website.

Outputs in agb_map_viz/:
  aef_2017.png / ls_2017.png / aef_2020.png / ls_2020.png  — coloured PNGs
  aef_change.png / ls_change.png                           — change PNGs
  aef_2017.dat … ls_change.dat                            — raw float32 arrays for click lookup
  bounds.js  — AOI bounds + layer shape metadata for index.html

Run with:
  /home/users/mliang77/.micromamba/envs/gdal-env/bin/python prepare_images.py
"""
import json
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.io import MemoryFile
from shapely.geometry import shape, mapping
from shapely.ops import unary_union, transform as shp_transform
from pyproj import Transformer
from PIL import Image
import matplotlib.colors as mcolors
import matplotlib
import pyproj

pyproj.datadir.set_data_dir('/home/users/mliang77/.micromamba/envs/gdal-env/share/proj')
matplotlib.use('Agg')

BASE = '/oak/stanford/groups/kailou/mliang77/AEF_GEDI_test'
OUT  = '/oak/stanford/groups/kailou/mliang77/AEF_GEDI_test/agb_map_viz'
AOI  = f'{BASE}/Kibale_40002.geojson'

VMIN_AGBD, VMAX_AGBD   = 0, 400      # Mg/ha for 2017 maps
VMIN_CHNG, VMAX_CHNG   = -150, 150   # Mg/ha for change maps
MAX_PX = 2048


# ── Load AOI geometries ──────────────────────────────────────────────────────
def load_aoi(crs_epsg=4326):
    """Return list of geometry dicts for rasterio.mask, in the requested CRS."""
    with open(AOI) as f:
        gj = json.load(f)
    geoms = [shape(feat['geometry']) for feat in gj['features']]
    aoi = unary_union(geoms)
    if crs_epsg != 4326:
        t = Transformer.from_crs(4326, crs_epsg, always_xy=True)
        aoi = shp_transform(t.transform, aoi)
    return [mapping(aoi)], aoi.bounds   # (minx, miny, maxx, maxy)


# ── Load + clip + reproject to EPSG:4326 ────────────────────────────────────
def load_clipped(path):
    """
    Returns (data_2d_float32, aoi_bounds_4326).
    Works for any input CRS (reprojects to 4326, then clips).
    """
    with rasterio.open(path) as src:
        src_epsg = src.crs.to_epsg()

        if src_epsg == 4326:
            geom_list, _ = load_aoi(4326)
            out, tr = rio_mask(src, geom_list, crop=True,
                               nodata=np.nan, all_touched=True)
            data = out[0].astype(np.float32)
            b = rasterio.transform.array_bounds(data.shape[0], data.shape[1], tr)
            bounds_4326 = (b[0], b[1], b[2], b[3])  # (W, S, E, N)

        else:
            # 1. Reproject entire raster to EPSG:4326 in memory
            dst_crs = 'EPSG:4326'
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds
            )
            profile = src.profile.copy()
            profile.update(crs=dst_crs, transform=transform,
                           width=width, height=height,
                           nodata=np.nan, dtype='float32')

            with MemoryFile() as mf:
                with mf.open(**profile) as dst:
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=rasterio.band(dst, 1),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.bilinear,
                    )
                # 2. Clip to AOI
                with mf.open() as dst:
                    geom_list, _ = load_aoi(4326)
                    out, tr = rio_mask(dst, geom_list, crop=True,
                                      nodata=np.nan, all_touched=True)

            data = out[0].astype(np.float32)
            b = rasterio.transform.array_bounds(data.shape[0], data.shape[1], tr)
            bounds_4326 = (b[0], b[1], b[2], b[3])

    # Mask non-positive as NaN (no data edges)
    data[data <= 0] = np.nan
    return data, bounds_4326


# ── Colour mapping → RGBA PNG ────────────────────────────────────────────────
def to_png(data, output_path, vmin, vmax, cmap_name):
    """Apply colourmap to 2-D float array and save as RGBA PNG (NaN = transparent)."""
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = matplotlib.colormaps[cmap_name]
    rgba = (cmap(norm(data)) * 255).astype(np.uint8)
    rgba[~np.isfinite(data)] = 0   # transparent where NaN

    img = Image.fromarray(rgba, mode='RGBA')
    if max(img.width, img.height) > MAX_PX:
        scale = MAX_PX / max(img.width, img.height)
        img = img.resize((int(img.width * scale), int(img.height * scale)),
                         Image.LANCZOS)
    img.save(output_path, optimize=True)
    print(f'  Saved {output_path}  ({img.width}×{img.height})')


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    print('Loading and clipping TIFs...')

    aef17, aef_bounds = load_clipped(f'{BASE}/GEE_annual_predictions/AEF_RF_predicted_agbd_kibale2017.tif')
    aef20, _          = load_clipped(f'{BASE}/GEE_annual_predictions/AEF_RF_predicted_agbd_kibale2020.tif')
    ls17,  ls_bounds  = load_clipped(f'{BASE}/ls_comp_annual_pred/agbd_2017_xgboost_sqrt_nbr_lhs_33_revkibale_2sig_cov_beam_no5_5corners_0507_mean.tif')
    ls20,  _          = load_clipped(f'{BASE}/ls_comp_annual_pred/agbd_2020_xgboost_sqrt_nbr_lhs_33_revkibale_2sig_cov_beam_no5_5corners_0507_mean.tif')

    # Compute change (2020 − 2017); mask pixels invalid in either year
    aef_chng = aef20 - aef17
    aef_chng[~np.isfinite(aef17) | ~np.isfinite(aef20)] = np.nan

    ls_chng  = ls20  - ls17
    ls_chng[~np.isfinite(ls17)  | ~np.isfinite(ls20)]  = np.nan

    # Report
    for name, arr in [('AEF 2017', aef17), ('LS 2017', ls17),
                      ('AEF chng', aef_chng), ('LS chng', ls_chng)]:
        v = arr[np.isfinite(arr)]
        print(f'  {name}: shape={arr.shape}, range=[{v.min():.1f}, {v.max():.1f}]')

    print('\nExporting PNGs...')
    to_png(aef17,    f'{OUT}/aef_2017.png',   VMIN_AGBD, VMAX_AGBD, 'YlGn')
    to_png(ls17,     f'{OUT}/ls_2017.png',    VMIN_AGBD, VMAX_AGBD, 'YlGn')
    to_png(aef20,    f'{OUT}/aef_2020.png',   VMIN_AGBD, VMAX_AGBD, 'YlGn')
    to_png(ls20,     f'{OUT}/ls_2020.png',    VMIN_AGBD, VMAX_AGBD, 'YlGn')
    to_png(aef_chng, f'{OUT}/aef_change.png', VMIN_CHNG, VMAX_CHNG, 'RdYlGn')
    to_png(ls_chng,  f'{OUT}/ls_change.png',  VMIN_CHNG, VMAX_CHNG, 'RdYlGn')

    print('\nExporting raw float32 arrays for click-value lookup...')
    layers = {
        'aef_2017': aef17, 'ls_2017': ls17,
        'aef_2020': aef20, 'ls_2020': ls20,
        'aef_change': aef_chng, 'ls_change': ls_chng,
    }
    for name, arr in layers.items():
        arr.astype(np.float32).tofile(f'{OUT}/{name}.dat')
        print(f'  Saved {name}.dat  shape={arr.shape}')

    # Write bounds.js — both sources clipped to same AOI so bounds match.
    W = min(aef_bounds[0], ls_bounds[0])
    S = min(aef_bounds[1], ls_bounds[1])
    E = max(aef_bounds[2], ls_bounds[2])
    N = max(aef_bounds[3], ls_bounds[3])

    # Include per-source shape so JS knows how to index the .dat files
    bounds_js = (
        f'var AOI_BOUNDS = [[{S:.6f}, {W:.6f}], [{N:.6f}, {E:.6f}]];\n'
        f'var LAYER_META = {{\n'
        f'  aef: {{rows: {aef17.shape[0]}, cols: {aef17.shape[1]}}},\n'
        f'  ls:  {{rows: {ls17.shape[0]},  cols: {ls17.shape[1]}}}\n'
        f'}};\n'
    )
    with open(f'{OUT}/bounds.js', 'w') as fh:
        fh.write(bounds_js)
    print(f'\nbounds.js written.')
    print('Done.')
