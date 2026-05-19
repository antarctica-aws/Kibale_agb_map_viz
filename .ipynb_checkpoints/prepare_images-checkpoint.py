"""
Converts the two 2017 AGBD GeoTIFFs to RGBA PNGs for web display.
Outputs:
  aef_2017.png   — GEE/AEF RF predictions
  ls_2017.png    — Landsat XGBoost predictions
Run with:
  /home/users/mliang77/.micromamba/envs/gdal-env/bin/python prepare_images.py
"""
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from PIL import Image
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pyproj

pyproj.datadir.set_data_dir('/home/users/mliang77/.micromamba/envs/gdal-env/share/proj')

BASE = '/oak/stanford/groups/kailou/mliang77/AEF_GEDI_test'
OUT  = '/oak/stanford/groups/kailou/mliang77/AEF_GEDI_test/agb_map_viz'

VMIN, VMAX = 0, 400      # Mg/ha colour scale
CMAP_NAME  = 'YlGn'


def tif_to_rgba_png(input_path, output_path):
    """Reproject to EPSG:4326 (if needed) and export as a coloured RGBA PNG."""
    with rasterio.open(input_path) as src:
        if src.crs.to_epsg() == 4326:
            data = src.read(1).astype(np.float32)
            nodata = src.nodata
        else:
            dst_crs  = 'EPSG:4326'
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds
            )
            data = np.zeros((height, width), dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
                dst_nodata=np.nan,
            )
            nodata = src.nodata

    # Build transparency mask
    mask = ~np.isfinite(data)
    if nodata is not None:
        mask |= np.isclose(data, nodata)
    # Also mask near-zero edge artefacts from reprojection
    mask |= (data <= 0)

    norm = mcolors.Normalize(vmin=VMIN, vmax=VMAX, clip=True)
    cmap = cm.get_cmap(CMAP_NAME)
    rgba = (cmap(norm(data)) * 255).astype(np.uint8)
    rgba[mask] = 0  # fully transparent

    img = Image.fromarray(rgba, mode='RGBA')

    # Downsample to max 2048px on the longest side for web performance
    MAX_PX = 2048
    if max(img.width, img.height) > MAX_PX:
        scale = MAX_PX / max(img.width, img.height)
        new_size = (int(img.width * scale), int(img.height * scale))
        img = img.resize(new_size, Image.LANCZOS)

    img.save(output_path, optimize=True)
    print(f'Saved {output_path}  ({img.width}x{img.height})')


if __name__ == '__main__':
    tif_to_rgba_png(
        f'{BASE}/GEE_annual_predictions/AEF_RF_predicted_agbd_kibale2017.tif',
        f'{OUT}/aef_2017.png',
    )
    tif_to_rgba_png(
        f'{BASE}/ls_comp_annual_pred/agbd_2017_xgboost_sqrt_nbr_lhs_33_revkibale_2sig_cov_beam_no5_5corners_0507_mean.tif',
        f'{OUT}/ls_2017.png',
    )
    print('Done.')
