# run_s2cloudless.py — load .jp2 bands, crop using bounds, generate cloud mask

import os
import argparse
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from s2cloudless import S2PixelCloudDetector
from skimage.transform import resize

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Path to IMG_DATA folder inside .SAFE tile")
parser.add_argument("--output", required=True, help="Path to save the cloud masks")
parser.add_argument("--mode", required=True, choices=["validation", "CVAT-VSM"])
parser.add_argument("--bounds", required=True, help="Bounds: minx,miny,maxx,maxy")
parser.add_argument("--name", required=True, help="Filename prefix for output")
args = parser.parse_args()

input_folder = args.input
save_to = args.output
bounds = tuple(map(float, args.bounds.split(',')))
identifier = ""

for fname in os.listdir(input_folder):
    if "B01.jp2" in fname:
        identifier = fname.split("B01")[0]
        break

os.makedirs(save_to, exist_ok=True)

band_names = ['B01', 'B02', 'B04', 'B05', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
band_paths = [os.path.join(input_folder, f"{identifier}{b}.jp2") for b in band_names]

# Read reference shape from B02
with rasterio.open(band_paths[1]) as ref:
    window = from_bounds(*bounds, transform=ref.transform)
    ref_data = ref.read(1, window=window)
    target_shape = ref_data.shape

bands_resampled = []
for path in band_paths:
    with rasterio.open(path) as ds:
        window = from_bounds(*bounds, transform=ds.transform)
        data = ds.read(1, window=window, resampling=Resampling.bilinear)
        data_resized = resize(data, target_shape, preserve_range=True, anti_aliasing=False)
        bands_resampled.append(data_resized / 10000.0)

stacked = np.dstack(bands_resampled)
bands = np.array([stacked])

cloud_detector = S2PixelCloudDetector(threshold=0.4, average_over=4, dilation_size=2, all_bands=False)
cloud_probs = cloud_detector.get_cloud_probability_maps(bands)
cloud_mask = cloud_detector.get_cloud_masks(bands).astype(np.uint8)

print(f"Cloud stats: prob min={cloud_probs.min()}, max={cloud_probs.max()}, mean={cloud_probs.mean()}")

with rasterio.open(band_paths[1]) as ref:
    transform = ref.window_transform(window)
    crs = ref.crs

profile = ref.profile.copy()
profile.update({
    'driver': 'GTiff',
    'height': target_shape[0],
    'width': target_shape[1],
    'transform': transform,
    'crs': crs,
    'count': 1,
    'dtype': 'uint8',
    'compress': 'lzw'
})

cloud_mask_path = os.path.join(save_to, f"{args.name}_cloud_mask.tif")
cloud_prob_path = os.path.join(save_to, f"{args.name}_cloud_prob.tif")

with rasterio.open(cloud_mask_path, 'w', **profile) as dst:
    dst.write(cloud_mask[0], 1)

with rasterio.open(cloud_prob_path, 'w', **profile) as dst:
    dst.write((cloud_probs[0] * 255).astype(np.uint8), 1)

print(f"✅ Saved cloud mask: {cloud_mask_path}")
print(f"✅ Saved cloud probability map: {cloud_prob_path}")
