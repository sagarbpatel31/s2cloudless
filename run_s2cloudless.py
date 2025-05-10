import os
import sys
import argparse
import numpy as np
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import math
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from s2cloudless import S2PixelCloudDetector
from skimage.transform import resize
# ----------------------- Argument Parsing -----------------------
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Path to IMG_DATA folder inside .SAFE tile")
parser.add_argument("--output", required=True, help="Path to save the cloud masks")
parser.add_argument("--mode", required=True, choices=["validation", "CVAT-VSM"], help="Output format mode")
parser.add_argument("--bounds", required=False, help="Bounds in format (minx,miny,maxx,maxy)")
parser.add_argument("--name", required=True, help="Filename prefix for output")
a = parser.parse_args()
input_folder = a.input
save_to = a.output
identifier = ""
# Parse optional bounds
bounds = tuple(map(float, a.bounds.strip("()").split(','))) if a.bounds else None
# If CVAT-VSM, overwrite save path to be inside tile folder
if a.mode == "CVAT-VSM":
    save_to = input_folder.replace("IMG_DATA", "S2CLOUDLESS_DATA")
# Ensure output folder exists
os.makedirs(save_to, exist_ok=True)
print(f"Saving output to: {save_to}")
# Get common filename identifier (e.g., S2A_MSIL1C_...)
for filename in os.listdir(input_folder):
    if "B01.jp2" in filename:
        identifier = filename.split("B01")[0]
        break
# ----------------------- Helper Functions -----------------------
def read_band(path, target_shape=None, bounds=None):
    with rasterio.open(path) as dataset:
        if bounds:
            window = from_bounds(*bounds, transform=dataset.transform)
            data = dataset.read(window=window)
        else:
            data = dataset.read()
        if target_shape:
            band_resized = resize(data[0], target_shape, preserve_range=True, order=1, anti_aliasing=False)
            return band_resized[np.newaxis, ...].astype(np.float32)
        else:
            return data
def plot_cloud_mask(mask):
    if a.mode == "validation":
        new_mask = np.where(mask == 1, 255, 0).astype(np.uint8)
        im_result = Image.fromarray(new_mask)
        im_result = im_result.resize((10980, 10980), Image.NEAREST)
        im_result.save(os.path.join(save_to, f"{a.name}_cloud_mask_preview.png"))
    else:
        im_result = Image.fromarray(mask.astype(np.uint8))
        im_result.save(os.path.join(save_to, f"{a.name}_cloud_mask_preview.png"))
def plot_probability_map(prob_map):
    if a.mode == "validation":
        plt.figure(figsize=(10, 10))
        plt.imshow(prob_map, cmap='inferno')
        plt.savefig(os.path.join(save_to, f"{a.name}_cloud_prob_colormap.png"))
    else:
        im_result = Image.fromarray((prob_map * 255).astype(np.uint8))
        im_result.save(os.path.join(save_to, f"{a.name}_cloud_prob.png"))
# ----------------------- Load and Resample Bands -----------------------
print("Reading and aligning bands...")
band_names = ['B01', 'B02', 'B04', 'B05', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
band_paths = [os.path.join(input_folder, f"{identifier}{b}.jp2") for b in band_names]
# First, read B02 to define target shape
B02 = read_band(band_paths[1], bounds=bounds)
target_shape = B02.shape[1:]
# Read all bands to the same shape
bands_resampled = []
for idx, path in enumerate(band_paths):
    band = read_band(path, target_shape=target_shape, bounds=bounds)
    print(f"{band_names[idx]} shape: {band.shape}")
    bands_resampled.append(band[0] / 10000.0)  # normalize reflectance
# Stack into (1, H, W, 10)
bands = np.array([np.dstack(bands_resampled)])

for i, b in enumerate(bands_resampled):
    print(f"B{i+1} min: {b.min()}, max: {b.max()}, shape: {b.shape}")
# ----------------------- Cloud Detection -----------------------
cloud_detector = S2PixelCloudDetector(threshold=0.4, average_over=22, dilation_size=11)
cloud_probs = cloud_detector.get_cloud_probability_maps(bands)
cloud_mask = cloud_detector.get_cloud_masks(bands).astype(np.uint8)
# Plot preview images
# plot_cloud_mask(cloud_mask[0])
# plot_probability_map(cloud_probs[0])
# ----------------------- Save GeoTIFFs -----------------------
ref_band_path = band_paths[1]  # use B02 as reference
with rasterio.open(ref_band_path) as ref:
    profile = ref.profile
    transform = ref.transform
    crs = ref.crs
# Update for single-band, 8-bit output
profile.update(
    dtype=rasterio.uint8,
    count=1,
    compress='lzw',
    height=target_shape[0],
    width=target_shape[1],
    transform=rasterio.transform.from_origin(ref.bounds.left, ref.bounds.top, 10.0, 10.0),
    crs=crs
)
cloud_mask_path = os.path.join(save_to, f"{a.name}_cloud_mask.tif")
cloud_prob_path = os.path.join(save_to, f"{a.name}_cloud_prob.tif")
# Write cloud mask
with rasterio.open(cloud_mask_path, 'w', **profile) as dst:
    dst.write(cloud_mask[0], 1)
# Write cloud probability (scaled 0–255)
with rasterio.open(cloud_prob_path, 'w', **profile) as dst:
    dst.write((cloud_probs[0] * 255).astype(np.uint8), 1)
print(f":white_check_mark: Saved cloud mask: {cloud_mask_path}")
print(f":white_check_mark: Saved cloud probability map: {cloud_prob_path}")
