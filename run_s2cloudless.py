import os
import argparse
import numpy as np
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from s2cloudless import S2PixelCloudDetector
from skimage.transform import resize

# ----------------------- Argument Parsing -----------------------
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Path to IMG_DATA folder inside .SAFE tile")
parser.add_argument("--output", required=True, help="Path to save the cloud masks")
parser.add_argument("--mode", required=True, choices=["validation", "CVAT-VSM"], help="Output format mode")
parser.add_argument("--bounds", required=False, help="Bounds in format minx,miny,maxx,maxy")
parser.add_argument("--name", required=True, help="Filename prefix for output")
args = parser.parse_args()

input_folder = args.input
save_to = args.output
identifier = ""

# Parse bounds if provided
bounds = tuple(map(float, args.bounds.strip("()").split(','))) if args.bounds else None

# Adjust output folder
if args.mode == "CVAT-VSM":
    save_to = input_folder.replace("IMG_DATA", "S2CLOUDLESS_DATA")
os.makedirs(save_to, exist_ok=True)

print(f"📁 Saving output to: {save_to}")

# Extract identifier from B01 file
for fname in os.listdir(input_folder):
    if "B01.jp2" in fname:
        identifier = fname.split("B01")[0]
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

# ----------------------- Load and Resample Bands -----------------------
print("📥 Reading and aligning bands...")
band_names = ['B01', 'B02', 'B04', 'B05', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
band_paths = [os.path.join(input_folder, f"{identifier}{b}.jp2") for b in band_names]

# Read B02 to define target shape
B02 = read_band(band_paths[1], bounds=bounds)
target_shape = B02.shape[1:]

# Read all bands and normalize
bands_resampled = []
for i, path in enumerate(band_paths):
    band = read_band(path, target_shape=target_shape, bounds=bounds)
    print(f"{band_names[i]} shape: {band.shape}")
    bands_resampled.append(band[0] / 10000.0)

# Stack into (1, H, W, 10)
bands = np.array([np.dstack(bands_resampled)])
print(f"✅ Input shape for cloud detection: {bands.shape}")

for i, b in enumerate(bands_resampled):
    print(f"{band_names[i]} min: {b.min():.4f}, max: {b.max():.4f}")

# ----------------------- Cloud Detection -----------------------
cloud_detector = S2PixelCloudDetector(
    threshold=0.4,
    average_over=4,
    dilation_size=2,
    all_bands=False  # Because you're using 10 standard bands only
)

cloud_probs = cloud_detector.get_cloud_probability_maps(bands)
cloud_mask = cloud_detector.get_cloud_masks(bands).astype(np.uint8)

print(f"🌥️ Cloud prob stats: min={cloud_probs.min():.4f}, max={cloud_probs.max():.4f}, mean={cloud_probs.mean():.4f}")
print(f"🧩 Unique values in cloud mask: {np.unique(cloud_mask)}")

# ----------------------- Save GeoTIFFs -----------------------
ref_band_path = band_paths[1]  # Use B02 for georeferencing
with rasterio.open(ref_band_path) as ref:
    profile = ref.profile
    transform = ref.transform
    crs = ref.crs

profile.update(
    dtype=rasterio.uint8,
    count=1,
    compress='lzw',
    height=target_shape[0],
    width=target_shape[1],
    transform=rasterio.transform.from_origin(ref.bounds.left, ref.bounds.top, 10.0, 10.0),
    crs=crs
)

cloud_mask_path = os.path.join(save_to, f"{args.name}_cloud_mask.tif")
cloud_prob_path = os.path.join(save_to, f"{args.name}_cloud_prob.tif")

with rasterio.open(cloud_mask_path, 'w', **profile) as dst:
    dst.write(cloud_mask[0], 1)

with rasterio.open(cloud_prob_path, 'w', **profile) as dst:
    dst.write((cloud_probs[0] * 255).astype(np.uint8), 1)

print(f"✅ Saved cloud mask: {cloud_mask_path}")
print(f"✅ Saved cloud probability map: {cloud_prob_path}")
