from s2cloudless import S2PixelCloudDetector
import numpy as np
import rasterio
import os
import argparse
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--mode", required=True, choices=["validation", "CVAT-VSM"])
parser.add_argument("--bounds", required=False)
parser.add_argument("--name", required=True)
a = parser.parse_args()

input_folder = a.input
save_to = a.output
bounds = tuple(map(float, a.bounds.strip("()").split(','))) if a.bounds else None
if a.mode == "CVAT-VSM":
    save_to = input_folder.replace("IMG_DATA", "S2CLOUDLESS_DATA")

os.makedirs(save_to, exist_ok=True)

identifier = ""
for filename in os.listdir(input_folder):
    if "B01.jp2" in filename:
        identifier = filename.split("B01")[0]

def read_band(path, scale=1, bounds=None):
    with rasterio.open(path) as dataset:
        if bounds:
            window = from_bounds(*bounds, transform=dataset.transform)
            data = dataset.read(
                out_shape=(dataset.count, round(window.height * scale), round(window.width * scale)),
                window=window,
                resampling=Resampling.bilinear
            )
        else:
            data = dataset.read(
                out_shape=(dataset.count, int(dataset.height * scale), int(dataset.width * scale)),
                resampling=Resampling.bilinear
            )
        return data

# Read bands
def read_band_safe(band_suffix, scale, bounds):
    path = os.path.join(input_folder, identifier + band_suffix + ".jp2")
    data = read_band(path, scale, bounds)
    print(f"{band_suffix}: shape={data.shape}, min={data.min()}, max={data.max()}, all_zero={np.all(data == 0)}")
    return data

B01 = read_band_safe("B01", 6, bounds)
B02 = read_band_safe("B02", 1, bounds)
B04 = read_band_safe("B04", 1, bounds)
B05 = read_band_safe("B05", 2, bounds)
B08 = read_band_safe("B08", 1, bounds)
B8A = read_band_safe("B8A", 2, bounds)
B09 = read_band_safe("B09", 6, bounds)
B10 = read_band_safe("B10", 6, bounds)
B11 = read_band_safe("B11", 2, bounds)
B12 = read_band_safe("B12", 2, bounds)

# Stack and normalize
bands = np.array([np.dstack((
    B01[0]/10000.0, B02[0]/10000.0, B04[0]/10000.0,
    B05[0]/10000.0, B08[0]/10000.0, B8A[0]/10000.0,
    B09[0]/10000.0, B10[0]/10000.0, B11[0]/10000.0, B12[0]/10000.0
))])

# Cloud detection
cloud_detector = S2PixelCloudDetector(threshold=0.2, average_over=4, dilation_size=2, all_bands=False)
cloud_probs = cloud_detector.get_cloud_probability_maps(bands)
mask = cloud_detector.get_cloud_masks(bands).astype(rasterio.uint8)

# Reference metadata
ref_band_path = os.path.join(input_folder, identifier + "B02.jp2")
with rasterio.open(ref_band_path) as ref:
    profile = ref.profile
profile.update(dtype=rasterio.uint8, count=1, compress='lzw')

cloud_mask_path = os.path.join(save_to, f"{a.name}_cloud_mask.tif")
cloud_prob_path = os.path.join(save_to, f"{a.name}_cloud_prob.tif")

with rasterio.open(cloud_mask_path, 'w', **profile) as dst:
    dst.write(mask[0], 1)
with rasterio.open(cloud_prob_path, 'w', **profile) as dst:
    dst.write((cloud_probs[0] * 255).astype(np.uint8), 1)

print(f"✅ Saved mask to: {cloud_mask_path}")
print(f"✅ Saved prob to: {cloud_prob_path}")
