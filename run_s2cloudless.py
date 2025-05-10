import os
import argparse
import numpy as np
import rasterio
from s2cloudless import S2PixelCloudDetector

# ---------------------- Parse Arguments ---------------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Path to IMG_DATA folder inside .SAFE tile")
parser.add_argument("--output", required=True, help="Path to save cloud masks")
parser.add_argument("--mode", required=True, choices=["validation", "CVAT-VSM"])
parser.add_argument("--name", required=True, help="Prefix for output filenames")
args = parser.parse_args()

input_folder = args.input
save_to = args.output
identifier = ""

# ---------------------- Identify Band Prefix ---------------------- #
for fname in os.listdir(input_folder):
    if "B01.jp2" in fname:
        identifier = fname.split("B01")[0]
        break

os.makedirs(save_to, exist_ok=True)

band_names = ['B01', 'B02', 'B04', 'B05', 'B08',
              'B8A', 'B09', 'B10', 'B11', 'B12']
band_paths = [os.path.join(input_folder, f"{identifier}{b}.jp2") for b in band_names]

# ---------------------- Load Bands ---------------------- #
bands_data = []
scale_detected = None

for path in band_paths:
    with rasterio.open(path) as ds:
        data = ds.read(1)
        print(f"{os.path.basename(path)} stats: min={data.min()}, max={data.max()}, mean={data.mean()}")
        if scale_detected is None:
            scale_detected = "raw" if data.max() > 100.0 else "normalized"
        if scale_detected == "raw":
            bands_data.append(data / 10000.0)
        else:
            bands_data.append(data)

print(f"📊 Detected reflectance scale: {scale_detected}")

# ---------------------- Stack and Run Cloud Detection ---------------------- #
stacked = np.dstack(bands_data)
bands = np.array([stacked])  # shape: (1, H, W, 10)

cloud_detector = S2PixelCloudDetector(
    threshold=0.4,
    average_over=4,
    dilation_size=2,
    all_bands=False
)

cloud_probs = cloud_detector.get_cloud_probability_maps(bands)
cloud_mask = cloud_detector.get_cloud_masks(bands).astype(np.uint8)

print(f"🌥️ Cloud prob stats: min={cloud_probs.min()}, max={cloud_probs.max()}, mean={cloud_probs.mean()}")
print(f"🧩 Unique values in cloud mask: {np.unique(cloud_mask)}")

# ---------------------- Save Output ---------------------- #
with rasterio.open(band_paths[1]) as ref:
    profile = ref.profile.copy()
    profile.update({
        'driver': 'GTiff',
        'height': bands.shape[1],
        'width': bands.shape[2],
        'count': 1,
        'dtype': 'uint8',
        'compress': 'lzw'
    })

    transform = ref.transform
    crs = ref.crs

profile.update({'transform': transform, 'crs': crs})

cloud_mask_path = os.path.join(save_to, f"{args.name}_cloud_mask.tif")
cloud_prob_path = os.path.join(save_to, f"{args.name}_cloud_prob.tif")

with rasterio.open(cloud_mask_path, 'w', **profile) as dst:
    dst.write(cloud_mask[0], 1)

with rasterio.open(cloud_prob_path, 'w', **profile) as dst:
    dst.write((cloud_probs[0] * 255).astype(np.uint8), 1)

print(f"✅ Saved full-tile cloud mask: {cloud_mask_path}")
print(f"✅ Saved full-tile cloud probability map: {cloud_prob_path}")
