from s2cloudless import S2PixelCloudDetector
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import sys
import os
import argparse
from rasterio.windows import from_bounds
from rasterio.enums import Resampling

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="path to the folder where the jp2 files of the L1C product are located (IMG_DATA)")
parser.add_argument("--output", required=True, help="path to the folder where thecloud masks are to be save are to be saved ")
parser.add_argument("--mode", required=True,  choices=["validation", "CVAT-VSM"], help="validation: output images rescaled to 10980x10980px, mask output with pixel values 0 or 255, probability output with colormap. CVAT-VSM: output image dimensions 1830x1830px, mask output with pixel values 0 or 1, probabilty output as greyscaled.")
parser.add_argument("--bounds", required=False, help="Bounds in format (minx,miny,maxx,maxy)")
parser.add_argument("--name", required=True, help="Filename prefix for output")

a = parser.parse_args()

input_folder=a.input
save_to=a.output
bounds = tuple(map(float, a.bounds.strip("()").split(','))) if a.bounds else None

if(a.mode=="CVAT-VSM"):
    save_to=input_folder.replace("IMG_DATA","S2CLOUDLESS_DATA")
    
if(os.path.isdir(save_to)==False):
    os.makedirs(save_to)
    print("Created folder "+save_to)


#Check the name of the product in the folder

identifier=""
for filename in os.listdir(input_folder):
    if("B01.jp2" in filename):
        identifier=filename.split("B01")[0]
        
def plot_cloud_mask(mask, figsize=(15, 15), fig=None):
    """
    Utility function for plotting a binary cloud mask.
    """ 
    if(a.mode=="validation"): 
        new_mask = [[255 if b==1 else b for b in i] for i in mask]
        im_result = Image.fromarray(np.uint8(new_mask))
        im_result=im_result.resize((10980,10980),Image.NEAREST)
        im_result.save(identifier+"_s2cloudless_prediction.png")
    else:
        im_result = Image.fromarray(np.uint8(mask))
        im_result.save(os.path.join(save_to,"s2cloudless_prediction.png"))

def plot_probability_map(prob_map, figsize=(15, 15)):
    if(a.mode=="validation"):
        plt.figure(figsize=figsize)
        plt.imshow(prob_map,cmap=plt.cm.inferno)
        plt.savefig(identifier+"_s2cloudless_probability.png")
    else:
        im_result = Image.fromarray(np.uint8(prob_map * 255))
        im_result.save(os.path.join(save_to,"s2cloudless_probability.png"))       

#Resampling to achieve 10 m resolution:
def read_band(path, scale=1, bounds=None):
    with rasterio.open(path) as dataset:
        if bounds:
            window = from_bounds(*bounds, transform=dataset.transform)
            data = dataset.read(
                out_shape=(
                    dataset.count,
                    round(window.height * scale),
                    round(window.width * scale)
                ),
                window=window,
                resampling=Resampling.bilinear
            )
        else:
            data = dataset.read(
                out_shape=(
                    dataset.count,
                    int(dataset.height * scale),
                    int(dataset.width * scale)
                ),
                resampling=Resampling.bilinear
            )
        return data

B01_path = os.path.join(input_folder,identifier+"B01.jp2")
B01 = read_band(B01_path, 6, bounds)
print(B01.shape, flush=True)

B02_path = os.path.join(input_folder,identifier+"B02.jp2")
B02 = read_band(B02_path, 1, bounds)
print(B01.shape, flush=True)

B04_path = os.path.join(input_folder,identifier+"B04.jp2")
B04 = read_band(B04_path, 1, bounds)
print(B04.shape, flush=True)

B05_path = os.path.join(input_folder,identifier+"B05.jp2")
B05 = read_band(B05_path, 2, bounds)
print(B05.shape, flush=True)

B08_path = os.path.join(input_folder,identifier+"B08.jp2")
B08 = read_band(B08_path, 1, bounds)
print(B08.shape, flush=True)

B8A_path = os.path.join(input_folder,identifier+"B8A.jp2")
B8A = read_band(B8A_path, 2, bounds)
print(B8A.shape, flush=True)

B09_path = os.path.join(input_folder,identifier+"B09.jp2")
B09 = read_band(B09_path, 6, bounds)
print(B09.shape, flush=True)

B10_path = os.path.join(input_folder,identifier+"B10.jp2")
B10 = read_band(B10_path, 6, bounds)
print(B05.shape, flush=True)

B11_path = os.path.join(input_folder,identifier+"B11.jp2")
B11 = read_band(B11_path, 2, bounds)
print(B11.shape, flush=True)

B12_path = os.path.join(input_folder,identifier+"B12.jp2")
B12 = read_band(B12_path, 2, bounds)
print(B12.shape, flush=True)

bands = np.array([np.dstack((B01[0]/10000.0,B02[0]/10000.0,B04[0]/10000.0,B05[0]/10000.0,B08[0]/10000.0,B8A[0]/10000.0,B09[0]/10000.0,B10[0]/10000.0,B11[0]/10000.0,B12[0]/10000.0))])

#Recommended parameters for 60 m resolution: average_over = 4, dilation_size=2
#Recommended parameters for 10 m resolution: average_over=22, dilation_size=11
#The actual best result is achievable by trying different values for different products.

cloud_detector = S2PixelCloudDetector(threshold=0.2, average_over=4, dilation_size=2,all_bands=False)  
cloud_probs = cloud_detector.get_cloud_probability_maps(bands)
mask = cloud_detector.get_cloud_masks(bands).astype(rasterio.uint8)

# Use B02 as reference for metadata
ref_band_path = os.path.join(input_folder, identifier + "B02.jp2")
with rasterio.open(ref_band_path) as ref:
    profile = ref.profile
    transform = ref.transform
    crs = ref.crs

# Update profile for 8-bit output
profile.update(
    dtype=rasterio.uint8,
    count=1,
    compress='lzw'
)

cloud_mask_path = os.path.join(save_to, f"{a.name}_cloud_mask.tif")
cloud_prob_path = os.path.join(save_to, f"{a.name}_cloud_prob.tif")

# Save cloud mask GeoTIFF
with rasterio.open(cloud_mask_path, 'w', **profile) as dst:
    dst.write(mask[0], 1)

# Save cloud probability GeoTIFF (scaled 0–255)
with rasterio.open(cloud_prob_path, 'w', **profile) as dst:
    dst.write((cloud_probs[0] * 255).astype(np.uint8), 1)

print(f"Cloud mask and probability maps saved to: {save_to}")
