import os
from osgeo import gdal
import os

from tqdm import tqdm
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

import geopandas as gpd
from shapely.geometry import Polygon
import rasterio
from rasterio.features import shapes
from osgeo import osr
from labelWorkflow import shp_to_tiff, stretch_2_percent, assign_categories_to_masks, masks_to_shapefile_with_class_no_geo, show_anns, masks_to_shapefile_with_class_no_geo_separate


# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

np.random.seed(3)


# 定义输入文件夹
folder_s2 = 'data/s2'
folder_label = 'data/label'
output_folder = 'data/out'

# sam2_checkpoint = "checkpoints/sam2_hiera_large.pt"
# model_cfg = "sam2_hiera_l.yaml"
sam2_checkpoint = "checkpoints/sam2_hiera_base_plus.pt"
model_cfg = "sam2_hiera_b+.yaml"
# SAM-2 
sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
mask_generator = SAM2AutomaticMaskGenerator(  # https://blog.csdn.net/qq_30150579/article/details/134742998
    model=sam2,
    points_per_side= 64, #64
    points_per_batch=128,
    pred_iou_thresh=0.7,
    stability_score_thresh=0.92,
    stability_score_offset=0.7,
    crop_n_layers=1,
    box_nms_thresh=0.7,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=5.0,
    use_m2m=True,
)

images_s2 = [f for f in os.listdir(folder_s2) if f.endswith('.tif')]
labels = [f for f in os.listdir(folder_label) if f.endswith('.tif')]

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for image_name in tqdm(images_s2):
    if image_name in labels:
        image_path = os.path.join(folder_s2, image_name)
        label_path = os.path.join(folder_label, image_name)

        S2_RGB_path = os.path.join(output_folder, image_name.replace('.tif', '_s2RGB.jpg'))
        S2_mask_path = os.path.join(output_folder, image_name.replace('.tif', '_s2Mask.jpg'))
        Mask_shp_path = os.path.join(output_folder, image_name.replace('.tif', '_mask.shp'))
        output_label_tif = os.path.join(output_folder, image_name.replace('.tif', '_RBLabel.tif'))
        output_label_png = os.path.join(output_folder, image_name.replace('.tif', '_RBLabel.png'))
        output_srclabel_png = os.path.join(output_folder, image_name.replace('.tif', '_RBSrcLabel.png'))
        
        dataset = gdal.Open(image_path, gdal.GA_ReadOnly)
        if dataset is None:
            print(f"无法打开图像文件: {image_path}")
            continue
        
        label_dataset = rasterio.open(label_path)
        if label_dataset is None:
            print(f"无法打开标签文件: {label_path}")
            continue
        label_array = label_dataset.read(1)
        label_array_uint8 = label_array.astype(np.uint8)
        label_image = Image.fromarray(label_array_uint8*255)
        label_image.save(output_srclabel_png)

        band_r = dataset.GetRasterBand(1).ReadAsArray()
        band_g = dataset.GetRasterBand(2).ReadAsArray()
        band_b = dataset.GetRasterBand(3).ReadAsArray()

        band_r_stretched = stretch_2_percent(band_r, 2)
        band_g_stretched = stretch_2_percent(band_g, 2)
        band_b_stretched = stretch_2_percent(band_b, 2)

        image = np.dstack((band_r_stretched, band_g_stretched, band_b_stretched))
        rgb_image = Image.fromarray(image, 'RGB')
        rgb_image.save(S2_RGB_path)


        masks = mask_generator.generate(image)

        plt.figure(figsize=(20, 20))
        plt.imshow(image)
        show_anns(masks)
        plt.axis('off')
        plt.savefig(S2_mask_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        assign_categories_to_masks(label_path, masks)
        masks_to_shapefile_with_class_no_geo_separate(masks, Mask_shp_path)

        shp_to_tiff(Mask_shp_path, label_path, output_label_tif, output_label_png)
        
        print("##############################################################################################")
        print(f"Num of Mask: {len(masks)}")
        # print(masks[0].keys())
        print(f"Pro done: {Mask_shp_path}")

        dataset = None
        label_dataset = None


