import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

import geopandas as gpd
from shapely.geometry import Polygon
import rasterio
from rasterio.features import shapes
from osgeo import gdal, ogr, osr

# # select the device for computation
# if torch.cuda.is_available():
#     device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
# else:
#     device = torch.device("cpu")
# print(f"using device: {device}")

# if device.type == "cuda":
#     # use bfloat16 for the entire notebook
#     torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
#     # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
#     if torch.cuda.get_device_properties(0).major >= 8:
#         torch.backends.cuda.matmul.allow_tf32 = True
#         torch.backends.cudnn.allow_tf32 = True
# elif device.type == "mps":
#     print(
#         "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
#         "give numerically different outputs and sometimes degraded performance on MPS. "
#         "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
#     )

# np.random.seed(3)

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(img)

def stretch_2_percent(band, percent):
    lower_percentile = np.percentile(band, percent)
    upper_percentile = np.percentile(band, 100-percent)
    
    stretched_band = np.clip(band, lower_percentile, upper_percentile)
    stretched_band = (stretched_band - lower_percentile) / (upper_percentile - lower_percentile) * 255
    
    return stretched_band.astype(np.uint8)


def masks_to_shapefile(masks, output_path, transform, crs):
    polygons = []
    
    for mask in masks:
        mask_array = mask['segmentation'].astype(np.uint8)
        results = ({'properties': {'raster_val': v}, 'geometry': s}
                    for i, (s, v) in enumerate(
                    shapes(mask_array, mask=mask_array, transform=transform)))
        
        for result in results:
            geom = result['geometry']
            if geom['type'] == 'Polygon':
                polygons.append(Polygon(geom['coordinates'][0]))

    gdf = gpd.GeoDataFrame({'geometry': polygons}, crs=crs)
    gdf.to_file(output_path)

def masks_to_shapefile_with_class(masks, output_path, transform, crs):
    polygons = []
    categories = []

    for mask in masks:
        mask_array = mask['segmentation'].astype(np.uint8)
        mask_category = mask.get('category', 0)

        results = ({'properties': {'raster_val': v}, 'geometry': s}
                   for i, (s, v) in enumerate(
                   shapes(mask_array, mask=mask_array, transform=transform)))
        
        for result in results:
            geom = result['geometry']
            if geom['type'] == 'Polygon':
                polygons.append(Polygon(geom['coordinates'][0]))
                categories.append(mask_category)

    gdf = gpd.GeoDataFrame({'geometry': polygons, 'category': categories}, crs=crs)
    gdf.to_file(output_path)

def masks_to_shapefile_with_class_no_geo(masks, output_path):
    polygons = []
    categories = []

    for mask in masks:
        mask_array = mask['segmentation'].astype(np.uint8)
        mask_category = mask.get('category', 0)

        results = ({'properties': {'raster_val': v}, 'geometry': s}
                   for i, (s, v) in enumerate(shapes(mask_array, mask=mask_array)))
        
        for result in results:
            geom = result['geometry']
            if geom['type'] == 'Polygon':
                polygons.append(Polygon(geom['coordinates'][0]))
                categories.append(mask_category)

    gdf = gpd.GeoDataFrame({'geometry': polygons, 'category': categories})
    gdf.to_file(output_path, driver='ESRI Shapefile')

def masks_to_shapefile_with_class_separate(masks, output_path, transform, crs):
    polygons = []
    categories = []

    kernel = np.ones((3,3), np.uint8)

    for mask in masks:
        mask_array = mask['segmentation'].astype(np.uint8)

        eroded_mask = cv2.erode(mask_array, kernel, iterations=2)

        mask_category = mask.get('category', 0)

        results = ({'properties': {'raster_val': v}, 'geometry': s}
                   for i, (s, v) in enumerate(shapes(eroded_mask, mask=eroded_mask, transform=transform)))
        
        for result in results:
            geom = result['geometry']
            if geom['type'] == 'Polygon':
                polygons.append(Polygon(geom['coordinates'][0]))
                categories.append(mask_category)

    gdf = gpd.GeoDataFrame({'geometry': polygons, 'category': categories}, crs=crs)
    gdf.to_file(output_path, driver='ESRI Shapefile')

def masks_to_shapefile_with_class_no_geo_separate(masks, output_path):
    polygons = []
    categories = []

    kernel = np.ones((2,2), np.uint8)

    for mask in masks:
        mask_array = mask['segmentation'].astype(np.uint8)

        eroded_mask = cv2.erode(mask_array, kernel, iterations=1)

        mask_category = mask.get('category', 0)

        results = ({'properties': {'raster_val': v}, 'geometry': s}
                   for i, (s, v) in enumerate(shapes(eroded_mask, mask=eroded_mask)))
        
        for result in results:
            geom = result['geometry']
            if geom['type'] == 'Polygon':
                polygons.append(Polygon(geom['coordinates'][0]))
                categories.append(mask_category)

    gdf = gpd.GeoDataFrame({'geometry': polygons, 'category': categories})
    gdf.to_file(output_path, driver='ESRI Shapefile')



def assign_categories_to_masks(label_path, masks):
    label_dataset = rasterio.open(label_path)
    label_array = label_dataset.read(1)

    for mask in masks:
        mask_array = mask['segmentation']
        labels_in_mask = label_array[mask_array]
        most_common = np.bincount(labels_in_mask.flat).argmax()
        mask['category'] = most_common
        

def shp_to_geotiff(input_shp, reference_tif, output_tif, output_png):
    ref_ds = gdal.Open(reference_tif, gdal.GA_ReadOnly)
    geo_transform = ref_ds.GetGeoTransform()
    projection = ref_ds.GetProjection()
    x_size = ref_ds.RasterXSize
    y_size = ref_ds.RasterYSize

    driver = gdal.GetDriverByName('GTiff')
    out_ras = driver.Create(output_tif, x_size, y_size, 1, gdal.GDT_Byte)
    out_ras.SetGeoTransform(geo_transform)
    out_ras.SetProjection(projection)
    
    ds = ogr.Open(input_shp)
    layer = ds.GetLayer()
    band = out_ras.GetRasterBand(1)
    band.Fill(0)
    gdal.RasterizeLayer(out_ras, [1], layer, options=["ATTRIBUTE=category"])
    
    array = band.ReadAsArray()*255
    img = Image.fromarray(array)
    img.save(output_png)

    band.FlushCache()
    out_ras = None

def shp_to_tiff(input_shp, reference_tif, output_tif, output_png):
    ref_ds = gdal.Open(reference_tif, gdal.GA_ReadOnly)
    x_size = ref_ds.RasterXSize
    y_size = ref_ds.RasterYSize

    driver = gdal.GetDriverByName('GTiff')
    out_ras = driver.Create(output_tif, x_size, y_size, 1, gdal.GDT_Byte)
    
    ds = ogr.Open(input_shp)
    layer = ds.GetLayer()
    band = out_ras.GetRasterBand(1)
    band.Fill(0)
    gdal.RasterizeLayer(out_ras, [1], layer, options=["ATTRIBUTE=category"])
    
    array = band.ReadAsArray()*255
    img = Image.fromarray(array)
    img.save(output_png)

    band.FlushCache()
    out_ras = None