import os
import numpy as np
import rasterio
import argparse
from tqdm import tqdm

def extract_features_and_labels(img_folder, label_folder, output_file):
    """
    Extract features and labels from corresponding TIF files in img and label folders.
    
    Args:
        img_folder (str): Path to folder containing multi-band image files
        label_folder (str): Path to folder containing single-band label files
        output_file (str): Path to output TXT file
    """
    # Get all tif files in the img folder
    img_files = sorted([f for f in os.listdir(img_folder) if f.endswith('.tif')])
    label_files = sorted([f for f in os.listdir(label_folder) if f.endswith('.tif')])
    
    # Verify matching files
    if len(img_files) != len(label_files):
        raise ValueError(f"Number of image files ({len(img_files)}) doesn't match number of label files ({len(label_files)})")
    
    for img_file, label_file in zip(img_files, label_files):
        if os.path.splitext(img_file)[0] != os.path.splitext(label_file)[0]:
            raise ValueError(f"Filenames don't match: {img_file} and {label_file}")
    
    # Open output file
    with open(output_file, 'w') as f_out:
        # Write header (we'll determine the number of bands from the first image)
        first_img_path = os.path.join(img_folder, img_files[0])
        with rasterio.open(first_img_path) as src:
            num_bands = src.count
            header = ','.join([f'band_{i+1}' for i in range(num_bands)]) + ',label\n'
            f_out.write(header)
        
        # Process each pair of files
        for img_file, label_file in tqdm(zip(img_files, label_files), total=len(img_files), desc="Processing files"):
            img_path = os.path.join(img_folder, img_file)
            label_path = os.path.join(label_folder, label_file)
            
            # Read image and label data
            with rasterio.open(img_path) as img_src, rasterio.open(label_path) as label_src:
                img_data = img_src.read()  # Shape: (bands, height, width)
                label_data = label_src.read(1)  # Shape: (height, width)
                
                # Get valid pixels (non-nodata)
                if img_src.nodata is not None:
                    valid_mask = img_data[0] != img_src.nodata
                else:
                    valid_mask = np.ones_like(label_data, dtype=bool)
                
                if label_src.nodata is not None:
                    valid_mask = valid_mask & (label_data != label_src.nodata)
                
                # Extract features and labels
                for i in range(valid_mask.shape[0]):
                    for j in range(valid_mask.shape[1]):
                        if valid_mask[i, j]:
                            # Get features for this pixel
                            pixel_features = [str(img_data[band, i, j]) for band in range(num_bands)]
                            pixel_label = str(int(label_data[i, j]))
                            
                            # Write to output file
                            line = ','.join(pixel_features) + ',' + pixel_label + '\n'
                            f_out.write(line)

def main():
    parser = argparse.ArgumentParser(description="Extract features and labels from image and label TIF files")
    parser.add_argument('--img_folder', required=True, help="Path to folder containing multi-band image files")
    parser.add_argument('--label_folder', required=True, help="Path to folder containing single-band label files")
    parser.add_argument('--output', required=True, help="Path to output TXT file")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract features and labels
    extract_features_and_labels(args.img_folder, args.label_folder, args.output)
    print(f"Feature extraction complete. Output saved to {args.output}")

if __name__ == "__main__":
    main()