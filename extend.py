import os
from skimage import io
import numpy as np

def extend_region_masks(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all TIFF files in the input folder
    tiff_files = [file for file in os.listdir(input_folder) if file.endswith('.tif')]
    print(tiff_files)
    for tiff_file in tiff_files:
        # Read the TIFF image
        image_path = os.path.join(input_folder, tiff_file)
        image = io.imread(image_path)
        max_values = np.max(image, axis=0)
    
    # Set the values in the original image to the maximum values
        image[:, :, :] = max_values
        # Save the extended mask to the output folder
        output_path = os.path.join(output_folder, tiff_file)
        io.imsave(output_path, image)

        print(f"Extended mask saved: {output_path}")

extend_region_masks("brain_region_masks", "brain_region_masks_extended")