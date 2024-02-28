from skimage import io, measure
import czifile
import glob
import os
import pandas as pd
import numpy as np

from nucleus import Nucleus

def getNucleiFromImage(imageFilename, maskFilename, imageName):

    if '.czi' in imageFilename:
        image = czifile.imread(imageFilename)
        image = np.squeeze(image)
        image = np.transpose(image, (1,2,3,0))
    else:
        image = io.imread(imageFilename)
    labels = io.imread(maskFilename)
    properties = measure.regionprops(labels, intensity_image=image)
    
    nuclei = []

    for prop in properties:
        region_label = prop.label
        region_area = prop.area
        centroid = prop.centroid
        region_mean_intensity = prop.mean_intensity
        if len(region_mean_intensity) > 3:
            ch1_intensity, ch2_intensity, ch3_intensity, ch4_intensity= region_mean_intensity
        else:
            ch1_intensity, ch2_intensity, ch3_intensity= region_mean_intensity
            ch4_intensity = None
        nuclei.append(
            Nucleus(imageName,
                    region_label,
                    region_area,
                    centroid,
                    ch1_intensity,
                    ch2_intensity,
                    ch3_intensity,
                    ch4_intensity,
            ))

    return nuclei
def match_images_and_masks(image_folder, mask_folder, roi_folder=None):
    image_files = []
    images = glob.glob(os.path.join(image_folder, '*.lsm'))
    for image_path in images:
        mask_path = os.path.join(mask_folder, os.path.basename(image_path).replace('.lsm', '_mask.tif'))
        if roi_folder != None:
            roi_path = os.path.join(roi_folder, os.path.basename(image_path).replace('.lsm', '_regions_mask.tif'))
        if os.path.exists(mask_path) and os.path.exists(roi_path):
            image_files.append([image_path, mask_path, roi_path])
    return image_files

def match_images_and_masks_without_ROI(image_folder, mask_folder, roi_folder=None):
    image_files = []
    images = glob.glob(os.path.join(image_folder, '*.lsm'))
    for image_path in images:
        mask_path = os.path.join(mask_folder, os.path.basename(image_path).replace('.lsm', '_mask.tif'))
        image_files.append([image_path, mask_path])
    return image_files

def initializeImages(images):
    from image import Image
    objects = []
    for image_info in images:
        name = os.path.basename(image_info[0])
        if len(image_info) > 2:  
            image_obj = Image(name, image_info[0], image_info[1], image_info[2])
        else:
            image_obj = Image(name, image_info[0], image_info[1])
        objects.append(image_obj)
    return objects

def createDataframe(obj, condition):

    nuclei_data = []

    for nucleus in obj.nuclei:
        nucleus_dict = {
                'Condition': condition,
                'ImageName': nucleus.imageName,
                'Label': nucleus.label,
                'Area': nucleus.area,
                'Centroid': nucleus.centroid,
                'CellType': nucleus.cellType,
                'Location': nucleus.location,
                'Ch1Intensity': nucleus.ch1Intensity,
                'Ch2Intensity': nucleus.ch2Intensity,
                'Ch3Intensity': nucleus.ch3Intensity,
                'Ch4Intensity': nucleus.ch4Intensity,
                'gfpPositive': nucleus.gfpPositive,
                'CytoCh1Intensity': nucleus.cyto_ch1_intensity,
                'CytoCh2Intensity': nucleus.cyto_ch2_intensity,
                'CytoCh3Intensity': nucleus.cyto_ch3_intensity,
                'CytoCh4Intensity': nucleus.cyto_ch4_intensity,
                # Add more attributes as needed
        }
            # Append the nucleus dictionary to the nuclei_data list
        nuclei_data.append(nucleus_dict)

# Create a DataFrame from the list of dictionaries
    nuclei_df = pd.DataFrame(nuclei_data)
    return nuclei_df
