from skimage import io, measure
import glob
import os

from nucleus import Nucleus
from image import Image

def getNucleiFromImage(imageFilename, maskFilename):
    image = io.imread(imageFilename)
    labels = io.imread(maskFilename)
    properties = measure.regionprops(labels, intensity_image=image)
    
    nuclei = []

    for prop in properties:
        region_label = prop.label
        region_area = prop.area
        region_mean_intensity = prop.mean_intensity
        ch1_intensity, ch2_intensity, ch3_intensity, ch4_intensity= region_mean_intensity
        nuclei.append(
            Nucleus(region_label,
                    region_area,
                    ch1_intensity,
                    ch2_intensity,
                    ch3_intensity,
                    ch4_intensity,
            ))

    return nuclei
def getNucleiFromClusters(image,labels):
    clusters = []
    for cluster in labels:
        nuclei = []
        properties = measure.regionprops(cluster, intensity_image=image)
        for prop in properties:
            region_label = prop.label
            region_area = prop.area
            region_mean_intensity = prop.mean_intensity
            ch1_intensity, ch2_intensity, ch3_intensity, ch4_intensity= region_mean_intensity
            nuclei.append(
                Nucleus(region_label,
                        region_area,
                        ch1_intensity,
                        ch2_intensity,
                        ch3_intensity,
                        ch4_intensity,
                ))
        clusters.append(nuclei)
    return clusters
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

def initializeImages(images):
    objects = []
    for image_info in images:
        name = os.path.basename(image_info[0])  
        image_obj = Image(name, image_info[0], image_info[1], image_info[2])
        objects.append(image_obj)
    return objects