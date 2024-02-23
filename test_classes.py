from utils import match_images_and_masks, initializeImages, createDataframe
from plot_functions import plotNeuronsRegions, plotNeuronsRegionsbyRegion, plotRegionNeuronsDensity
from image import Image
import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

contra_image_folder = 'imagesAndMasks/images_HI_contra'
ipsi_image_folder = "imagesAndMasks/images_HI_ipsi"
HI_mask_folder = 'imagesAndMasks/masks_HI'
roi_folder = 'imagesAndMasks/brain_region_masks_extended'

sham_image_folder ="imagesAndMasks/images_sham"
sham_mask_folder = "imagesAndMasks/masks_sham"

sham_images = match_images_and_masks(sham_image_folder, sham_mask_folder, roi_folder)
ipsi_images = match_images_and_masks(ipsi_image_folder, HI_mask_folder, roi_folder)
contra_images = match_images_and_masks(contra_image_folder, HI_mask_folder, roi_folder)
contra_objects = initializeImages(contra_images)
ipsi_objects = initializeImages(ipsi_images)
sham_objects = initializeImages(sham_images)

nucleus_df = pd.DataFrame(columns=['Condition', 'ImageName','Label', 'Area', 'Centroid', 'CellType', 'Location', 'Ch1Intensity', 'Ch2Intensity', 'Ch3Intensity', 'Ch4Intensity'])
image_df = pd.DataFrame(columns=['Condition','ImageName', 'CA1Volume', 'CA3Volume', 'DGVolume','Ch1Intensity','Ch2Intensity','Ch3Intensity','Ch4Intensity'])



for object in contra_objects:
    print(object.name, len(object.nuclei))
    object.nuclei = object.classifyCells(inspect_classified_masks=False, plot_selectionChannel=False)
    object.calculate_nuclei_locations()
    object.calculateRoiVolume()
    object.calculateIntensitiesImage()
    #object.measureCyto()
    #object.visualize_nuclei_locations()
    object_df = createDataframe(object, condition='Contra')
    nucleus_df = pd.concat([nucleus_df, object_df], ignore_index=True)
    image_df = pd.concat([image_df, pd.DataFrame({'Condition': 'Contra','ImageName': [object.name],
                                                               'CA1Volume': [object.ca1Volume],
                                                               'CA3Volume': [object.ca3Volume],
                                                               'DGVolume': [object.dgVolume],
                                                               'Ch1Intensity': [object.ch1Intensity],
                                                               'Ch2Intensity': [object.ch2Intensity],
                                                               'Ch3Intensity': [object.ch3Intensity],
                                                               'Ch4Intensity': [object.ch4Intensity]})])

for object in ipsi_objects:
    print(object.name, len(object.nuclei))
    object.nuclei = object.classifyCells(inspect_classified_masks=False, plot_selectionChannel=False)
    object.calculate_nuclei_locations()
    object.calculateRoiVolume()
    object.calculateIntensitiesImage()
    #object.measureCyto()
    print(object.ca1Volume, object.ca3Volume, object.dgVolume)
    #object.visualize_nuclei_locations()
    object_df = createDataframe(object, condition='Ipsi')
    nucleus_df = pd.concat([nucleus_df, object_df], ignore_index=True)
    image_df = pd.concat([image_df, pd.DataFrame({'Condition': 'Ipsi','ImageName': [object.name], 
                                                               'CA1Volume': [object.ca1Volume],
                                                               'CA3Volume': [object.ca3Volume],
                                                               'DGVolume': [object.dgVolume],
                                                               'Ch1Intensity': [object.ch1Intensity],
                                                               'Ch2Intensity': [object.ch2Intensity],
                                                               'Ch3Intensity': [object.ch3Intensity],
                                                               'Ch4Intensity': [object.ch4Intensity]})])

for object in sham_objects:
    print(object.name, len(object.nuclei))
    object.nuclei = object.classifyCells(inspect_classified_masks=False, plot_selectionChannel=False)
    object.calculate_nuclei_locations()
    object.calculateRoiVolume()
    object.calculateIntensitiesImage()
    #object.measureCyto()
    #object.visualize_nuclei_locations()
    object_df = createDataframe(object,condition='Sham')
    nucleus_df = pd.concat([nucleus_df, object_df], ignore_index=True)
    image_df = pd.concat([image_df, pd.DataFrame({'Condition': 'Sham','ImageName': [object.name],
                                                               'CA1Volume': [object.ca1Volume],
                                                               'CA3Volume': [object.ca3Volume],
                                                               'DGVolume': [object.dgVolume],
                                                               'Ch1Intensity': [object.ch1Intensity],
                                                               'Ch2Intensity': [object.ch2Intensity],
                                                               'Ch3Intensity': [object.ch3Intensity],
                                                               'Ch4Intensity': [object.ch4Intensity]})])

#nucleus_df.to_csv("dataAnalysisNotebooks/csv/nuclei.csv", index=False)
image_df.to_csv("dataAnalysisNotebooks/csv/images.csv", index=False)