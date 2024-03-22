from utils import match_images_and_masks, initializeImages, createDataframe, match_images_and_masks_without_ROI
from plot_functions import plotNeuronsRegions, plotNeuronsRegionsbyRegion, plotRegionNeuronsDensity
from image import Image
import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

contra_image_folder = 'imagesAndMasks\GFAP\images\contra'
ipsi_image_folder = "imagesAndMasks\GFAP\images\ipsi"
HI_mask_folder = 'imagesAndMasks\GFAP\masks'
roi_folder = 'imagesAndMasks\GFAP\ROI_extended'

sham_image_folder ="imagesAndMasks\GFAP\images\sham"
sham_mask_folder = "imagesAndMasks\GFAP\masks"

sham_images = match_images_and_masks(sham_image_folder, sham_mask_folder,roi_folder)
print(sham_images)
ipsi_images = match_images_and_masks(ipsi_image_folder, HI_mask_folder,roi_folder)
contra_images = match_images_and_masks(contra_image_folder, HI_mask_folder,roi_folder)
contra_objects = initializeImages(contra_images)
ipsi_objects = initializeImages(ipsi_images)
sham_objects = initializeImages(sham_images)


nucleus_df = pd.DataFrame(columns=['Condition', 'ImageName','Label', 'Area', 'Centroid', 'CellType', 'Location', 'Ch1Intensity', 'Ch2Intensity', 'Ch3Intensity', 'Ch4Intensity', 'CytoCh1Intensity','CytoCh2Intensity','CytoCh3Intensity','CytoCh4Intensity'])
image_df = pd.DataFrame(columns=['Condition','ImageName', 'CA1Volume', 'CA3Volume', 'DGVolume','Ch1Intensity','Ch2Intensity','Ch3Intensity','Ch4Intensity'])



for object in contra_objects:
    print(object.name, len(object.nuclei))
    object.nuclei = object.classifyCells(inspect_classified_masks=False, plot_selectionChannel=False)
    object.calculate_nuclei_locations()
    object.calculateRoiVolume()
    object.calculateIntensitiesImage()
    object.g4Background = object.measureBackground()
    object.measureCyto()
    #object.visualize_nuclei_locations()
    object_df = createDataframe(object, condition='Contra')
    nucleus_df = pd.concat([nucleus_df, object_df], ignore_index=True)
    image_df = pd.concat([image_df, pd.DataFrame({'Condition': 'Contra','ImageName': [object.name],
                                                               'CA1Volume': [object.ca1Volume],
                                                               'CA3Volume': [object.ca3Volume],
                                                               'DGVolume': [object.dgVolume],
                                                               'g4Background': [object.g4Background],
                                                               'Ch1Intensity': [object.ch1Intensity],
                                                               'Ch2Intensity': [object.ch2Intensity],
                                                               'Ch3Intensity': [object.ch3Intensity],
                                                               'Ch4Intensity': [object.ch4Intensity],
                                                               'Shape': [object.image.shape]})])



for object in ipsi_objects:
    print(object.name, len(object.nuclei))
    object.nuclei = object.classifyCells(inspect_classified_masks=False, plot_selectionChannel=False)
    object.calculate_nuclei_locations()
    object.calculateRoiVolume()
    object.calculateIntensitiesImage()
    object.g4Background = object.measureBackground()
    object.measureCyto()
    #object.visualize_nuclei_locations()
    object_df = createDataframe(object, condition='Ipsi')
    nucleus_df = pd.concat([nucleus_df, object_df], ignore_index=True)
    image_df = pd.concat([image_df, pd.DataFrame({'Condition': 'Ipsi','ImageName': [object.name],
                                                               'CA1Volume': [object.ca1Volume],
                                                               'CA3Volume': [object.ca3Volume],
                                                               'DGVolume': [object.dgVolume],
                                                               'g4Background': [object.g4Background],
                                                               'Ch1Intensity': [object.ch1Intensity],
                                                               'Ch2Intensity': [object.ch2Intensity],
                                                               'Ch3Intensity': [object.ch3Intensity],
                                                               'Ch4Intensity': [object.ch4Intensity],
                                                               'Shape': [object.image.shape] })])

for object in sham_objects:
    print(object.name, len(object.nuclei))
    object.nuclei = object.classifyCells(inspect_classified_masks=False, plot_selectionChannel=False)
    object.calculate_nuclei_locations()
    object.calculateRoiVolume()
    object.calculateIntensitiesImage()
    object.g4Background = object.measureBackground()
    object.measureCyto()
    #object.visualize_nuclei_locations()
    object_df = createDataframe(object, condition='Sham')
    nucleus_df = pd.concat([nucleus_df, object_df], ignore_index=True)
    image_df = pd.concat([image_df, pd.DataFrame({'Condition': 'Sham','ImageName': [object.name],
                                                               'CA1Volume': [object.ca1Volume],
                                                               'CA3Volume': [object.ca3Volume],
                                                               'DGVolume': [object.dgVolume],
                                                               'g4Background': [object.g4Background],
                                                               'Ch1Intensity': [object.ch1Intensity],
                                                               'Ch2Intensity': [object.ch2Intensity],
                                                               'Ch3Intensity': [object.ch3Intensity],
                                                               'Ch4Intensity': [object.ch4Intensity],
                                                               'Shape': [object.image.shape]})])

nucleus_df.to_csv("dataAnalysisNotebooks/csv/nuclei_gfap.csv", index=False)
image_df.to_csv("dataAnalysisNotebooks/csv/images_gfap.csv", index=False)