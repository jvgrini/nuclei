from utils import match_images_and_masks, initializeImages, createDataframe, match_images_and_masks_without_ROI
from plot_functions import plotNeuronsRegions, plotNeuronsRegionsbyRegion, plotRegionNeuronsDensity
from image import Image
import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

contraMix2_folder = 'imagesAndMasks\images_HI_contra_mix2'
ipsiMix2_folder = 'imagesAndMasks\images_HI_ipsi_mix2'

contra_image_folder = 'imagesAndMasks\images_HI_contra'
ipsi_image_folder = "imagesAndMasks\images_HI_ipsi"
HI_mask_folder = 'imagesAndMasks\masks_HI'
roi_folder = 'imagesAndMasks/brain_region_masks_extended'

sham_image_folder ="imagesAndMasks\images_sham"
sham_mask_folder = "imagesAndMasks\masks_sham"

sham_images = match_images_and_masks(sham_image_folder, sham_mask_folder, roi_folder)
ipsi_images = match_images_and_masks(ipsi_image_folder, HI_mask_folder, roi_folder)
contra_images = match_images_and_masks(contra_image_folder, HI_mask_folder, roi_folder)
contra_images2 = match_images_and_masks_without_ROI(contraMix2_folder, HI_mask_folder)
ipsi_images2 = match_images_and_masks_without_ROI(ipsiMix2_folder, HI_mask_folder)
contra_objects = initializeImages(contra_images)
ipsi_objects = initializeImages(ipsi_images)
sham_objects = initializeImages(sham_images)
contra_objects2 = initializeImages(contra_images2)
ipsi_objects2 = initializeImages(ipsi_images2)

nucleus_df = pd.DataFrame(columns=['Condition', 'ImageName','Label', 'Area', 'Centroid', 'CellType', 'Location', 'Ch1Intensity', 'Ch2Intensity', 'Ch3Intensity', 'Ch4Intensity', 'CytoCh1Intensity','CytoCh2Intensity','CytoCh3Intensity','CytoCh4Intensity'])
image_df = pd.DataFrame(columns=['Condition','ImageName', 'CA1Volume', 'CA3Volume', 'DGVolume','Ch1Intensity','Ch2Intensity','Ch3Intensity','Ch4Intensity'])



for object in contra_objects:
    print(object.name, len(object.nuclei))
    object.nuclei = object.classifyCells(inspect_classified_masks=False, plot_selectionChannel=False)
    object.calculate_nuclei_locations()
    object.calculateRoiVolume()
    object.calculateIntensitiesImage()
    object.g4Background = object.measureBackground()
    #object.measureCyto()
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
                                                               'Ch4Intensity': [object.ch4Intensity]})])

for object in contra_objects2:
    print(object.name, len(object.nuclei))
    object.nuclei = object.classifyCells(inspect_classified_masks=False, plot_selectionChannel=False)
    #object.calculate_nuclei_locations()
    #object.calculateRoiVolume()
    object.calculateIntensitiesImage()
    object.g4Background = object.measureBackground()
    #object.measureCyto()
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
                                                               'Ch4Intensity': [object.ch4Intensity]})])

for object in ipsi_objects2:
    print(object.name, len(object.nuclei))
    object.nuclei = object.classifyCells(inspect_classified_masks=False, plot_selectionChannel=False)
    #object.calculate_nuclei_locations()
    #object.calculateRoiVolume()
    object.calculateIntensitiesImage()
    object.g4Background = object.measureBackground()
    #object.measureCyto()
    #object.visualize_nuclei_locations()
    object_df = createDataframe(object, condition='Ipsi')
    nucleus_df = pd.concat([nucleus_df, object_df], ignore_index=True)
    image_df = pd.concat([image_df, pd.DataFrame({'Condition': 'Contra','ImageName': [object.name],
                                                               'CA1Volume': [object.ca1Volume],
                                                               'CA3Volume': [object.ca3Volume],
                                                               'DGVolume': [object.dgVolume],
                                                               'g4Background': [object.g4Background],
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
    object.g4Background = object.measureBackground()
    #object.measureCyto()
    #object.visualize_nuclei_locations()
    object_df = createDataframe(object, condition='Ipsi')
    nucleus_df = pd.concat([nucleus_df, object_df], ignore_index=True)
    image_df = pd.concat([image_df, pd.DataFrame({'Condition': 'Contra','ImageName': [object.name],
                                                               'CA1Volume': [object.ca1Volume],
                                                               'CA3Volume': [object.ca3Volume],
                                                               'DGVolume': [object.dgVolume],
                                                               'g4Background': [object.g4Background],
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
    object.g4Background = object.measureBackground()
    #object.measureCyto()
    #object.visualize_nuclei_locations()
    object_df = createDataframe(object, condition='Sham')
    nucleus_df = pd.concat([nucleus_df, object_df], ignore_index=True)
    image_df = pd.concat([image_df, pd.DataFrame({'Condition': 'Contra','ImageName': [object.name],
                                                               'CA1Volume': [object.ca1Volume],
                                                               'CA3Volume': [object.ca3Volume],
                                                               'DGVolume': [object.dgVolume],
                                                               'g4Background': [object.g4Background],
                                                               'Ch1Intensity': [object.ch1Intensity],
                                                               'Ch2Intensity': [object.ch2Intensity],
                                                               'Ch3Intensity': [object.ch3Intensity],
                                                               'Ch4Intensity': [object.ch4Intensity]})])

nucleus_df.to_csv("dataAnalysisNotebooks/csv/nuclei_g4_2.csv", index=False)
image_df.to_csv("dataAnalysisNotebooks/csv/images_g4_2.csv", index=False)