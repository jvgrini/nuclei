from utils import match_images_and_masks, initializeImages, createDataframe, match_images_and_masks_without_ROI
from image import Image
import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


image_folder_wt = 'imagesAndMasks\Mouzuna\wt'
image_folder_n3 = 'imagesAndMasks\Mouzuna/n3'


image_files_wt = match_images_and_masks(image_folder_wt, image_folder_wt, image_folder_wt)
image_files_n3 = match_images_and_masks(image_folder_n3, image_folder_n3, image_folder_n3)
print(image_files_wt)
#image_files_ca3 = match_images_and_masks_without_ROI(image_folder_ca3, mask_folder)
#image_files_dg = match_images_and_masks_without_ROI(image_folder_dg, mask_folder)
image_objects_wt = initializeImages(image_files_wt)
image_objects_n3 = initializeImages(image_files_n3)
#image_objects_ca3 = initializeImages(image_files_ca3)
#image_objects_dg = initializeImages(image_files_dg)
#image_objects_gfp = initializeImages(image_files_gfp)


nucleus_df = pd.DataFrame(columns=['Condition', 'ImageName','Label', 'Area', 'Centroid', 'CellType', 'Location', 'Ch1Intensity', 'Ch2Intensity', 'Ch3Intensity', 'Ch4Intensity', 'gfpPositive'])

for object in image_objects_wt:
    print(object.name, len(object.nuclei))
    object.nuclei = object.getNeurons(channel=3)
    #object.nuclei = object.getPositiveGFP(channel=3)
    object.calculate_nuclei_locations()
    #object.visualize_nuclei_locations()
    object_df = createDataframe(object, condition='WT')
    nucleus_df = pd.concat([nucleus_df, object_df], ignore_index=True)

for object in image_objects_n3:
    print(object.name, len(object.nuclei))
    object.nuclei = object.getNeurons(channel=3)
    #object.nuclei = object.getPositiveGFP(channel=3)
    object.calculate_nuclei_locations()
    #object.visualize_nuclei_locations()
    object_df = createDataframe(object, condition='N3')
    nucleus_df = pd.concat([nucleus_df, object_df], ignore_index=True)

# for object in image_objects_ca3:
#     print(object.name, len(object.nuclei))
#     object.nuclei = object.getNeurons(channel=1)
#     object.nuclei = object.getPositiveGFP(channel=3)
#     object_df = createDataframe(object, condition='CA3')
#     nucleus_df = pd.concat([nucleus_df, object_df], ignore_index=True)

# for object in image_objects_dg:
#     print(object.name, len(object.nuclei))
#     object.nuclei = object.getNeurons(channel=1)
#     object.nuclei = object.getPositiveGFP(channel=3)
#     object_df = createDataframe(object, condition='DG')
#     nucleus_df = pd.concat([nucleus_df, object_df], ignore_index=True)

# for object in image_objects_gfp:
#     print(object.name, len(object.nuclei))
#     object.nuclei = object.classifyCells(inspect_classified_masks=False, plot_selectionChannel=True, channel=3)
#     object_df = createDataframe(object, condition='HeterozygousKnockout')
#     nucleus_df = pd.concat([nucleus_df, object_df], ignore_index=True)

nucleus_df.to_csv("dataAnalysisNotebooks/csv/nuclei_mouzuna.csv", index=False)

