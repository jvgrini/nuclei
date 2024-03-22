from utils import match_images_and_masks, initializeImages, createDataframe, match_images_and_masks_without_ROI
from image import Image
import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


image_folder_p2wt = 'imagesAndMasks\Mouzuna\wt p2'
image_folder_p2n3 = 'imagesAndMasks\Mouzuna/n3 p2'
image_folder_p8wt = 'imagesAndMasks\Mouzuna\wt p8'
image_folder_p8n3 = 'imagesAndMasks\Mouzuna/n3 p8'


image_files_p2wt = match_images_and_masks(image_folder_p2wt, image_folder_p2wt, image_folder_p2wt)
image_files_p2n3 = match_images_and_masks(image_folder_p2n3, image_folder_p2n3, image_folder_p2n3)
image_files_p8wt = match_images_and_masks(image_folder_p8wt, image_folder_p8wt, image_folder_p8wt)
image_files_p8n3 = match_images_and_masks(image_folder_p8n3, image_folder_p8n3, image_folder_p8n3)

#image_files_ca3 = match_images_and_masks_without_ROI(image_folder_ca3, mask_folder)
#image_files_dg = match_images_and_masks_without_ROI(image_folder_dg, mask_folder)
image_objects_p2wt = initializeImages(image_files_p2wt)
image_objects_p2n3 = initializeImages(image_files_p2n3)
image_objects_p8wt = initializeImages(image_files_p8wt)
image_objects_p8n3 = initializeImages(image_files_p8n3)
#image_objects_ca3 = initializeImages(image_files_ca3)
#image_objects_dg = initializeImages(image_files_dg)
#image_objects_gfp = initializeImages(image_files_gfp)


nucleus_df = pd.DataFrame(columns=['Condition', 'ImageName','Label', 'Area', 'Centroid', 'CellType', 'Location', 'Ch1Intensity', 'Ch2Intensity', 'Ch3Intensity', 'Ch4Intensity', 'gfpPositive'])

for object in image_objects_p2wt:
    print(object.name, len(object.nuclei))
    object.nuclei = object.getNeurons(channel=3)
    #object.nuclei = object.getPositiveGFP(channel=3)
    object.calculate_nuclei_locations()
    #object.visualize_nuclei_locations()
    object_df = createDataframe(object, condition='P2WT')
    nucleus_df = pd.concat([nucleus_df, object_df], ignore_index=True)

for object in image_objects_p2n3:
    print(object.name, len(object.nuclei))
    object.nuclei = object.getNeurons(channel=3)
    #object.nuclei = object.getPositiveGFP(channel=3)
    object.calculate_nuclei_locations()
    #object.visualize_nuclei_locations()
    object_df = createDataframe(object, condition='P2N3')
    nucleus_df = pd.concat([nucleus_df, object_df], ignore_index=True)

for object in image_objects_p8wt:
    print(object.name, len(object.nuclei))
    object.nuclei = object.getNeurons(channel=3)
    #object.nuclei = object.getPositiveGFP(channel=3)
    object.calculate_nuclei_locations()
    #object.visualize_nuclei_locations()
    object_df = createDataframe(object, condition='P8WT')
    nucleus_df = pd.concat([nucleus_df, object_df], ignore_index=True)

for object in image_objects_p8n3:
    print(object.name, len(object.nuclei))
    object.nuclei = object.getNeurons(channel=3)
    #object.nuclei = object.getPositiveGFP(channel=3)
    object.calculate_nuclei_locations()
    #object.visualize_nuclei_locations()
    object_df = createDataframe(object, condition='P8N3')
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

nucleus_df.to_csv("dataAnalysisNotebooks/csv/nuclei_mouzuna_2.csv", index=False)

