from utils import match_images_and_masks
from plot_functions import plotNeuronsRegions
from image import Image
import os
import numpy as np
from matplotlib import pyplot as plt

image_folder = 'images_mix4'
mask_folder = 'masks'
roi_folder = 'masks_regions_extended'

# sham_image_folder ="images_sham"
# sham_mask_folder = "masks_sham"

# sham_images = match_images_and_masks(sham_image_folder, sham_mask_folder, roi_folder)

images = match_images_and_masks(image_folder, mask_folder, roi_folder)
print(len(images))
image_objects = []
for image_info in images:
    name = os.path.basename(image_info[0])  
    image_obj = Image(name, image_info[0], image_info[1], image_info[2])
    image_objects.append(image_obj)
# for image_info in sham_images:
#     name = os.path.basename(image_info[0])  
#     image_obj = Image(name, image_info[0], image_info[1], image_info[2])
#     image_objects.append(image_obj)

cluster_values = []
mean_background = []
mean_signal = []
for object in image_objects:
    print(object.name, len(object.nuclei))
    mean_signal.append(object.getMeanFluorescenceChannel(channel=3))
    object.clusterMasks = object.classifyCells(inspect_classified_masks=True, plot_selectionChannel=False)
    object.clusterNuclei = object.measureClusterNucleiInImage(object.clusterMasks)
    object.ca1Clusters = object.measureClusterNucleiInRegion(object.roi[0])
    object.ca3Clusters = object.measureClusterNucleiInRegion(object.roi[1])
    object.dgClusters = object.measureClusterNucleiInRegion(object.roi[2])
    print(len(object.ca1Clusters[0]))
    #background = object.measureBackground()
    #mean_background.append(background)
    print('Non neurons: ',len(object.clusterNuclei[0]))
    print('Immature neurons: ',len(object.clusterNuclei[1]))
    print('Mature neurons: ',len(object.clusterNuclei[2]))
    fluo0, fluo1, fluo2 = object.getMeanFluorescenceChannel(3, clusters=True)
    print(np.mean(fluo0), np.mean(fluo1), np.mean(fluo2))

ca1Clusters = [object.ca1Clusters for object in image_objects]
ca3Clusters = [object.ca3Clusters for object in image_objects]
dgClusters = [object.dgClusters for object in image_objects]

print(len(ca1Clusters[0]), len(ca1Clusters[1]))

plotNeuronsRegions(ca1Clusters, ca3Clusters, dgClusters)