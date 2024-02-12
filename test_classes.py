from utils import match_images_and_masks
from plot_functions import plotNeuronsRegions, plotNeuronsRegionsbyRegion, plotRegionNeuronsDensity
from image import Image
import os
import numpy as np
from matplotlib import pyplot as plt

contra_image_folder = 'images_HI_contra'
ipsi_image_folder = "images_HI_ipsi"
HI_mask_folder = 'masks_HI'
roi_folder = 'brain_region_masks_extended'

sham_image_folder ="images_sham"
sham_mask_folder = "masks_sham"

sham_images = match_images_and_masks(sham_image_folder, sham_mask_folder, roi_folder)
ipsi_images = match_images_and_masks(ipsi_image_folder, HI_mask_folder, roi_folder)
contra_images = match_images_and_masks(contra_image_folder, HI_mask_folder, roi_folder)
contra_objects = []
ipsi_objects = []
sham_objects = []
for image_info in contra_images:
    name = os.path.basename(image_info[0])  
    image_obj = Image(name, image_info[0], image_info[1], image_info[2])
    contra_objects.append(image_obj)
for image_info in ipsi_images:
    name = os.path.basename(image_info[0])  
    image_obj = Image(name, image_info[0], image_info[1], image_info[2])
    ipsi_objects.append(image_obj)
for image_info in sham_images:
    name = os.path.basename(image_info[0])  
    image_obj = Image(name, image_info[0], image_info[1], image_info[2])
    sham_objects.append(image_obj)

cluster_values = []
mean_background = []
mean_signal = []
for object in contra_objects:
    print(object.name, len(object.nuclei))
    mean_signal.append(object.getMeanFluorescenceChannel(channel=3))
    object.clusterMasks = object.classifyCells(inspect_classified_masks=False, plot_selectionChannel=False)
    object.clusterNuclei = object.measureClusterNucleiInImage(object.clusterMasks)
    object.ca1Clusters = object.measureClusterNucleiInRegion(object.roi, region=1, inspect_regions=False)
    object.dgClusters = object.measureClusterNucleiInRegion(object.roi, region =2, inspect_regions=False)
    object.ca3Clusters = object.measureClusterNucleiInRegion(object.roi, region=3)

    object.ca1Density = object.getDensity(object.roi, region=1)
    object.dgDensity = object.getDensity(object.roi, region =2)
    object.ca3Density = object.getDensity(object.roi, region=3)

    print(len(object.dgClusters[0]), len(object.dgClusters[1]), len(object.dgClusters[2]))
    #background = object.measureBackground()
    #mean_background.append(background)
    print('Non neurons: ',len(object.clusterNuclei[0]))
    print('Immature neurons: ',len(object.clusterNuclei[1]))
    print('Mature neurons: ',len(object.clusterNuclei[2]))
    fluo0, fluo1, fluo2 = object.getMeanFluorescenceChannel(3, clusters=True)
    print(np.mean(fluo0), np.mean(fluo1), np.mean(fluo2))

for object in ipsi_objects:
    print(object.name, len(object.nuclei))
    mean_signal.append(object.getMeanFluorescenceChannel(channel=3))
    object.clusterMasks = object.classifyCells(inspect_classified_masks=False, plot_selectionChannel=False)
    object.clusterNuclei = object.measureClusterNucleiInImage(object.clusterMasks)
    object.ca1Clusters = object.measureClusterNucleiInRegion(object.roi, region=1, inspect_regions=False)
    object.dgClusters = object.measureClusterNucleiInRegion(object.roi, region =2, inspect_regions=False)
    object.ca3Clusters = object.measureClusterNucleiInRegion(object.roi, region=3)
    
    object.ca1Density = object.getDensity(object.roi, region=1)
    object.dgDensity = object.getDensity(object.roi, region =2)
    object.ca3Density = object.getDensity(object.roi, region=3)    

    print(len(object.dgClusters[0]), len(object.dgClusters[1]), len(object.dgClusters[2]))
    #background = object.measureBackground()
    #mean_background.append(background)
    print('Non neurons: ',len(object.clusterNuclei[0]))
    print('Immature neurons: ',len(object.clusterNuclei[1]))
    print('Mature neurons: ',len(object.clusterNuclei[2]))
    fluo0, fluo1, fluo2 = object.getMeanFluorescenceChannel(3, clusters=True)
    print(np.mean(fluo0), np.mean(fluo1), np.mean(fluo2))

for object in sham_objects:
    print(object.name, len(object.nuclei))
    mean_signal.append(object.getMeanFluorescenceChannel(channel=3))
    object.clusterMasks = object.classifyCells(inspect_classified_masks=False, plot_selectionChannel=False)
    object.clusterNuclei = object.measureClusterNucleiInImage(object.clusterMasks)
    object.ca1Clusters = object.measureClusterNucleiInRegion(object.roi, region=1, inspect_regions=False)
    object.dgClusters = object.measureClusterNucleiInRegion(object.roi, region =2, inspect_regions=False)
    object.ca3Clusters = object.measureClusterNucleiInRegion(object.roi, region=3)

    object.ca1Density = object.getDensity(object.roi, region=1)
    object.dgDensity = object.getDensity(object.roi, region =2)
    object.ca3Density = object.getDensity(object.roi, region=3)

    print(len(object.dgClusters[0]), len(object.dgClusters[1]), len(object.dgClusters[2]))
    #background = object.measureBackground()
    #mean_background.append(background)
    print('Non neurons: ',len(object.clusterNuclei[0]))
    print('Immature neurons: ',len(object.clusterNuclei[1]))
    print('Mature neurons: ',len(object.clusterNuclei[2]))
    fluo0, fluo1, fluo2 = object.getMeanFluorescenceChannel(3, clusters=True)
    print(np.mean(fluo0), np.mean(fluo1), np.mean(fluo2))
plotRegionNeuronsDensity(contra_objects, ipsi_objects,sham_objects)