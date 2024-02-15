from utils import match_images_and_masks, initializeImages, createDataframe
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
contra_objects = initializeImages(contra_images)
ipsi_objects = initializeImages(ipsi_images)
sham_objects = initializeImages(sham_images)

cluster_values = []
mean_background = []
mean_signal_sham = []
mean_signal_contra = []
mean_signal_ipsi = []
#Hvis du ser på dette Simen, så vit at jeg driver å endrer hvordan jeg håndterer cellekjerner av forskjellige
#celletyper og posisjoner i hjernen. Planen nå er å i stedet ha dette som en egenskap for hvert enkelt 
#cellekjerneobjekt.

for object in contra_objects:
    print(object.name, len(object.nuclei))
    mean_signal_contra.append(object.getMeanFluorescenceChannel(channel=1))
    object.nuclei = object.classifyCells(inspect_classified_masks=False, plot_selectionChannel=False)
    df = createDataframe(object)
    df.to_csv("test.csv", index=False)
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
    mean_signal_ipsi.append(object.getMeanFluorescenceChannel(channel=1))
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
    mean_signal_sham.append(object.getMeanFluorescenceChannel(channel=1))
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

print(f"Mean Neun sham: {mean_signal_sham}")
print(f"Mean Neun contra: {mean_signal_contra}")
print(f"Mean Neun ipsi: {mean_signal_ipsi}")
plotRegionNeuronsDensity(contra_objects, ipsi_objects,sham_objects)