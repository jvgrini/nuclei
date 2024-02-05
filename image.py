from utils import getNucleiFromImage, getNucleiFromClusters
from plot_functions import violinAndBoxplotClusters
import numpy as np
import napari
from skimage import io, measure, morphology
from sklearn.cluster import KMeans
import seaborn as sns
from matplotlib import pyplot as plt

class Image:
    def __init__(self, name, imageFilepath, maskFilepath, roi_mask):
        self.name = name
        self.nuclei = getNucleiFromImage(imageFilepath, maskFilepath)
        self.image = io.imread(imageFilepath)
        self.masks = io.imread(maskFilepath)
        self.roi= io.imread(roi_mask)
        self.clusterMasks = None
        self.clusterNuclei = None
                
    def getNuclei(self, nucleiNumber):
        for i in range(len(self.nuclei)):
            if (nucleiNumber == self.nuclei[i].label):
                return self.nuclei[i]
    
    def getNucleiWithinRegion(self, regionMask, mask, clusterMasks = False):
        if clusterMasks:
            nuclei_in_region = [clusterMask * regionMask for clusterMask in mask]
        else:
            nuclei_in_region = regionMask * mask
        return nuclei_in_region
        
    def measureNucleiInRegion(self, roi_mask, mask):
        nucleiMasks = self.getNucleiWithinRegion(roi_mask, mask, clusterMasks=True)
        nuclei = getNucleiFromClusters(self.image, nucleiMasks)
        ## do operation..
        return nuclei
        
    def getMeanFluorescenceChannel(self, channel, clusters=False):
        channelToMeasure = f"ch{channel}Intensity"
        if not clusters:    
            intensityList = [getattr(nucleus, channelToMeasure) for nucleus in self.nuclei]
            return np.mean(intensityList)
        if clusters:
            cluster0_fluo = [getattr(nucleus, channelToMeasure) for nucleus in self.clusterNuclei[0]]
            cluster1_fluo = [getattr(nucleus, channelToMeasure) for nucleus in self.clusterNuclei[1]]
            cluster2_fluo = [getattr(nucleus, channelToMeasure) for nucleus in self.clusterNuclei[2]]
            return cluster0_fluo, cluster1_fluo, cluster2_fluo
        
    def classifyCells(self, numClusters=3, applyROI=True, minArea=250, channel=1, inspect_classified_masks = False, plot_selectionChannel = False):
        channelToMeasure = f"ch{channel}Intensity"   
        intensityList = [getattr(nucleus, channelToMeasure) for nucleus in self.nuclei]
        intensity_values_reshaped = np.array(intensityList).reshape(-1, 1)
        kmeans = KMeans(n_clusters=numClusters, random_state=42, n_init="auto")
        clusters = kmeans.fit_predict(intensity_values_reshaped)
    
        sorted_clusters = np.argsort(kmeans.cluster_centers_.flatten())
        cluster_labels = [np.array([prop.label for prop, cluster in zip(self.nuclei, clusters) if cluster == i])
                      for i in sorted_clusters]
        separatedMasks = [np.zeros_like(self.masks) for _ in range(numClusters)]
        for i, cluster_mask in enumerate(cluster_labels):
            separatedMasks[i][np.isin(self.masks, cluster_mask)] = self.masks[np.isin(self.masks, cluster_mask)]
        
        if inspect_classified_masks:
            # Visualize clustered labels using napari
            spacing = ([0.9278, 0.3459, 0.3459])
            viewer = napari.view_image(self.image, scale=spacing, ndisplay=2, channel_axis=3)

            # Add clustered labels to the viewer
            for i, cluster_mask in enumerate(separatedMasks):
                viewer.add_labels(cluster_mask, name=f'Cluster {i + 1}', scale=spacing)
            napari.run()
        
        if plot_selectionChannel:
            cluster_0_values = intensity_values_reshaped[clusters == sorted_clusters[0]].flatten()
            cluster_1_values = intensity_values_reshaped[clusters == sorted_clusters[1]].flatten()
            cluster_2_values = intensity_values_reshaped[clusters == sorted_clusters[2]].flatten()

            violinAndBoxplotClusters(intensityList, cluster_0_values, cluster_1_values, cluster_2_values)
        return separatedMasks
    def measureBackground(self):
        nucleiMask = morphology.dilation(self.masks, morphology.ball(5))
        nucleiMask = nucleiMask.astype(bool)
    
        # Create a background mask where nuclei are labeled as 0 and background as 1
        backgroundMask = np.logical_not(nucleiMask)

        # spacing = ([0.9278, 0.3459, 0.3459])
        # viewer = napari.view_image(self.image, channel_axis=3, scale=spacing)
        # viewer.add_labels(backgroundMask, name="B", scale=spacing)

        # napari.run()
        properties = measure.regionprops(backgroundMask.astype(np.uint8), self.image)
        for prop in properties:
            intensities = prop.mean_intensity
        intensity = intensities[2]
        print(f"Mean background intensity: {intensity}")
        return intensity
        
        
    def viewImage(self):
        spacing = ([0.9278, 0.3459, 0.3459])
        viewer = napari.view_image(
            self.image,
            channel_axis=3,
            scale = spacing,
            ndisplay=2
            )
        
        viewer.add_labels(
            self.masks,
            name="Labels",
            scale=spacing
        )
        napari.run()