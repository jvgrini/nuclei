from utils import getNucleiFromImage
from plot_functions import violinAndBoxplotClusters
import czifile
import numpy as np
import napari
from skimage import io, measure, morphology
from sklearn.cluster import KMeans
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.ndimage import distance_transform_edt, binary_dilation, maximum_filter
from scipy.spatial import Voronoi, voronoi_plot_2d

class Image:
    def __init__(self, name, imageFilepath, maskFilepath, roi_mask = None):
        self.name = name
        self.scale = ([0.9278, 0.3459, 0.3459])
        self.nuclei = getNucleiFromImage(imageFilepath, maskFilepath, self.name)
        self.masks = io.imread(maskFilepath)
        if '.czi' in imageFilepath:
            self.image = czifile.imread(imageFilepath)
            self.image = np.squeeze(self.image)
            self.image = np.transpose(self.image, (1,2,3,0))
        else:
            self.image = io.imread(imageFilepath)
        if roi_mask != None:
            self.roi= io.imread(roi_mask)
        self.ca1Volume = None
        self.ca3Volume = None
        self.dgVolume = None
    
    def measureCyto(self,visualizeDistanceTransform=False):
        max_dilation_distance = [3 / scale_factor for scale_factor in self.scale]
    
        combined_mask = self.masks.astype(bool)
        distance_transform = distance_transform_edt(~combined_mask)

        if visualizeDistanceTransform:
            # Visualize the distance transform
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(combined_mask[3], cmap='gray')
            plt.title('Combined Nuclei Mask')

            plt.subplot(1, 2, 2)
            plt.imshow(distance_transform[3], cmap='jet')
            plt.colorbar(label='Distance')
            plt.title('Distance Transform')
            plt.show()

        cytoplasm_masks = []
        unique_labels = np.unique(self.masks)
        for label in unique_labels:
            print('yay')
            # Initialize the initial cytoplasm mask as a copy of the nucleus mask
            nucleus_mask = self.masks == label
            cytoplasm_mask = nucleus_mask.copy()
            
            # Initialize the previous distance transform value
            prev_distance_transform = np.inf
            
            # Dilate the cytoplasm mask iteratively
            for _ in range(int(np.ceil(max(max_dilation_distance)))):
                # Dilate the cytoplasm mask along each axis separately
                next_cytoplasm_mask = binary_dilation(cytoplasm_mask)
                
                # Compute the distance transform for the next cytoplasm mask
                next_distance_transform = distance_transform_edt(~next_cytoplasm_mask)
                
                # Find the maximum distance within the nucleus region
                nucleus_distance = np.max(next_distance_transform[nucleus_mask])
                
                # Check if the next distance transform value is greater than the previous or if the nucleus distance is smaller than the previous
                if np.max(next_distance_transform) > prev_distance_transform or nucleus_distance < prev_nucleus_distance:
                    # If we are getting closer to the next nucleus or reaching maximum dilation, stop the dilation
                    break
                
                # Update the cytoplasm mask and the previous distance transform value
                cytoplasm_mask = next_cytoplasm_mask
                prev_distance_transform = np.max(next_distance_transform)
                prev_nucleus_distance = nucleus_distance
            
            # Add the cytoplasm mask to the list
            cytoplasm_masks.append(cytoplasm_mask)

        cytoplasm_masks_array = np.array(cytoplasm_masks)
        viewer = napari.Viewer()
        # Add cytoplasm masks to the viewer
        viewer.add_labels(self.masks,scale=self.scale)
        viewer.add_labels(cytoplasm_masks_array, scale=self.scale,name='Cyto')
        napari.run()

        pass
    def calculateRoiVolume(self):
        # Initialize volumes for each region
        ca1_volume = 0
        ca3_volume = 0
        dg_volume = 0

        # Calculate properties of each region in the ROI mask
        properties = measure.regionprops(self.roi)

        # Iterate over properties of each region
        for prop in properties:
            region_label = prop.label
            region_area = prop.area

            if region_label == 1: 
                ca1_volume += region_area
            elif region_label == 2:  
                ca3_volume += region_area
            elif region_label == 3:  
                dg_volume += region_area

        self.ca1Volume = ca1_volume * np.prod(self.scale)
        self.ca3Volume = ca3_volume * np.prod(self.scale)
        self.dgVolume = dg_volume * np.prod(self.scale)
        

    
    def calculate_nuclei_locations(self):
        for nucleus in self.nuclei:
            # Calculate the centroid of the nucleus
            centroid_z, centroid_y, centroid_x = nucleus.centroid  # Assuming centroid is in (x, y, z) format

            # Determine the region based on the roi mask
            region = self.roi[int(centroid_z), int(centroid_y), int(centroid_x)]  # Adjust indexing for 3D

            # Assign location based on the region
            if region == 1:
                nucleus.location = "CA1"
            elif region == 2:
                nucleus.location = "CA3"
            elif region == 3:
                nucleus.location = "DG"
            else:
                nucleus.location = "Undefined"
    def visualize_nuclei_locations(self):
    # Create a Napari viewer
        viewer = napari.Viewer()

    # Add ROI mask as an image layer
        viewer.add_image(self.roi, colormap='gray', name='ROI Mask')

        # Extract nuclei centroids and locations
        centroids = np.array([nucleus.centroid for nucleus in self.nuclei])
        locations = [nucleus.location for nucleus in self.nuclei]

        # Create separate points layers for each location
        for location in set(locations):
            indices = [i for i, loc in enumerate(locations) if loc == location]
            centroids_location = centroids[indices]
            if location == 'CA1':
                color = 'green'
     
            elif location == 'DG':
                color = 'blue'
 
            elif location == 'CA3':
                color = 'red'
   
            else:
                color = 'yellow'  # Default color for unknown locations

            viewer.add_points(centroids_location[:, [0, 1, 2]], size=10, symbol='o', edge_color=color, face_color=color, name=f'Nuclei {location}')

        # Run the Napari viewer
        napari.run()
        
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
    
    def getPositiveGFP(self, channel=1):
        channelToMeasure = f"ch{channel}Intensity"   
        intensityList = [getattr(nucleus, channelToMeasure) for nucleus in self.nuclei]
        intensity_values_reshaped = np.array(intensityList).reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto")
        clusters = kmeans.fit_predict(intensity_values_reshaped)

        sorted_clusters = np.argsort(kmeans.cluster_centers_.flatten())
        for nucleus, cluster_label in zip(self.nuclei, clusters):
            if cluster_label == sorted_clusters[1]:
                nucleus.gfpPositive = True
        return self.nuclei
    
    def getNeurons(self, channel=1):
        channelToMeasure = f"ch{channel}Intensity"   
        intensityList = [getattr(nucleus, channelToMeasure) for nucleus in self.nuclei]
        intensity_values_reshaped = np.array(intensityList).reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto")
        clusters = kmeans.fit_predict(intensity_values_reshaped)

        sorted_clusters = np.argsort(kmeans.cluster_centers_.flatten())
        for nucleus, cluster_label in zip(self.nuclei, clusters):
            if cluster_label == sorted_clusters[1]:
                nucleus.cellType = 'Neuron'
        return self.nuclei
            

    def classifyCells(self, numClusters=3, applyROI=True, minArea=250, channel=1, inspect_classified_masks = False, plot_selectionChannel = False):
        channelToMeasure = f"ch{channel}Intensity"   
        intensityList = [getattr(nucleus, channelToMeasure) for nucleus in self.nuclei]
        intensity_values_reshaped = np.array(intensityList).reshape(-1, 1)
        kmeans = KMeans(n_clusters=numClusters, random_state=42, n_init="auto")
        clusters = kmeans.fit_predict(intensity_values_reshaped)
    
        sorted_clusters = np.argsort(kmeans.cluster_centers_.flatten())

        for nucleus, cluster_label in zip(self.nuclei, clusters):
            if cluster_label == sorted_clusters[1]:  # Check for neunPositiveLow
                nucleus.cellType = 'neunPositiveLow'
            elif cluster_label == sorted_clusters[2]:  # Check for neunPositive
                nucleus.cellType = 'neunPositive'
        if inspect_classified_masks:
            # Visualize clustered labels using napari
            spacing = ([0.9278, 0.3459, 0.3459])
            viewer = napari.view_image(self.image, scale=spacing, ndisplay=2, channel_axis=3)

            cluster_labels = [np.array([prop.label for prop, cluster in zip(self.nuclei, clusters) if cluster == i])
                      for i in sorted_clusters]
            separatedMasks = [np.zeros_like(self.masks) for _ in range(numClusters)]
            for i, cluster_mask in enumerate(cluster_labels):
                separatedMasks[i][np.isin(self.masks, cluster_mask)] = self.masks[np.isin(self.masks, cluster_mask)]
            for i, cluster_mask in enumerate(separatedMasks):
                viewer.add_labels(cluster_mask, name=f'Cluster {i + 1}', scale=spacing)

            napari.run()
        
        if plot_selectionChannel:
            cluster_0_values = intensity_values_reshaped[clusters == sorted_clusters[0]].flatten()
            cluster_1_values = intensity_values_reshaped[clusters == sorted_clusters[1]].flatten()
            cluster_2_values = intensity_values_reshaped[clusters == sorted_clusters[2]].flatten()

            print(f"Cluster 0 median: {np.median(cluster_0_values)}")
            print(f"Cluster 1 median: {np.median(cluster_1_values)}")
            print(f"Cluster 2 median: {np.median(cluster_2_values)}")
            violinAndBoxplotClusters(intensityList, cluster_0_values, cluster_1_values, cluster_2_values)
        return self.nuclei
    
    def getDensity(self, roi, region):
        labeled_regions = measure.label(roi)
        binary_mask = (labeled_regions == region)
        masksInRegion = binary_mask * self.clusterMasks
        volume = np.sum(binary_mask) * np.prod(self.scale)

        clusters = getNucleiFromClusters(self.image, masksInRegion)
        density = [((len(nuclei) / volume) * 100**3) for nuclei in clusters]
        print("Density: ", density)
        return density

    def measureBackground(self):
        nucleiMask = morphology.dilation(self.masks, morphology.ball(5))
        nucleiMask = nucleiMask.astype(bool)
    
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