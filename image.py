from utils import getNucleiFromImage
from plot_functions import violinAndBoxplotClusters
import czifile
import numpy as np
import napari
from skimage import io, measure, morphology
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.ndimage import distance_transform_edt, binary_dilation, maximum_filter
from scipy.spatial import cKDTree

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
        self.ch1Intensity = None
        self.ch2Intensity = None
        self.ch3Intensity = None
        self.ch4Intensity = None
    
    def calculateIntensitiesImage(self):
        num_channels = self.image.shape[-1]

        self.ch1Intensity = np.mean(self.image[:, :, :, 0])
        self.ch2Intensity = np.mean(self.image[:, :, :, 1])
        self.ch3Intensity = np.mean(self.image[:, :, :, 2])
        if num_channels >= 4:
            self.ch4Intensity = np.mean(self.image[:, :, :, 3])


    def measureCyto(self, cytoRadiusUm =.5 ):
        
        binaryMask = np.copy(self.masks).astype(bool)    
        
        dilation_radius_voxels = [int(np.ceil(cytoRadiusUm / s)) for s in self.scale]
        # Define the dimensions of the spheroid-shaped structuring element based on the desired size
        z_size = dilation_radius_voxels[0] * 2
        xy_size = dilation_radius_voxels[1]  * 2 

        # Create a mesh grid of coordinates corresponding to the dimensions of the structuring element
        z, x, y = np.ogrid[-z_size // 2:z_size // 2 + 1, -xy_size // 2:xy_size // 2 + 1, -xy_size // 2:xy_size // 2 + 1]
        # Create a spheroid-shaped structuring element
        structuring_element = (z**2 / dilation_radius_voxels[0]**2 + x**2 / dilation_radius_voxels[1]**2 + y**2 / dilation_radius_voxels[2]**2) <= 1
        
        print('selem: ',structuring_element.shape)
        center_z = structuring_element.shape[0] // 2

        # Visualize each z-plane of the structuring element

        # Perform dilation on the binary mask using the defined structuring element
        isotropic_dilated_mask = binary_dilation(binaryMask, structuring_element)
        nucleiSubtracted = np.logical_and(isotropic_dilated_mask, np.logical_not(binaryMask))
        #viewer = napari.view_labels(nucleiSubtracted, scale=self.scale)
        #napari.run()
        #plt.imshow(nucleiSubtracted[4])
        #plt.axis('off')
        #plt.savefig('nucleiSubtracted.pdf', bbox_inches='tight')
        #plt.show()
        nucleus_indices_with_centroids = [(idx, nucleus.label) for idx, nucleus in enumerate(self.nuclei)]

        nuclei_centroids = [nucleus.centroid for nucleus in self.nuclei]
        nuclei_centroids = np.array(nuclei_centroids)
        
        voxel_grid = np.copy(isotropic_dilated_mask).astype(np.int16)
        tree = cKDTree(nuclei_centroids)
        print('unique', np.unique(voxel_grid))
        # Iterate over each pixel in the voxel grid
        true_indices = np.argwhere(voxel_grid == 1)
        print('shape voxel grid: ', voxel_grid.shape)
        print(len(true_indices))

        # Iterate over each index with value True in voxel_grid
        for idx in true_indices:
            # Convert index to three-dimensional coordinates
            pixel = np.array(idx)
            # Find the index of the closest centroid to the current pixel
            closest_idx = tree.query(pixel)[1]
            original_nucleus_index = nucleus_indices_with_centroids[closest_idx][1]
            voxel_grid[idx[0], idx[1], idx[2]] = original_nucleus_index
        
        #plt.imshow(voxel_grid[3])
        #plt.show()
        
        properties = measure.regionprops(voxel_grid, intensity_image=self.image)

        nuclei_dict = {nucleus.label: nucleus for nucleus in self.nuclei}

        for prop in properties:
            region_label = prop.label
            region_mean_intensity = prop.mean_intensity
            if len(region_mean_intensity) > 3:
                ch1_intensity, ch2_intensity, ch3_intensity, ch4_intensity= region_mean_intensity
            else:
                ch1_intensity, ch2_intensity, ch3_intensity= region_mean_intensity
                ch4_intensity = None
            corresponding_nucleus = nuclei_dict.get(region_label)
    
            # Check if the corresponding nucleus object exists
            if corresponding_nucleus is not None:
                # Add mean intensities as attributes to the nucleus object
                corresponding_nucleus.cyto_ch1_intensity = ch1_intensity
                corresponding_nucleus.cyto_ch2_intensity = ch2_intensity
                corresponding_nucleus.cyto_ch3_intensity = ch3_intensity
                corresponding_nucleus.cyto_ch4_intensity = ch4_intensity
        
        #viewer = napari.view_image(self.image, scale=self.scale, channel_axis=3)
        #viewer.add_labels(self.masks, scale=self.scale, name="nuclei")
        #viewer.add_labels(voxel_grid, scale=self.scale, name="Cyto")
        #napari.run()
        pass
    def calculateRoiVolume(self):
        # Initialize volumes for each region
        ca1_volume = 0
        ca3_volume = 0
        dg_volume = 0

        # Calculate properties of each region in the ROI mask
        properties = measure.regionprops(self.roi)
        print('Shape: ', self.roi.shape)
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
                nucleus.location = "DG"
            # elif region == 2:
            #     nucleus.location = "CA3"
            # elif region == 3:
            #     nucleus.location = "DG"
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