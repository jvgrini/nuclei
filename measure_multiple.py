from matplotlib import pyplot as plt
from scipy import stats
import numpy as np
from glob import glob
import os
import seaborn as sns
from skimage import io, measure, filters
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import re
import napari

spacing = ([0.3459, 0.3459, 0.9278])

def extend_region_masks(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all TIFF files in the input folder
    tiff_files = [file for file in os.listdir(input_folder) if file.endswith('.tif')]

    for tiff_file in tiff_files:
        # Read the TIFF image
        image_path = os.path.join(input_folder, tiff_file)
        image = io.imread(image_path)
        max_values = np.max(image, axis=0)
    
    # Set the values in the original image to the maximum values
        image[:, :, :] = max_values
        # Save the extended mask to the output folder
        output_path = os.path.join(output_folder, tiff_file)
        io.imsave(output_path, image)

        print(f"Extended mask saved: {output_path}")

def classify_neurons2(labels, properties, name, num_clusters=3, savefig=False, inspect_classified_masks=False, region_label = None):
    pattern = re.compile(r'images_HI\\(.*?)\s+G4')

    # Use the regular expression to extract the desired part of the file name
    match = pattern.search(name)
    if match:
        new_name = match.group(1)
        print(new_name)
    else:
        # Handle the case where the input name doesn't match the expected pattern
        print(f'Invalid name format: {name}')
        new_name = name
    min_area_threshold = 500
    labels = io.imread(labels)
     # Get labeled regions and their properties
    regions = measure.regionprops(labels)

    # Filter labels based on area threshold
    valid_labels = [region.label for region in regions if region.area >= min_area_threshold]

    # Extract Channel2_intensity values from properties
    intensity_values = [prop['Channel1_intensity'] for i, prop in enumerate(properties) if i + 1 in valid_labels]

    # Reshape the intensity values array for clustering
    intensity_values_reshaped = np.array(intensity_values).reshape(-1, 1)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init="auto")
    clusters = kmeans.fit_predict(intensity_values_reshaped)
    
    sorted_clusters = np.argsort(kmeans.cluster_centers_.flatten())

    # Extract labels for each cluster
    cluster_labels = [np.array([prop['label'] for prop, cluster in zip(properties, clusters) if cluster == i])
                      for i in sorted_clusters]

    # Initialize empty masks for each cluster
    separated_labels = [np.zeros_like(labels) for _ in range(num_clusters)]

    for i, cluster_mask in enumerate(cluster_labels):
        # Use boolean indexing to assign labels directly
        separated_labels[i][np.isin(labels, cluster_mask)] = labels[np.isin(labels, cluster_mask)]

    if inspect_classified_masks:
        # Visualize clustered labels using napari
        spacing = (1.0, 1.0, 1.0)  # Adjust this based on your data
        viewer = napari.view_labels(labels, scale=spacing, ndisplay=3)

        # Add clustered labels to the viewer
        for i, cluster_mask in enumerate(separated_labels):
            viewer.add_labels(cluster_mask, name=f'Cluster {i + 1}')
        napari.run()
    cluster_0_values = intensity_values_reshaped[clusters == sorted_clusters[0]].flatten()
    cluster_1_values = intensity_values_reshaped[clusters == sorted_clusters[1]].flatten()
    cluster_2_values = intensity_values_reshaped[clusters == sorted_clusters[2]].flatten()

    fig, ax = plt.subplots()
    # Create a violin plot for all data points
    sns.violinplot(y=np.array(intensity_values).flatten(), inner=None, color="lightblue", ax=ax, zorder=1)
    # Plot boxplots for each cluster at the specified x-values
    ax.boxplot([cluster_0_values], positions=[0], widths=0.2, patch_artist=True,
           boxprops=dict(facecolor="seagreen", zorder=2, alpha=.7),
           medianprops=dict(color="black"),
           whiskerprops=dict(color="darkgreen"),
           capprops=dict(color="darkgreen"))

    ax.boxplot([cluster_1_values], positions=[0], widths=0.2, patch_artist=True,
           boxprops=dict(facecolor="lightcoral", zorder=2, alpha=.7),
           medianprops=dict(color="black"),
           whiskerprops=dict(color="firebrick"),
           capprops=dict(color="firebrick"))
    ax.boxplot([cluster_2_values], positions=[0], widths=0.2, patch_artist=True,
           boxprops=dict(facecolor="sandybrown", zorder=2, alpha=.7),
           medianprops=dict(color="black"),
           whiskerprops=dict(color="tan"),
           capprops=dict(color="tan"))
    plt.xticks([0],"")
    plt.title(new_name)
    plt.ylabel('NeuN intensity')
    save_path = os.path.join("D:\\Users\\Jonas\\plots", f"{new_name}_neurons.pdf")
    # plt.savefig(save_path)

    plt.show()

    # Count the number of neurons in each cluster
    for i, cluster_mask in enumerate(cluster_labels):
        print(f"Number of neurons in Cluster {i + 1}: {len(cluster_mask)}")

    return [len(cluster_0_values), len(cluster_1_values), len(cluster_2_values)]
def classify_neurons(labels, properties, name, num_clusters=2, savefig=False, inspect_classified_masks = True):

    pattern = re.compile(r'Images\/(.*?)\s+G4')

    # Use the regular expression to extract the desired part of the file name
    match = pattern.search(name)
    if match:
        new_name = match.group(1)

        # You can use the new_name or return it as needed
    else:
        # Handle the case where the input name doesn't match the expected pattern
        print(f'Invalid name format: {name}')
        new_name = name
        
    labels = io.imread(labels)
    # Extract Channel2_intensity values from properties
    intensity_values = [prop['Channel1_intensity'] for prop in properties]

    # Reshape the intensity values array for clustering
    intensity_values_reshaped = np.array(intensity_values).reshape(-1, 1)


    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init="auto")
    clusters = kmeans.fit_predict(intensity_values_reshaped)
    
    max_intensity_cluster_index = np.argmax(kmeans.cluster_centers_)


    # Extract labels for the cluster with the maximum intensity value
    positive_labels = [prop['label'] for prop, cluster in zip(properties, clusters) if cluster == max_intensity_cluster_index]

    # Extract labels for the other cluster
    negative_labels = [prop['label'] for prop, cluster in zip(properties, clusters) if cluster != max_intensity_cluster_index]


    # Initialize an empty mask for the separated positive labels
    separated_positive_labels = np.zeros_like(labels)

    positive_mask = np.isin(labels, positive_labels)

    # Use boolean indexing to assign positive labels directly
    separated_positive_labels[positive_mask] = labels[positive_mask]

    if inspect_classified_masks:
        # Visualize positively clustered labels using napari
        spacing = (1.0, 1.0, 1.0)  # Adjust this based on your data
        viewer = napari.view_labels(labels, scale=spacing, ndisplay=3)
        
        # Add positively clustered labels to the viewer
        viewer.add_labels(separated_positive_labels, name='Positive Labels')
        napari.run()

    cluster_0_mask = (clusters == 0)
    cluster_1_mask = (clusters == 1)

    cluster_0_values = intensity_values_reshaped[cluster_0_mask].flatten()
    cluster_1_values = intensity_values_reshaped[cluster_1_mask].flatten()




    fig, ax = plt.subplots()
    # Create a violin plot for all data points


    # Plot boxplots for both clusters at the specified x-values
    sns.violinplot(y=np.array(intensity_values).flatten(), inner=None, color="lightblue", ax=ax, zorder=1)


        # Plot the boxplots on top
    ax.boxplot([cluster_0_values], positions=[0], widths=0.2, patch_artist=True,
           boxprops=dict(facecolor="seagreen", zorder=2, alpha=.7),
           medianprops=dict(color="black"),
           whiskerprops=dict(color="darkgreen"),
           capprops=dict(color="darkgreen"))

    ax.boxplot([cluster_1_values], positions=[0], widths=0.2, patch_artist=True,
           boxprops=dict(facecolor="lightcoral", zorder=2, alpha=.7),
           medianprops=dict(color="black"),
           whiskerprops=dict(color="firebrick"),
           capprops=dict(color="firebrick"))


    plt.xticks([0],"")
    plt.title(new_name)
    plt.ylabel('NeuN intensity')

    save_path = os.path.join("D:\\Users\\Jonas\\plots", f"{new_name}_neurons.pdf")
    #plt.savefig(save_path)

    plt.show()

    # Count positive and negative clusters
    positive_count = len(positive_labels)
    negative_count = len(negative_labels)
    print(new_name)
    print(f"Number of astrocytes in positive cluster: {positive_count}")
    print(f"Number of astrocytes in negative cluster: {negative_count}")

    return labels

def measure_nuc(image_path, label_path, read_images=False, read_masks = True):

    if not read_images: 
        image = io.imread(image_path)
        print(image_path)
     # Load the label file
    else:
        image = image_path
    
    if not read_masks:
        labels = io.imread(label_path)
    else:
        labels = label_path

    properties = measure.regionprops(labels, intensity_image=image)
    label_properties = []

    for prop in properties:
        region_label = prop.label
        region_area = prop.area
        region_mean_intensity = prop.mean_intensity
        ch1_intensity, ch2_intensity, ch3_intensity, ch4_intensity = region_mean_intensity
        label_properties.append({
            "label": region_label,
            "Area": region_area,
            "Channel1_intensity": ch1_intensity,
            "Channel2_intensity": ch2_intensity,
            "Channel3_intensity": ch3_intensity,
            "Channel4_intensity": ch4_intensity,
        })

    return label_properties

def measure_cyto(image_path, label_path, read_images=False):

    if not read_images: 
        image = io.imread(image_path)
        print(image_path)
     # Load the label file
        labels = io.imread(label_path)
    
    else:
        image = image_path
        labels = label_path

    properties = measure.regionprops(labels, intensity_image=image)
    label_properties = []

    for prop in properties:
        region_label = prop.label
        region_area = prop.area
        region_mean_intensity = prop.mean_intensity
        ch1_intensity, ch2_intensity, ch3_intensity, ch4_intensity = region_mean_intensity
        label_properties.append({
            "label": region_label,
            "Area": region_area,
            "Channel1_intensity": ch1_intensity,
            "Channel2_intensity": ch2_intensity,
            "Channel3_intensity": ch3_intensity,
            "Channel4_intensity": ch4_intensity,
        })

    return label_properties

def measure_g4(properties):

    G4_fluo = []

    for i in range(len(properties)):
        G4= []
        for item in properties[i]:
            G4.append(item["Channel3_intensity"])
        print(np.mean(G4))
        G4_fluo.append(np.mean(G4))

    return G4_fluo

def measure_g4_voxels(mask_path, image_path, g4_channel=2):
    image = io.imread(image_path)
    masks = mask_path #io.imread(mask_path)
    print(image_path)
    print(np.shape(image))
    thresh = filters.threshold_otsu(image[:,:,:,g4_channel])
    print("thresh: ", thresh)
    binary_mask = image[:,:,:,g4_channel] > 80

    # viewer = napari.view_labels(binary_mask, scale=spacing)
    # viewer.add_image(image, name="image", channel_axis=3, scale=spacing)
    # napari.run()
    regions = measure.regionprops(masks)

    proportions = []

    # Iterate over each labeled area
    for region in regions:
        bbox = region.bbox
        label_mask = masks[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]] == region.label

        # Calculate the proportion of foreground pixels in the labeled area
        foreground_pixels = (binary_mask[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]] * label_mask).sum()
        total_pixels = label_mask.sum()
        proportion = foreground_pixels / total_pixels if total_pixels > 0 else 0
        # Append the proportion to the list
        proportions.append(proportion)
    print(len(proportions), np.mean(proportions))
    return proportions

def plot_g4_whole_image(ipsii_properties, contra_properties, sham_properties):
    G4_fluo_ipsii, G4_fluo_contra = measure_g4(ipsii_properties), measure_g4(contra_properties)
    G4_fluo_sham = measure_g4(sham_properties)
    data=[G4_fluo_ipsii, G4_fluo_contra, G4_fluo_sham]

    t_statistic, p_value = stats.ttest_ind(data[0], data[1])
    print("mean contra: ", np.mean(data[1]))
    print("Contra STD: ", stats.tstd(data[1]))
    print("mean ipsii: ", np.mean(data[0]))
    print("Ipsii STD: ", stats.tstd(data[0]))
    print("t-statistic:", t_statistic)
    print("p-value:", p_value)
    xtick_labels = ["Ipsilateral", "Contralateral", "Sham"]
    sns.boxplot(data=data, palette="Set2")
    sns.stripplot(data=data, palette="Set1")
    plt.xticks([0, 1, 2], xtick_labels)
    plt.show()


def load_images_and_masks(image_folder, mask_folder):
    image_files = glob(os.path.join(image_folder, '*.lsm'))

    contra_pairs = []
    ipsii_pairs = []
    contra_names = []
    ipsii_names = []

    for image_path in image_files:
        mask_path = os.path.join(mask_folder, os.path.basename(image_path).replace('.lsm', '_mask.tif'))
        if os.path.exists(mask_path) and all(substring not in image_path for substring in ["bleached"]):
            if 'Contralateral' in image_path:
                contra_pairs.append([image_path, mask_path])
                contra_names.append(image_path)
            elif 'Ipsilateral' in image_path:
                ipsii_pairs.append([image_path, mask_path])
                ipsii_names.append(image_path)
    return ipsii_pairs, contra_pairs, ipsii_names, contra_names

def measure_properties(ipsii_pairs, contra_pairs):    
    contra_properties = []
    ipsii_properties = []

    for i in range(len(contra_pairs)):
        properties = measure_nuc(contra_pairs[i][0], contra_pairs[i][1])
        contra_properties.append(properties)
        print(f"Type of properties: {type(properties)}")

    for i in range(len(ipsii_pairs)):
        properties = measure_nuc(ipsii_pairs[i][0], ipsii_pairs[i][1])
        ipsii_properties.append(properties)

    print(len(ipsii_properties))
    print(len(contra_properties))
    return ipsii_properties, contra_properties

def load_images_masks_and_regionmasks(image_folder, mask_folder, mask_region_folder):
    image_files = glob(os.path.join(image_folder, '*.lsm'))

    contra_pairs = []
    ipsii_pairs = []
    contra_names = []
    ipsii_names = []

    for image_path in image_files:
        mask_path = os.path.join(mask_folder, os.path.basename(image_path).replace('.lsm', '_mask.tif'))
        region_mask_path = os.path.join(mask_region_folder, os.path.basename(image_path).replace('.lsm', '_mask_regions.tif'))

        if os.path.exists(mask_path) and os.path.exists(region_mask_path) and all(substring not in image_path for substring in ['MIX2', 'MIX1']):
            print(image_path)
            if 'CONTRA' in image_path:
                contra_pairs.append([image_path, mask_path, region_mask_path])
                contra_names.append(image_path)
            elif 'IPSII' in image_path:
                ipsii_pairs.append([image_path, mask_path, region_mask_path])
                ipsii_names.append(image_path)
    regions_label = [1, 2, 3]

    contra_properties = []
    ipsii_properties = []

    for i in range(len(ipsii_pairs)):
        mask = io.imread(ipsii_pairs[i][1])
        region_mask = io.imread(ipsii_pairs[i][2])

        # Find bounding boxes for each region label
        labeled_regions = measure.label(region_mask)

        binary_mask = (labeled_regions == regions_label[0])
        ca1 = mask * binary_mask

        binary_mask = (labeled_regions == regions_label[1])
        dg = mask * binary_mask

        binary_mask = (labeled_regions == regions_label[2])
        ca3 = mask * binary_mask

        temp_image = io.imread(ipsii_pairs[i][0])

        ca1_props = measure_nuc(temp_image, ca1, read_images=True)
        ca3_props = measure_nuc(temp_image, ca3, read_images=True)
        dg_props = measure_nuc(temp_image, dg, read_images=True)

        ipsii_properties.append([ca1_props, ca3_props, dg_props])
        #ipsii_region_areas.append(labeled_area)
    print("len contra pairs", len(contra_pairs))
    for i in range(len(contra_pairs)):
        print(i)
        mask = io.imread(contra_pairs[i][1])
        region_mask = io.imread(contra_pairs[i][2])

        # Find bounding boxes for each region label
        labeled_regions = measure.label(region_mask)
        #region_props = measure.regionprops(labeled_regions)
        #labeled_area = [prop.area for prop in region_props]

        # Initialize an empty list for each region
        binary_mask = (labeled_regions == regions_label[0])
        ca1 = mask * binary_mask

        binary_mask = (labeled_regions == regions_label[1])
        dg = mask * binary_mask

        binary_mask = (labeled_regions == regions_label[2])
        ca3 = mask * binary_mask

        temp_image = io.imread(contra_pairs[i][0])

        ca1_props = measure_nuc(temp_image, ca1, read_images=True)
        ca3_props = measure_nuc(temp_image, ca3, read_images=True)
        dg_props = measure_nuc(temp_image, dg, read_images=True)

        contra_properties.append([ca1_props, ca3_props, dg_props])

    return ipsii_properties, contra_properties

def load_images_masks_and_single_ROI(image_folder, mask_folder, mask_region_folder):
    image_files = glob(os.path.join(image_folder, '*.lsm'))

    contra_pairs = []
    ipsii_pairs = []
    contra_names = []
    ipsii_names = []

    for image_path in image_files:
        mask_path = os.path.join(mask_folder, os.path.basename(image_path).replace('.lsm', '_mask.tif'))
        region_mask_path = os.path.join(mask_region_folder, os.path.basename(image_path).replace('.lsm', '_mask_region.tif'))

        if os.path.exists(mask_path) and os.path.exists(region_mask_path): #and all(substring not in image_path for substring in ['MIX2', 'MIX1']):
            print(image_path)
            if 'Contralateral' in image_path:
                contra_pairs.append([image_path, mask_path, region_mask_path])
                contra_names.append(image_path)
            elif 'Ipsilateral' in image_path:
                ipsii_pairs.append([image_path, mask_path, region_mask_path])
                ipsii_names.append(image_path)
    regions_label = [1, 2, 3]

    ipsii_pairs_region = []
    contra_pairs_region = []

    for i in range(len(ipsii_pairs)):
        mask = io.imread(ipsii_pairs[i][1])
        region_mask = io.imread(ipsii_pairs[i][2])

        # Find bounding boxes for each region label
        labeled_regions = measure.label(region_mask)

        binary_mask = (labeled_regions == regions_label[0])
        hippocampus = mask * binary_mask

        # viewer = napari.view_labels(hippocampus, scale=spacing)
        # napari.run()

        ipsii_pairs_region.append([ipsii_pairs[i][0], hippocampus])
    print("len contra pairs", len(contra_pairs))
    for i in range(len(contra_pairs)):
        print(i)
        mask = io.imread(contra_pairs[i][1])
        region_mask = io.imread(contra_pairs[i][2])

        labeled_regions = measure.label(region_mask)
        binary_mask = (labeled_regions == regions_label[0])
        hippocampus = mask * binary_mask

        contra_pairs_region.append([contra_pairs[i][0], hippocampus])

    return ipsii_pairs_region, contra_pairs_region, ipsii_names, contra_names
    
def measure_channel_regions(ipsii_properties, contra_properties):
    ipsii_g4 = []
    contra_g4 = []

    print(len(contra_properties))

    for i in range(len(contra_properties)):
        contra_g4.append(measure_g4(contra_properties[i]))
    for i in range(len(ipsii_properties)):
        ipsii_g4.append(measure_g4(ipsii_properties[i]))
    print(len(contra_g4))
    print(len(ipsii_g4))
    t_statistic, p_value = stats.ttest_ind(ipsii_g4[0], contra_g4[0])
    print(f"CA1 p-val {p_value}")
    t_statistic, p_value = stats.ttest_ind(ipsii_g4[1], contra_g4[1])
    print(f"CA3 p-val {p_value}")
    t_statistic, p_value = stats.ttest_ind(ipsii_g4[2], contra_g4[2])
    print(f"DG p-val {p_value}")

    ipsii_labels = [label for i in range(len(ipsii_g4[0])) for label in [props[i] for props in ipsii_g4]]
    contra_labels = [label for i in range(len(contra_g4[0])) for label in [props[i] for props in contra_g4]]

    colors = ["red", "red", "red", "red", "blue", "blue", "blue", "blue", "cyan", "cyan", "cyan", "cyan"]

    bar_width = 0.35
    index_ipsii = np.arange(len(ipsii_labels))
    index_contra = np.arange(len(contra_labels)) + len(ipsii_labels) + bar_width

    midpoint_ipsii = np.mean(index_ipsii)
    midpoint_contra = np.mean(index_contra)

        # Create bar plots for ipsii_labels
    plt.bar(index_ipsii, ipsii_labels, width=bar_width, color=colors)

        # Create bar plots for contra_labels
    plt.bar(index_contra, contra_labels, width=bar_width, color=colors)

        # Set up labels and title
    plt.xlabel('Regions')
    plt.ylabel('G4 fluorescence')

    # Set the x-axis ticks and labels
    plt.xticks([midpoint_ipsii, midpoint_contra], ["IPSII","CONTRA"])
    plt.grid()

    # Add legend

    legend_labels=["CA1", "CA3", "DG"]
    legend_colors=["red", "blue", "cyan"]

    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color, label=label) for color, label in zip(legend_colors, legend_labels)]
    plt.legend(handles=legend_handles)
    plt.show()

def measure_region_nuclei(ipsii_properties, contra_properties):

    print("ipsii_properties", len(ipsii_properties[0]))
    ipsii_labels = [[len(region_props) for region_props in props] for props in ipsii_properties]
    contra_labels = [[len(region_props) for region_props in props] for props in contra_properties]

    print("Means Ca1: ", np.mean([label[0]for label in ipsii_labels]), np.mean([label[0]for label in contra_labels]))
    print("Means Ca3: ", np.mean([label[1]for label in ipsii_labels]), np.mean([label[1]for label in contra_labels]))
    print("Means DG ", np.mean([label[2]for label in ipsii_labels]), np.mean([label[2]for label in contra_labels]))
    t_statistic, p_value = stats.ttest_ind([label[0]for label in ipsii_labels], [label[0]for label in contra_labels])
    print(f"CA1 p-val {p_value}")
    t_statistic, p_value = stats.ttest_ind([label[1]for label in ipsii_labels], [label[1]for label in contra_labels])
    print(f"CA3 p-val {p_value}")
    t_statistic, p_value = stats.ttest_ind([label[2]for label in ipsii_labels], [label[2]for label in contra_labels])
    print(f"DG p-val {p_value}")

    ipsii_total = [sum([label for label in ipsii_labels[0]]),sum([label for label in ipsii_labels[1]]),sum([label for label in ipsii_labels[2]]),sum([label for label in ipsii_labels[3]])]
    contra_total = [sum([label for label in contra_labels[0]]),sum([label for label in contra_labels[1]]),sum([label for label in contra_labels[2]]),sum([label for label in contra_labels[3]])]
    print("ipsii total: ", ipsii_total)
    print("contra total: ", contra_total)

    t_statistic, p_value = stats.ttest_ind(ipsii_total, contra_total)
    print(f"Total p-val {p_value}")
    print(f"Total t-stat: {t_statistic}")

    ipsii_labels = [label for i in range(len(ipsii_labels[0])) for label in [props[i] for props in ipsii_labels]]
    contra_labels = [label for i in range(len(contra_labels[0])) for label in [props[i] for props in contra_labels]]


    colors = ["red", "red", "red", "red", "blue", "blue", "blue", "blue", "cyan", "cyan", "cyan", "cyan"]

    print(ipsii_labels)
    print(contra_labels)

    # Set up bar positions and width
    bar_width = 0.35
    index_ipsii = np.arange(len(ipsii_labels))
    index_contra = np.arange(len(contra_labels)) + len(ipsii_labels) + bar_width

    midpoint_ipsii = np.mean(index_ipsii)
    midpoint_contra = np.mean(index_contra)

    # Create bar plots for ipsii_labels
    plt.bar(index_ipsii, ipsii_labels, width=bar_width, color=colors)

    # Create bar plots for contra_labels
    plt.bar(index_contra, contra_labels, width=bar_width, color=colors)

    # Set up labels and title
    plt.xlabel('Regions')
    plt.ylabel('Nuclei')

    # Set the x-axis ticks and labels
    plt.xticks([midpoint_ipsii, midpoint_contra], ["IPSII","CONTRA"])
    plt.grid()


    legend_labels=["CA1", "CA3", "DG"]
    legend_colors=["red", "blue", "cyan"]

    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color, label=label) for color, label in zip(legend_colors, legend_labels)]
    plt.legend(handles=legend_handles)
    plt.show()


