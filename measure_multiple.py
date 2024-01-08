from matplotlib import pyplot as plt
from scipy import stats
import numpy as np
from glob import glob
import os
import seaborn as sns
from skimage import io, measure
from sklearn.cluster import KMeans
import re


def classify_neurons(properties, name, num_clusters=2, savefig=False):

    pattern = re.compile(r'images\\(.*?)\s+G4')

    # Use the regular expression to extract the desired part of the file name
    match = pattern.search(name)
    if match:
        new_name = match.group(1)

        # You can use the new_name or return it as needed
    else:
        # Handle the case where the input name doesn't match the expected pattern
        print(f'Invalid name format: {name}')
        new_name = name
        
    
    # Extract Channel2_intensity values from properties
    channel2_intensity_values = [prop["Channel2_intensity"] for prop in properties]

    # Convert the list to a NumPy array
    data_array = np.array(channel2_intensity_values).reshape(-1, 1)

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto")
    labels = kmeans.fit_predict(data_array)

    cluster_means = [np.mean(data_array[labels == i]) for i in range(num_clusters)]
    negative_cluster = np.argmin(cluster_means)

    cluster_0_values = data_array[labels == negative_cluster].flatten()
    cluster_1_values = data_array[labels != negative_cluster].flatten()


    fig, ax = plt.subplots()
    # Create a violin plot for all data points


    # Plot boxplots for both clusters at the specified x-values
    sns.violinplot(y=data_array.flatten(), inner=None, color="lightblue", ax=ax, zorder=1)

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



    plt.title(new_name)
    plt.ylabel('NeuN intensity')

    save_path = os.path.join("D:\\Users\\Jonas\\plots", f"{new_name}_astrocytes.pdf")
    #plt.savefig(save_path)

    plt.show()

    # Count positive and negative clusters
    positive_count = np.sum(labels == 1)
    negative_count = np.sum(labels == 0)
    print(new_name)
    print(f"Number of astrocytes in positive cluster: {positive_count}")
    print(f"Number of astrocytes in negative cluster: {negative_count}")

    return labels

def measure_nuc(image_path, label_path, read_images=False):

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

    print("Type: ", type(properties[0]))

    for i in range(len(properties)):
        G4= []
        for item in properties[i]:
            G4.append(item["Channel3_intensity"])
        G4_fluo.append(np.mean(G4))

    return G4_fluo

def plot_g4_whole_image(ipsii_properties, contra_properties):
    G4_fluo_ipsii, G4_fluo_contra = measure_g4(ipsii_properties), measure_g4(contra_properties)
    
    data=[G4_fluo_ipsii, G4_fluo_contra]

    t_statistic, p_value = stats.ttest_ind(data[0], data[1])

    print("t-statistic:", t_statistic)
    print("p-value:", p_value)
    xtick_labels = ["IPSII", "CONTRA"]
    sns.boxplot(data=data, palette="Set2")
    sns.stripplot(data=data, palette="Set1")
    plt.xticks([0, 1], xtick_labels)
    plt.show()


def load_images_and_masks(image_folder, mask_folder):
    image_files = glob(os.path.join(image_folder, '*.lsm'))

    contra_pairs = []
    ipsii_pairs = []
    contra_names = []
    ipsii_names = []

    for image_path in image_files:
        mask_path = os.path.join(mask_folder, os.path.basename(image_path).replace('.lsm', '_mask.tif'))
        if os.path.exists(mask_path) and all(substring not in image_path for substring in ['MIX2','MIX1', 'MIX3']):
            if 'CONTRA' in image_path:
                contra_pairs.append([image_path, mask_path])
                contra_names.append(image_path)
            elif 'IPSII' in image_path:
                ipsii_pairs.append([image_path, mask_path])
                ipsii_names.append(image_path)

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
    return ipsii_properties, contra_properties, ipsii_names, contra_names

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


