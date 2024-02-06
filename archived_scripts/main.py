import measure_multiple as mp
import numpy as np
from skimage import io
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import napari

image_folder_HI = 'images_HI'
mask_folder_HI = 'masks_HI'
mask_region_folder = "region_masks_extended"
image_folder_sham = "images_sham"
mask_folder_sham = "masks_sham"



ipsii_pairs_HI, contra_pairs_HI, ipsii_names_HI, contra_names_HI = mp.load_images_masks_and_single_ROI_2(image_folder_HI, mask_folder_HI, mask_region_folder)
ipsii_pairs_sham, contra_pairs_sham, ipsii_names_sham, contra_names_sham = mp.load_images_masks_and_single_ROI_2(image_folder_sham, mask_folder_sham, mask_region_folder)
contra_properties_HI = []
ipsii_properties_HI = []
contra_properties_sham = []
ipsii_properties_sham = []
ipsii_properties, contra_properties, ipsii_names, contra_names = mp.measure_properties(ipsii_pairs_HI, contra_pairs_HI, ipsii_names_HI, contra_names_HI)

ipsii_properties_sham, contra_properties_sham, ipsii_names_sham, contra_names_sham = mp.measure_properties(ipsii_pairs_sham, contra_pairs_sham, ipsii_names_sham, contra_names_sham)

contra_labels = []
ipsii_labels = []
contra_labels_sham = []
ipsii_labels_sham = []

for i in range(len(ipsii_properties)):
    print(i)
    ipsii_labels.append(mp.classify_neurons2(ipsii_pairs_HI[i][1],ipsii_properties[i], ipsii_names[i], region_label=ipsii_pairs_HI[i][2]))
for i in range(len(contra_properties)):
    contra_labels.append(mp.classify_neurons2(contra_pairs_HI[i][1],contra_properties[i], contra_names[i], region_label=contra_pairs_HI[i][2]))

for i in range(len(ipsii_properties_sham)):
    print(i)
    ipsii_labels_sham.append(mp.classify_neurons2(ipsii_pairs_sham[i][1],ipsii_properties_sham[i], ipsii_names_sham[i], region_label=ipsii_pairs_sham[i][2]))
for i in range(len(contra_properties_sham)):
    contra_labels_sham.append(mp.classify_neurons2(contra_pairs_sham[i][1],contra_properties_sham[i], contra_names_sham[i], region_label=contra_pairs_sham[i][2]))

spacing = ([0.9278, 0.3459, 0.3459])
viewer = napari.view_image(
    io.imread(contra_pairs_HI[0][0]), 
    scale=spacing,
    ndisplay=2,
    channel_axis=3
                  )
for i in range(len(contra_labels[0])):
    viewer.add_labels(contra_labels[0][i], name=f'Cluster {i + 1}', scale=spacing)
napari.run()


properties_mature_contra = []
properties_immature_contra = []
properties_non_neuron_contra = []

properties_mature_ipsi = []
properties_immature_ipsi = []
properties_non_neuron_ipsi = []

properties_mature_sham = []
properties_immature_sham = []
properties_non_neuron_sham = []

properties_mature_sham_ipsi = []
properties_immature_sham_ipsi = []
properties_non_neuron_sham_ipsi = []

properties_mature_sham_contra = []
properties_immature_sham_contra = []
properties_non_neuron_sham_contra = []

for i in range(len(contra_labels_sham)):
    properties_mature_sham.append(mp.measure_nuc(contra_pairs_sham[i][0], contra_labels_sham[i][2], read_masks=True))
    properties_immature_sham.append(mp.measure_nuc(contra_pairs_sham[i][0], contra_labels_sham[i][1], read_masks=True))
    properties_non_neuron_sham.append(mp.measure_nuc(contra_pairs_sham[i][0], contra_labels_sham[i][0], read_masks=True))

for i in range(len(ipsii_labels_sham)):
    properties_mature_sham.append(mp.measure_nuc(ipsii_pairs_sham[i][0], ipsii_labels_sham[i][2], read_masks=True))
    properties_immature_sham.append(mp.measure_nuc(ipsii_pairs_sham[i][0], ipsii_labels_sham[i][1], read_masks=True))
    properties_non_neuron_sham.append(mp.measure_nuc(ipsii_pairs_sham[i][0], ipsii_labels_sham[i][0], read_masks=True))

properties_neurons_sham = [a + b for a, b, in zip(properties_mature_sham, properties_immature_sham)]


print("se her!")
print(len(properties_non_neuron_sham))
print(len(properties_neurons_sham))

g4_mature_sham = mp.measure_g4(properties_mature_sham)
g4_immature_sham = mp.measure_g4(properties_immature_sham)
g4_non_neuron_sham = mp.measure_g4(properties_non_neuron_sham)
g4_neuron_sham = mp.measure_g4(properties_neurons_sham)

print(g4_mature_sham, g4_immature_sham, g4_non_neuron_sham)

for i in range(len(contra_labels)):
    properties_mature_contra.append(mp.measure_nuc(contra_pairs_HI[i][0], contra_labels[i][2], read_masks=True))
    properties_immature_contra.append(mp.measure_nuc(contra_pairs_HI[i][0], contra_labels[i][1], read_masks=True))
    properties_non_neuron_contra.append(mp.measure_nuc(contra_pairs_HI[i][0], contra_labels[i][0], read_masks=True))

properties_neurons_contra = [a + b for a, b, in zip(properties_mature_contra, properties_immature_contra)]

g4_mature_contra = mp.measure_g4(properties_mature_contra)
g4_immature_contra = mp.measure_g4(properties_immature_contra)
g4_non_neurons_contra = mp.measure_g4(properties_non_neuron_contra)
g4_neurons_contra = mp.measure_g4(properties_neurons_contra)

print(g4_mature_contra, g4_immature_contra, g4_non_neurons_contra)

for i in range(len(ipsii_labels)):
    properties_mature_ipsi.append(mp.measure_nuc(ipsii_pairs_HI[i][0], ipsii_labels[i][2], read_masks=True))
    properties_immature_ipsi.append(mp.measure_nuc(ipsii_pairs_HI[i][0], ipsii_labels[i][1], read_masks=True))
    properties_non_neuron_ipsi.append(mp.measure_nuc(ipsii_pairs_HI[i][0], ipsii_labels[i][0], read_masks=True))

properties_neurons_ipsi = [a + b for a, b, in zip(properties_mature_ipsi, properties_immature_ipsi)]

g4_mature_ipsi = mp.measure_g4(properties_mature_ipsi)
g4_immature_ipsi = mp.measure_g4(properties_immature_ipsi)
g4_non_neurons_ipsi = mp.measure_g4(properties_non_neuron_ipsi)
g4_neurons_ipsi = mp.measure_g4(properties_neurons_ipsi)

print(g4_mature_ipsi, g4_immature_ipsi, g4_non_neurons_ipsi)

max_length = max(len(g4_mature_sham), len(g4_immature_sham), len(g4_non_neuron_sham),
                  len(g4_mature_contra), len(g4_immature_contra), len(g4_non_neurons_contra),
                  len(g4_mature_ipsi), len(g4_immature_ipsi), len(g4_non_neurons_ipsi))



df = pd.DataFrame({
    'Mature Sham': pd.Series(g4_mature_sham).reindex(range(max_length)),
    'Immature Sham': pd.Series(g4_immature_sham).reindex(range(max_length)),
    'Non-Neurons Sham': pd.Series(g4_non_neuron_sham).reindex(range(max_length)),
    'Mature Contra': pd.Series(g4_mature_contra).reindex(range(max_length)),
    'Immature Contra': pd.Series(g4_immature_contra).reindex(range(max_length)),
    'Non-Neurons Contra': pd.Series(g4_non_neurons_contra).reindex(range(max_length)),
    'Mature Ipsi': pd.Series(g4_mature_ipsi).reindex(range(max_length)),
    'Immature Ipsi': pd.Series(g4_immature_ipsi).reindex(range(max_length)),
    'Non-Neurons Ipsi': pd.Series(g4_non_neurons_ipsi).reindex(range(max_length)),
})

df2 = pd.DataFrame({
    'Neurons Sham': pd.Series(g4_neuron_sham).reindex(range(max_length)),
    'Non-Neurons Sham': pd.Series(g4_non_neuron_sham).reindex(range(max_length)),
    'Neurons Contra': pd.Series(g4_neurons_contra).reindex(range(max_length)),
    'Non-Neurons Contra': pd.Series(g4_non_neurons_contra).reindex(range(max_length)),
    'Neurons Ipsi': pd.Series(g4_neurons_ipsi).reindex(range(max_length)),
    'Non-Neurons Ipsi': pd.Series(g4_non_neurons_ipsi).reindex(range(max_length)),
})

# Create a figure and axis
plt.figure(figsize=(12, 8))

# Create a boxplot
sns.boxplot(data=df2, palette="Set3")

# Create a stripplot
sns.stripplot(data=df2, color="black", jitter=True, size=4)

# Set plot labels and title
plt.xticks(rotation=45, ha="right")
plt.ylabel('G4 fluorescence (0-255)')
plt.tight_layout()
# Show the plot
plt.show()

# for i in range(len(ipsii_pairs_HI)):
#     print(ipsii_pairs_HI[i][1],ipsii_pairs_HI[i][0])
#     ipsii_properties_HI.append(np.mean(mp.measure_g4_voxels(ipsii_pairs_HI[i][1],ipsii_pairs_HI[i][0])))
# for i in range(len(contra_pairs_HI)):
#    contra_properties_HI.append(np.mean(mp.measure_g4_voxels(contra_pairs_HI[i][1],contra_pairs_HI[i][0])))

# for i in range(len(ipsii_pairs_sham)):
#     print(ipsii_pairs_sham[i][1],ipsii_pairs_sham[i][0])
#     ipsii_properties_sham.append(np.mean(mp.measure_g4_voxels(ipsii_pairs_sham[i][1],ipsii_pairs_sham[i][0])))
# for i in range(len(contra_pairs_sham)):
#    contra_properties_sham.append(np.mean(mp.measure_g4_voxels(contra_pairs_sham[i][1],contra_pairs_sham[i][0])))

# sham_properties = contra_properties_sham + ipsii_properties_sham
# print("contra mean: ", np.mean(contra_properties_HI))
# print("ipsii mean: ", np.mean(ipsii_properties_HI))
# print("Contra STD: ", stats.tstd(contra_properties_HI))
# print("ipsii STD: ", stats.tstd(ipsii_properties_HI))

# tstat, pval = stats.ttest_ind(ipsii_properties_HI, contra_properties_HI)

# print("pval: ", pval)

# df = pd.DataFrame({
#     'Group': ['Ipsilateral'] * len(ipsii_properties_HI) + ['Contralateral'] * len(contra_properties_HI) + ["Sham"] * len(sham_properties),
#     'Proportion': np.concatenate([ipsii_properties_HI, contra_properties_HI, sham_properties])
# })
# plt.figure(figsize=(10, 6))
# sns.boxplot(x='Group', y='Proportion', data=df)
# sns.stripplot(x='Group', y='Proportion', data=df, palette='Set1')
# plt.title('Proportion of Foreground Pixels in Labeled Areas')
# plt.show()


# print("sham")
# print("contra mean: ", np.mean(contra_properties_sham))
# print("ipsii mean: ", np.mean(ipsii_properties_sham))
# print("Contra STD: ", stats.tstd(contra_properties_sham))
# print("ipsii STD: ", stats.tstd(ipsii_properties_sham))

# tstat, pval = stats.ttest_ind(ipsii_properties_sham, contra_properties_sham)

# print("pval: ", pval)

# df = pd.DataFrame({
#     'Group': ['Ipsilateral'] * len(ipsii_properties_sham) + ['Contralateral'] * len(contra_properties_sham),
#     'Proportion': np.concatenate([ipsii_properties_sham, contra_properties_sham])
# })
# plt.figure(figsize=(10, 6))
# sns.boxplot(x='Group', y='Proportion', data=df)
# sns.stripplot(x='Group', y='Proportion', data=df, palette='Set1')
# plt.title('Proportion of Foreground Pixels in Labeled Areas')
# plt.show()

# ipsii_properties, contra_properties, ipsii_names, contra_names = mp.measure_properties(ipsii_pairs_HI, contra_pairs_HI, ipsii_names_HI, contra_names_HI)
# ipsii_properties_sham, contra_properties_sham, ipsii_names_sham, contra_names_sham= mp.measure_properties(ipsii_pairs_sham, contra_pairs_sham, ipsii_names_sham, contra_names_sham)
# sham_properties = contra_properties_sham + ipsii_properties_sham
# sham_names = contra_names_sham + ipsii_names_sham

# print(len(ipsii_properties))

# for i in range(len(ipsii_properties)):
#     g4 = []
#     for item in ipsii_properties[i]:
#         g4.append(item["Channel3_intensity"])
#     print(f"Name: {ipsii_names[i]}, mean G4: {np.mean(g4)}")
# for i in range(len(contra_properties)):
#     g4 = []
#     for item in contra_properties[i]:
#         g4.append(item["Channel3_intensity"])
#     print(f"Name: {contra_names[i]}, mean G4: {np.mean(g4)}")
# for i in range(len(sham_properties)):
#     g4 = []
#     for item in sham_properties[i]:
#         g4.append(item["Channel3_intensity"])
#     print(f"Name: {sham_names[i]}, mean G4: {np.mean(g4)}")


# contra_proportions = []
# ipsii_proportions=[]
# contra_values = []
# ipsii_values = []

# for i in range(len(ipsii_properties)):
#     ipsii_values.append(mp.classify_neurons2(ipsii_pairs[i][1],ipsii_properties[i], ipsii_names[i]))
# for i in range(len(contra_properties)):
#     contra_values.append(mp.classify_neurons2(contra_pairs[i][1],contra_properties[i], contra_names[i]))


# for item in contra_values:
#     print(item)
#     non_neurons, neurons1, neurons2 = item[0], item[1], item[2]

#     total = non_neurons + neurons1 + neurons2
#     proportion = (neurons1+neurons2)/total
#     contra_proportions.append(proportion)
# for item in ipsii_values:
#     non_neurons, neurons1, neurons2 = item[0], item[1], item[2]

#     total = non_neurons + neurons1 + neurons2
#     proportion = (neurons1+neurons2)/total
#     ipsii_proportions.append(proportion)
# print(ipsii_proportions)
# print(contra_proportions)
# indices_ipsii = np.arange(len(ipsii_proportions))
# indices_contra = np.arange(len(contra_proportions))

# tstat, pval = stats.ttest_ind(contra_proportions, ipsii_proportions)

# print(f"Mean contra: {np.mean(contra_proportions)}, mean ipsii: {np.mean(ipsii_proportions)}")
# print(f"pval: {pval}")
# print(f"t-stat: {tstat}")

# # Plotting ipsii_proportions
# plt.bar(indices_ipsii, ipsii_proportions, width=0.4, label='Ipsii', color='blue', alpha=0.7)

# # Plotting contra_proportions after ipsii_proportions
# plt.bar(indices_contra + len(ipsii_proportions), contra_proportions, width=0.4, label='Contra', color='red', alpha=0.7)

# # Customize the plot
# plt.ylabel('Proportion')
# plt.title('Proportions of Neurons in Ipsii and Contra')
# plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
# plt.tight_layout()
# # Show the plot
# plt.show()

# mp.plot_g4_whole_image(ipsii_properties, contra_properties, sham_properties)
# mp.plot_g4_whole_image(ipsii_properties_sham, contra_properties_sham, sham_properties)

# ipsii_properties, contra_properties = mp.load_images_masks_and_regionmasks(image_folder, mask_folder, mask_region_folder)
# mp.measure_channel_regions(ipsii_properties, contra_properties)
# mp.measure_region_nuclei(ipsii_properties, contra_properties)