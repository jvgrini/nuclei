import measure_multiple as mp
import numpy as np
from skimage import io
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

image_folder_HI = 'images_HI'
mask_folder_HI = 'masks_HI'
mask_region_folder = "region_masks_extended"
image_folder_sham = "images_sham"
mask_folder_sham = "masks_sham"



ipsii_pairs_HI, contra_pairs_HI, ipsii_names_HI, contra_names_HI = mp.load_images_masks_and_single_ROI(image_folder_HI, mask_folder_HI, mask_region_folder)
ipsii_pairs_sham, contra_pairs_sham, ipsii_names_sham, contra_names_sham = mp.load_images_masks_and_single_ROI(image_folder_sham, mask_folder_sham, mask_region_folder)
contra_properties_HI = []
ipsii_properties_HI = []
contra_properties_sham = []
ipsii_properties_sham = []
for i in range(len(ipsii_pairs_HI)):
    print(ipsii_pairs_HI[i][1],ipsii_pairs_HI[i][0])
    ipsii_properties_HI.append(np.mean(mp.measure_g4_voxels(ipsii_pairs_HI[i][1],ipsii_pairs_HI[i][0])))
for i in range(len(contra_pairs_HI)):
   contra_properties_HI.append(np.mean(mp.measure_g4_voxels(contra_pairs_HI[i][1],contra_pairs_HI[i][0])))

for i in range(len(ipsii_pairs_sham)):
    print(ipsii_pairs_sham[i][1],ipsii_pairs_sham[i][0])
    ipsii_properties_sham.append(np.mean(mp.measure_g4_voxels(ipsii_pairs_sham[i][1],ipsii_pairs_sham[i][0])))
for i in range(len(contra_pairs_sham)):
   contra_properties_sham.append(np.mean(mp.measure_g4_voxels(contra_pairs_sham[i][1],contra_pairs_sham[i][0])))

sham_properties = contra_properties_sham + ipsii_properties_sham
print("contra mean: ", np.mean(contra_properties_HI))
print("ipsii mean: ", np.mean(ipsii_properties_HI))
print("Contra STD: ", stats.tstd(contra_properties_HI))
print("ipsii STD: ", stats.tstd(ipsii_properties_HI))

tstat, pval = stats.ttest_ind(ipsii_properties_HI, contra_properties_HI)

print("pval: ", pval)

df = pd.DataFrame({
    'Group': ['Ipsilateral'] * len(ipsii_properties_HI) + ['Contralateral'] * len(contra_properties_HI) + ["Sham"] * len(sham_properties),
    'Proportion': np.concatenate([ipsii_properties_HI, contra_properties_HI, sham_properties])
})
plt.figure(figsize=(10, 6))
sns.boxplot(x='Group', y='Proportion', data=df)
sns.stripplot(x='Group', y='Proportion', data=df, palette='Set1')
plt.title('Proportion of Foreground Pixels in Labeled Areas')
plt.show()


print("sham")
print("contra mean: ", np.mean(contra_properties_sham))
print("ipsii mean: ", np.mean(ipsii_properties_sham))
print("Contra STD: ", stats.tstd(contra_properties_sham))
print("ipsii STD: ", stats.tstd(ipsii_properties_sham))

tstat, pval = stats.ttest_ind(ipsii_properties_sham, contra_properties_sham)

print("pval: ", pval)

df = pd.DataFrame({
    'Group': ['Ipsilateral'] * len(ipsii_properties_sham) + ['Contralateral'] * len(contra_properties_sham),
    'Proportion': np.concatenate([ipsii_properties_sham, contra_properties_sham])
})
plt.figure(figsize=(10, 6))
sns.boxplot(x='Group', y='Proportion', data=df)
sns.stripplot(x='Group', y='Proportion', data=df, palette='Set1')
plt.title('Proportion of Foreground Pixels in Labeled Areas')
plt.show()

ipsii_properties, contra_properties = mp.measure_properties(ipsii_pairs_HI, contra_pairs_HI)
ipsii_properties_sham, contra_properties_sham = mp.measure_properties(ipsii_pairs_sham, contra_pairs_sham)
sham_properties = contra_properties_sham + ipsii_properties_sham

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

mp.plot_g4_whole_image(ipsii_properties, contra_properties, sham_properties)
mp.plot_g4_whole_image(ipsii_properties_sham, contra_properties_sham, sham_properties)

# ipsii_properties, contra_properties = mp.load_images_masks_and_regionmasks(image_folder, mask_folder, mask_region_folder)
# mp.measure_channel_regions(ipsii_properties, contra_properties)
# mp.measure_region_nuclei(ipsii_properties, contra_properties)