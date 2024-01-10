import measure_multiple as mp
import numpy as np
from skimage import io
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

image_folder = 'Images'
mask_folder = 'masks'
mask_region_folder = "masks_regions_extended"



ipsii_pairs, contra_pairs, ipsii_names, contra_names = mp.load_images_and_masks(image_folder, mask_folder)

contra_properties = []
ipsii_properties = []
for i in range(len(ipsii_pairs)):
    ipsii_properties.append(np.mean(mp.measure_g4_voxels(ipsii_pairs[i][1],ipsii_pairs[i][0])))
for i in range(len(contra_pairs)):
   contra_properties.append(np.mean(mp.measure_g4_voxels(contra_pairs[i][1],contra_pairs[i][0])))

print("contra mean: ", np.mean(contra_properties))
print("ipsii mean: ", np.mean(ipsii_properties))

tstat, pval = stats.ttest_ind(ipsii_properties, contra_properties)

print("pval: ", pval)

df = pd.DataFrame({
    'Group': ['Ipsilateral'] * len(ipsii_properties) + ['Contralateral'] * len(contra_properties),
    'Proportion': np.concatenate([ipsii_properties, contra_properties])
})
plt.figure(figsize=(10, 6))
sns.boxplot(x='Group', y='Proportion', data=df)
plt.title('Proportion of Foreground Pixels in Labeled Areas')
plt.show()
# ipsii_properties, contra_properties = mp.measure_properties(ipsii_pairs, contra_pairs)

# # for i in range(len(ipsii_properties)):
# #     mp.classify_neurons(ipsii_pairs[i][1],ipsii_properties[i], ipsii_names[i])
# # for i in range(len(contra_properties)):
# #     mp.classify_neurons(contra_pairs[i][1],contra_properties[i], contra_names[i])

# mp.plot_g4_whole_image(ipsii_properties, contra_properties)

# ipsii_properties, contra_properties = mp.load_images_masks_and_regionmasks(image_folder, mask_folder, mask_region_folder)
# mp.measure_channel_regions(ipsii_properties, contra_properties)
# mp.measure_region_nuclei(ipsii_properties, contra_properties)