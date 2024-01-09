import measure_multiple as mp
import numpy as np
from skimage import io

image_folder = 'Images'
mask_folder = 'masks'
mask_region_folder = "masks_regions_extended"



ipsii_pairs, contra_pairs, ipsii_names, contra_names = mp.load_images_and_masks(image_folder, mask_folder)

ipsii_properties, contra_properties = mp.measure_properties(ipsii_pairs, contra_pairs)

for i in range(len(ipsii_properties)):
    mp.classify_neurons(ipsii_pairs[i][1],ipsii_properties[i], ipsii_names[i])
for i in range(len(contra_properties)):
    mp.classify_neurons(contra_pairs[i][1],contra_properties[i], contra_names[i])

#mp.plot_g4_whole_image(ipsii_properties, contra_properties)

#ipsii_properties, contra_properties = mp.load_images_masks_and_regionmasks(image_folder, mask_folder, mask_region_folder)
#mp.measure_channel_regions(ipsii_properties, contra_properties)
#mp.measure_region_nuclei(ipsii_properties, contra_properties)