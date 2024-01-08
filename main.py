import measure_multiple as mp

image_folder = 'images'
mask_folder = 'masks'
mask_region_folder = "masks_regions_extended"



#ipsii_properties, contra_properties, ipsii_names, contra_names = mp.load_images_and_masks(image_folder, mask_folder)

#for i in range(len(ipsii_properties)):
#    mp.classify_neurons(ipsii_properties[i], ipsii_names[i])
#for i in range(len(contra_properties)):
#    mp.classify_neurons(contra_properties[i], contra_names[i])

#mp.plot_g4_whole_image(ipsii_properties, contra_properties)

ipsii_properties, contra_properties = mp.load_images_masks_and_regionmasks(image_folder, mask_folder, mask_region_folder)
mp.measure_channel_regions(ipsii_properties, contra_properties)
mp.measure_region_nuclei(ipsii_properties, contra_properties)