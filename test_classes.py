from utils import match_images_and_masks
from image import Image
import os

image_folder = 'Images'
mask_folder = 'masks'
roi_folder = 'masks_regions_extended'

images = match_images_and_masks(image_folder, mask_folder, roi_folder)

image_objects = []
for image_info in images:
    name = os.path.basename(image_info[0])  
    image_obj = Image(name, image_info[0], image_info[1], image_info[2])
    image_objects.append(image_obj)

cluster_values = []
for object in image_objects:
    print(object.name, len(object.nuclei))
    print(object.getMeanFluorescenceChannel(channel=1))
    object.clusterMasks = object.classifyCells(inspect_classified_masks=True, plot_selectionChannel=False)
    object.measureNucleiInRegion(object.roi, object.clusterMasks)
    print('Non neurons: ',len(object.clusterNuclei[0]))
    print('Immature neurons: ',len(object.clusterNuclei[1]))
    print('Mature neurons: ',len(object.clusterNuclei[2]))
    