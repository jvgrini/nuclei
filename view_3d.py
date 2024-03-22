import napari
from skimage import io
import czifile
import numpy as np
import tifffile
from utils import readImage

# img = czifile.imread("imagesAndMasks\liv\mec/170123_A1_363_NeuN-GFP_S1_MEC.czi")
# img = np.squeeze(img)
# img = np.transpose(img, (1,2,3,0))

img = readImage('imagesAndMasks/liv/mec/170123_A1_363_NeuN-GFP_S1_MEC.czi')
mask = io.imread("imagesAndMasks/liv/masks/170123_A1_363_NeuN-GFP_S1_MEC_mask.tif")
spacing = ([0.9278349, 0.3459441, 0.3459441])
#spacing = [1,1,1]
#roi = io.imread("imagesAndMasks/region_masks_extended\HI 1 Contralateral Mouse 8 Slide18 G4green NeuNpink CD86red 40x 4x4 technical replica 2_mask_region.tif")
print(img.shape)
print(type(img))


viewer = napari.view_image(
    img,
    channel_axis=3,
    scale = spacing,
    ndisplay=2
)
viewer.add_labels(mask, scale=spacing)
#viewer.add_labels(roi, scale=spacing)
napari.run()


# P9 10 MIX4 HI IPSII G4green IBA1pink CD68red 3x4 40x 1_mask_regions .tif
# P9 10 MIX4 HI IPSII G4green IBA1pink CD68red 3x4 40x 1_mask.tif

#masks_regions_extended\P9 10 MIX4 HI IPSII G4green IBA1pink CD68red 3x4 40x 1_mask_regions.tif
#masks_regions_extended\P9 10 MIX4 HI IPSII G4green IBA1pink CD68red 3x4 40x 1_mask_regions .tif
#masks\P9 10 MIX4 HI IPSII G4green IBA1pink CD68red 3x4 40x 1_mask.tif 
#images\P9 10 MIX4 HI IPSII G4green IBA1pink CD68red 3x4 40x 1.lsm
