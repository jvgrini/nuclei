import napari
from skimage import io
import czifile
import numpy as np

# img = czifile.imread("imagesAndMasks\liv\mec/231122_A1_241_NeuN-GFP_S1_MEC.czi")
# img = np.squeeze(img)
# img = np.transpose(img, (1,2,3,0))

img = io.imread('imagesAndMasks\images_sham\Sham 1 Contralateral Mouse 6 Slide15 G4green NeuNpink CD86red 40x 5x4 1 one tile of G4 channel bleached.lsm')
mask = io.imread("imagesAndMasks\masks_sham\Sham 1 Contralateral Mouse 6 Slide15 G4green NeuNpink CD86red 40x 5x4 1 one tile of G4 channel bleached_mask.tif")
spacing = ([0.9278, 0.3459, 0.3459])
#roi = io.imread("imagesAndMasks/brain_region_masks_extended\Sham 1 Ipsilateral Mouse 6 Slide15 G4green NeuNpink CD86red 40x 5x4 1_regions_mask.tif")
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