import napari
from skimage import io
import czifile
import numpy as np

img = czifile.imread("imagesAndMasks/angus/291123_B2_766cntr_NeuN-Arc-GFP_S1_HPC.czi")
mask = io.imread("imagesAndMasks\masks_angus/291123_B2_766cntr_NeuN-Arc-GFP_S1_HPC_mask.tif")
spacing = ([0.9278, 0.3459, 0.3459])

img = np.squeeze(img)
img = np.transpose(img, (1,2,3,0))
viewer = napari.view_image(
    img,
    channel_axis=3,
    scale = spacing,
    ndisplay=2
)
viewer.add_labels(mask, scale=spacing)
napari.run()


# P9 10 MIX4 HI IPSII G4green IBA1pink CD68red 3x4 40x 1_mask_regions .tif
# P9 10 MIX4 HI IPSII G4green IBA1pink CD68red 3x4 40x 1_mask.tif

#masks_regions_extended\P9 10 MIX4 HI IPSII G4green IBA1pink CD68red 3x4 40x 1_mask_regions.tif
#masks_regions_extended\P9 10 MIX4 HI IPSII G4green IBA1pink CD68red 3x4 40x 1_mask_regions .tif
#masks\P9 10 MIX4 HI IPSII G4green IBA1pink CD68red 3x4 40x 1_mask.tif 
#images\P9 10 MIX4 HI IPSII G4green IBA1pink CD68red 3x4 40x 1.lsm