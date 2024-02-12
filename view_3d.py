import napari
from skimage import io


img = io.imread("angus/081223_A1_772ko_NeuN-Cfos-GFP_S1_HPC.lsm")
#mask = io.imread("brain_region_masks\HI 1 Contralateral Mouse 8 Slide18 G4green NeuNpink CD86red 40x 4x4 technical replica1_regions_mask.tif")
spacing = ([0.9278, 0.3459, 0.3459])
viewer = napari.view_image(
    img,
    channel_axis=3,
    scale = spacing,
    ndisplay=2
)
#viewer.add_labels(mask, scale=spacing)
napari.run()


# P9 10 MIX4 HI IPSII G4green IBA1pink CD68red 3x4 40x 1_mask_regions .tif
# P9 10 MIX4 HI IPSII G4green IBA1pink CD68red 3x4 40x 1_mask.tif

#masks_regions_extended\P9 10 MIX4 HI IPSII G4green IBA1pink CD68red 3x4 40x 1_mask_regions.tif
#masks_regions_extended\P9 10 MIX4 HI IPSII G4green IBA1pink CD68red 3x4 40x 1_mask_regions .tif
#masks\P9 10 MIX4 HI IPSII G4green IBA1pink CD68red 3x4 40x 1_mask.tif 
#images\P9 10 MIX4 HI IPSII G4green IBA1pink CD68red 3x4 40x 1.lsm