import napari
from skimage import io

img = io.imread("Images\P9 10 MIX4 HI IPSII G4green IBA1pink CD68red 3x4 40x 1.lsm")
labels = io.imread("masks\P9 10 MIX4 HI IPSII G4green IBA1pink CD68red 3x4 40x 1_mask.tif")
region_label = io.imread("masks_regions_extended\P9 10 MIX4 HI IPSII G4green IBA1pink CD68red 3x4 40x 1_mask_regions.tif")
spacing = ([0.3459, 0.3459, 0.9278])

viewer = napari.view_image(
    img,
    channel_axis=3,
    scale = spacing,
    ndisplay=3
)

viewer.add_labels(
    labels,
    name="1",
    scale=spacing
)
viewer.add_labels(
    region_label,
    name="region",
    scale=spacing
)
#)
napari.run()


# P9 10 MIX4 HI IPSII G4green IBA1pink CD68red 3x4 40x 1_mask_regions .tif
# P9 10 MIX4 HI IPSII G4green IBA1pink CD68red 3x4 40x 1_mask.tif

#masks_regions_extended\P9 10 MIX4 HI IPSII G4green IBA1pink CD68red 3x4 40x 1_mask_regions.tif
#masks_regions_extended\P9 10 MIX4 HI IPSII G4green IBA1pink CD68red 3x4 40x 1_mask_regions .tif
#masks\P9 10 MIX4 HI IPSII G4green IBA1pink CD68red 3x4 40x 1_mask.tif 
#images\P9 10 MIX4 HI IPSII G4green IBA1pink CD68red 3x4 40x 1.lsm