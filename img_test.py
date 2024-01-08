import napari
import tifffile
import numpy as np
from stardist.models import StarDist3D
from stardist.plot import render_label
from csbdeep.utils import normalize
from matplotlib import pyplot as plt

nuclei = tifffile.imread("C4-229086 tilescan 63x NE.tif")
spacing = np.array([0.5, 0.1318, 0.1318])

#StarDist3D.from_pretrained()

norm_img = normalize(nuclei)

viewer = napari.view_image(
    nuclei,
    scale=spacing,
    ndisplay=3
)

viewer.add_image(
    norm_img,
    name="normalized",
    scale=spacing,
)

napari.run()