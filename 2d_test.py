from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np
import matplotlib
matplotlib.rcParams["image.interpolation"] = 'none'
import matplotlib.pyplot as plt

import napari

from glob import glob
from skimage.io import imread
from csbdeep.utils import Path, normalize
from csbdeep.io import save_tiff_imagej_compatible

from stardist import random_label_cmap, _draw_polygons, export_imagej_rois
from stardist.models import StarDist2D

#spacing = np.array([0.4151, 0.4151])
spacing = np.array([0.3459,0.3459])
np.random.seed(6)
lbl_cmap = random_label_cmap()
image = "Images\SUM_P9 10 MIX4 HI IPSII G4green IBA1pink CD68red 3x4 40x 1.tif"
read_image = imread(image)
X = read_image[:,:,3]

n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]
axis_norm = (0,1)   # normalize channels independently
# axis_norm = (0,1,2) # normalize channels jointly
if n_channel > 1:
    print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))

demo_model = False

if demo_model:
    print (
        "NOTE: This is loading a previously trained demo model!\n"
        "      Please set the variable 'demo_model = False' to load your own trained model.",
        file=sys.stderr, flush=True
    )
    model = StarDist2D.from_pretrained('2D_demo')
else:
    model = StarDist2D.from_pretrained("2D_versatile_fluo")
None;

img =  normalize(X, 1,99.8, axis=axis_norm)
labels, details = model.predict_instances(img)

viewer = napari.view_image(
    read_image,
    channel_axis= 2,
    scale = spacing

)
viewer.add_labels(
    labels,
    name = "True labels",
    scale = spacing
)
napari.run()