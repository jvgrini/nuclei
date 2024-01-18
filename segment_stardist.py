import numpy as np
import os
from skimage import io
from stardist.models import StarDist3D
import napari
from csbdeep.utils import normalize
import glob

spacing = ([0.3459, 0.3459, 0.9278])

folder_path = "images_sham"
image_files = glob.glob(f"{folder_path}/*.lsm")


model = StarDist3D(None, name='Hippocampus7.0', basedir='models')

def segment(img_path):

    new_image = io.imread(img_path)
    print(new_image.shape)
    if new_image.shape[-1] == 4:
        normalized = normalize(new_image[:,:,:,3])
    else:
        normalized = normalize(new_image[:,:,:,2])

    labels, _ = model.predict_instances(normalized, n_tiles=(8,8,1))

    directory, filename = os.path.split(img_path)
    without_extension, extension = os.path.splitext(filename)
    mask_file_name = f"{without_extension}_mask.tif"
    mask_path = os.path.join("masks_sham", mask_file_name)

    io.imsave(mask_path, labels)

for image_path in image_files:
    segment(image_path)

