import numpy as np
import os
from skimage import io
from stardist.models import StarDist3D
import napari
from csbdeep.utils import normalize
import glob
import czifile


spacing = ([0.3459, 0.3459, 0.9278])

folder_path = "imagesAndMasks\liv\ca3"
image_files = glob.glob(f"{folder_path}/*.czi")


model = StarDist3D(None, name='Hippocampus9.1', basedir='models')

def segment(img_path):

    if '.czi' in img_path:
        new_image = czifile.imread(img_path)
        new_image = np.squeeze(new_image)
        new_image = np.transpose(new_image, (1,2,3,0))
    else:
        new_image = io.imread(img_path)
    print(new_image.shape)
    
    if new_image.shape[-1] == 4:
        normalized = normalize(new_image[:,:,:,1])
    else:
        normalized = normalize(new_image[:,:,:,1])

    labels, _ = model.predict_instances(normalized, n_tiles=(4,4,1))

    directory, filename = os.path.split(img_path)
    without_extension, extension = os.path.splitext(filename)
    mask_file_name = f"{without_extension}_mask.tif"
    mask_path = os.path.join("imagesAndMasks\liv\masks", mask_file_name)

    io.imsave(mask_path, labels)

for image_path in image_files:
    segment(image_path)

