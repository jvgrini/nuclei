import numpy as np
import napari
from skimage import io, filters, measure, restoration
import csv

img = io.imread("Images\P9 10 MIX4 HI IPSII G4green IBA1pink CD68red 3x4 40x 1.lsm")
n_labels = io.imread("masks\P9 10 MIX4 HI IPSII G4green IBA1pink CD68red 3x4 40x 1_masks.tif")
#spacing = ([0.4151, 0.4151])
spacing = ([0.3459, 0.3459, 0.9278])

labled_nuc = measure.label(n_labels)

print(labled_nuc.shape)
print(img.shape)

properties = measure.regionprops(labled_nuc, intensity_image=img)
pixel_volume = np.prod(spacing)
label_properties = []

for prop in properties:
    region_label = prop.label
    region_area = prop.area
    region_area_real = region_area * pixel_volume

    if region_area >20:
        region_mean_intensity = prop.mean_intensity
        ch1_intensity, ch2_intensity, ch3_intensity, ch4_intensity = region_mean_intensity

        label_properties.append({
            "label": region_label,
            "Area": region_area,
            "Area_um3": region_area_real,
            "Channel1_intensity": ch1_intensity,
            "Channel2_intensity": ch2_intensity,
            "Channel3_intensity": ch3_intensity,
            "Channel4_intensity": ch4_intensity,
        })

csv_file_path ="P9 10 MIX4 HI IPSII G4green IBA1pink CD68red 3x4 40x 1.csv"
fields = label_properties[0].keys()

with open(csv_file_path, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fields)
    writer.writeheader()
    writer.writerows(label_properties)