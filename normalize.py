import mrcfile 
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import correlate2d
# INPUT FILE
input_file = "cropped_image_test.mrc"

def normalize_image(image):
    """
    Normalize image to range [0, 1]
    """
    min_val = np.min(image)
    max_val = np.max(image)
    
    if max_val > min_val:
        normalized = (image - min_val) / (max_val - min_val)
    else:
        normalized = np.zeros_like(image)
    
    return normalized.astype(np.float32)


with mrcfile.open(input_file, permissive=True) as mrc:
    data = mrc.data

    # Handle 3D stack choose which slice to normalize
    if data.ndim == 3:
        image_array = data[0] #change this 
    else:
        image_array = data
normalized_image = normalize_image(data)
# OUTPUT MRC FILE
output_file = (
    f"normalized_crop.mrc")
# SAVE Normalized IMAGE AS MRC
with mrcfile.new(
    output_file,
    overwrite=True
) as mrc_out:

    mrc_out.set_data(
        normalized_image.astype(np.float32)
    )

print("Normalization applied")
print(f"Saved MRC: {output_file}")
print("Shape :", normalized_image.shape)
print("dtype :", normalized_image.dtype)
print("Min :", normalized_image.min())
print("Max :", normalized_image.max())

plt.imshow(normalized_image, cmap="gray")