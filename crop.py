"""cropping of the image for testing purposes"""
import mrcfile
import numpy as np
import os
import matplotlib.pyplot as plt

# INPUT FILE
input_file = "GPI_TL/binning_4_14sep05c_00024sq_00006hl_00003es_c.mrc"

# CHECK FILE
if not os.path.exists(input_file):
    raise FileNotFoundError(f"File not found: {input_file}")

# GET FILE NAME
base_name = os.path.splitext(os.path.basename(input_file))[0]   

def crop_image(image, crop_size):
    """
    Crop the image to the specified size.
    """
    h, w = image.shape
    crop_h, crop_w = crop_size

    if crop_h > h or crop_w > w:
        raise ValueError("Crop size must be smaller than the image size.")

    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2

    cropped = image[start_h:start_h + crop_h, start_w:start_w + crop_w]

    return cropped.astype(np.float32)

# OPEN ORIGINAL MRC
with mrcfile.open(input_file, permissive=True) as mrc:
    image_array = mrc.data

crop_size = (512, 512)
cropped_image = crop_image(image_array, crop_size)

# OUTPUT MRC FILE
output_file = f"cropped_image_test.mrc"
# SAVE CROPPED IMAGE AS MRC
with mrcfile.new(output_file, overwrite=True) as mrc_out:
    mrc_out.set_data(cropped_image.astype(np.float32))
print("Cropping applied")
print(f"Saved MRC: {output_file}")
print("Shape :", cropped_image.shape)
print("dtype :", cropped_image.dtype)
print("Min :", cropped_image.min())
print("Max :", cropped_image.max())
