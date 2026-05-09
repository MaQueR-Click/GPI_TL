import os
import numpy as np
import mrcfile
import matplotlib.pyplot as plt

def apply_padding(img, pad_width, mode='constant', constant_value=255):
    """
    Apply padding to an image using NumPy.
    
    Parameters:
    -----------
    img : ndarray
        Input image
    pad_width : int or tuple
        Number of pixels to pad on each side
    mode : str
        Padding mode: 'constant', 'edge', 'reflect', 'wrap','mean','median'
    constant_value : int
        Value used for constant padding
    
    Returns:
    --------
    padded : ndarray
        Padded image
    """
    if isinstance(pad_width, int):
        pad_width = ((pad_width, pad_width), (pad_width, pad_width))
    
    if mode == 'constant':
        padded = np.pad(img, pad_width, mode='constant', constant_values=constant_value)
    else:
        padded = np.pad(img, pad_width, mode=mode)
    
    return padded

# INPUT FILE
input_file = "/home/tlaborde/GPI_proj/binning_4_14sep05c_c_00003gr_00014sq_00005hl_00003es_c.mrc"

# CHECK FILE
if not os.path.exists(input_file):
    raise FileNotFoundError(
        f"File not found: {input_file}"
    )   

# GET FILE NAME
base_name = os.path.splitext(
    os.path.basename(input_file))[0]    

# OPEN ORIGINAL MRC
with mrcfile.open(
    input_file,
    permissive=True
) as mrc:

    data = mrc.data

    # Handle 3D stack choose which slice to pad
    if data.ndim == 3:
        image_array = data[0] #change this 
    else:
        image_array = data

# Padding parameters
pad_width = 200 # Number of pixels to pad on each side 

# PAD IMAGE
padded_image = apply_padding(image_array, pad_width, mode='constant', constant_value=0)  # Change mode and constant_value as needed between 0 and 32 

# OUTPUT MRC FILE
output_file = (
    f"padded_{pad_width}_{base_name}.mrc"
)

# SAVE Padded IMAGE AS MRC
with mrcfile.new(
    output_file,
    overwrite=True
) as mrc_out:

    mrc_out.set_data(
        padded_image.astype(np.float32)
    )

print(f"Padding applied: {pad_width} pixels")
print(np.sum(padded_image == 0), "pixels padded with 0")
print(image_array.dtype)
print(image_array.min(), image_array.max())

# DISPLAY PADDED IMAGE
plt.imshow(
    padded_image,
    cmap='gray',
    vmin=np.percentile(padded_image, 1),
    vmax=np.percentile(padded_image, 99))
plt.title("Padded Image")
plt.axis("off")
plt.show()