import os
import numpy as np
import mrcfile
import matplotlib.pyplot as plt


def bin_image(image, bin_size):
    """
    Downsample image using average binning.
    """

    h, w = image.shape

    new_h = h // bin_size
    new_w = w // bin_size

    cropped = image[
        :new_h * bin_size,
        :new_w * bin_size
    ]

    binned = cropped.reshape(
        new_h,
        bin_size,
        new_w,
        bin_size
    ).mean(axis=(1, 3))

    return binned.astype(np.float32)


# INPUT FILE
input_file = "14sep05c_c_00003gr_00014sq_00005hl_00003es_c.mrc"

# CHECK FILE
if not os.path.exists(input_file):
    raise FileNotFoundError(
        f"File not found: {input_file}"
    )

# GET FILE NAME
base_name = os.path.splitext(
    os.path.basename(input_file)
)[0]

# OPEN ORIGINAL MRC
with mrcfile.open(
    input_file,
    permissive=True
) as mrc:

    data = mrc.data

    # Handle 3D stack choose which slice to bin
    if data.ndim == 3:
        image_array = data[0] #change this 
    else:
        image_array = data


# BINNING FACTOR
bin_size = 4   #change this to change the binning factor

# BIN IMAGE
binned_image = bin_image(
    image_array,
    bin_size
)

# OUTPUT MRC FILE
output_file = (
    f"binning_{bin_size}_{base_name}.mrc"
)

# SAVE BINNED IMAGE AS MRC
with mrcfile.new(
    output_file,
    overwrite=True
) as mrc_out:

    mrc_out.set_data(
        binned_image.astype(np.float32)
    )

print(f"Saved MRC: {output_file}")

# DISPLAY SIDE BY SIDE
fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 6)
)

# ORIGINAL
axes[0].imshow(
    image_array,
    cmap="gray"
)

axes[0].set_title(
    f"Original\n{image_array.shape}"
)

axes[0].axis("off")

# BINNED
axes[1].imshow(
    binned_image,
    cmap="gray"
)

axes[1].set_title(
    f"Binned {bin_size}x{bin_size}\n{binned_image.shape}"
)

axes[1].axis("off")

plt.tight_layout()
plt.show()