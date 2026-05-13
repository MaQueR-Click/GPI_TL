import numpy as np
import mrcfile
import matplotlib.pyplot as plt

from skimage.feature import match_template
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max

# ============================================================
# LOAD MICROGRAPH
# ============================================================

input_file = "padded_200_normalized_crop.mrc"

with mrcfile.open(input_file) as mrc:

    image = mrc.data.astype(np.float32)

    pixel_size = mrc.voxel_size.x

print(f"Pixel size: {pixel_size} Å/pixel")
print(f"Image shape: {image.shape}")

# Normalize micrograph
image = (image - np.mean(image)) / np.std(image)

# ============================================================
# TEMPLATE FILES
# ============================================================

template_files = [
    "6BDF_XY_template.mrc",
    "6BDF_XZ_template.mrc",
    "6BDF_YZ_template.mrc"
]

# Detection threshold
threshold = 0.60
pixel_size = 2.64 # Å/pixel, from PDB opening file
# Expected proteasome size
particle_size_pixels = int(150 / pixel_size)

print(f"Expected particle size: {particle_size_pixels} pixels")

# ============================================================
# PROCESS TEMPLATES
# ============================================================

for template_file in template_files:

    print("\n" + "="*60)
    print(f"Processing {template_file}")

    # --------------------------------------------------------
    # LOAD TEMPLATE
    # --------------------------------------------------------

    with mrcfile.open(template_file) as mrc:

        template = mrc.data.astype(np.float32)

    print(f"Template shape: {template.shape}")

    # --------------------------------------------------------
    # OPTIONAL BLUR
    # --------------------------------------------------------

    template = gaussian_filter(template, sigma=1)

    # --------------------------------------------------------
    # NORMALIZE TEMPLATE
    # --------------------------------------------------------

    template = (
        template - np.mean(template)
    ) / np.std(template)

    # --------------------------------------------------------
    # TEMPLATE MATCHING
    # --------------------------------------------------------

    result = match_template(
        image,
        template,
        pad_input=True
    )

    # --------------------------------------------------------
    # FIND PEAKS
    # --------------------------------------------------------

    coordinates = peak_local_max(
        result,
        min_distance=particle_size_pixels // 2,
        threshold_abs=threshold
    )

    print(f"Detections: {len(coordinates)}")

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8,8))

    ax.imshow(image, cmap='gray')

    for y, x in coordinates:

        circle = plt.Circle(
            (x, y),
            particle_size_pixels / 2,
            color='red',
            fill=False,
            linewidth=1.5
        )

        ax.add_patch(circle)

        score = result[y, x]

        ax.text(
            x,
            y,
            f"{score:.2f}",
            color='yellow',
            fontsize=8
        )

    ax.set_title(template_file)

    plt.show()

    # --------------------------------------------------------
    # SHOW CORRELATION MAP
    # --------------------------------------------------------

    plt.figure(figsize=(6,6))

    plt.imshow(result, cmap='viridis')

    plt.colorbar(label='Cross-correlation')

    plt.title(f"Correlation Map - {template_file}")

    plt.show()