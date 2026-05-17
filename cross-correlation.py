# ============================================================
# MULTI-TEMPLATE NCC MATCHING + PARTICLE EXTRACTION
# ============================================================

import os
import numpy as np
import mrcfile
import matplotlib.pyplot as plt

from scipy.ndimage import rotate, gaussian_filter
from skimage.feature import match_template, peak_local_max


# ============================================================
# UTILITIES
# ============================================================

def add_padding(image, pad):
    return np.pad(
        image,
        ((pad, pad), (pad, pad)),
        mode='constant',
        constant_values=0
    )


def normalize(x, eps=1e-8):
    return (x - np.mean(x)) / (np.std(x) + eps)


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(image, sigma=1, pad_size=50):
    image = gaussian_filter(image, sigma=sigma)
    image = normalize(image)
    return add_padding(image, pad_size)


# ============================================================
# PREPROCESS TEMPLATE
# ============================================================

def preprocess_template(template, sigma=1, pad_size=50, noise_sigma=0.0):
    template = gaussian_filter(template, sigma=sigma)

    if noise_sigma > 0:
        template += np.random.normal(0, noise_sigma, template.shape)

    template = normalize(template)
    return add_padding(template, pad_size)


# ============================================================
# PARTICLE EXTRACTION
# ============================================================

def extract_particle(image, center_y, center_x, box_shape):

    h, w = box_shape

    half_h = h // 2
    half_w = w // 2

    y1 = center_y - half_h
    y2 = y1 + h

    x1 = center_x - half_w
    x2 = x1 + w

    # safety check
    if y1 < 0 or x1 < 0:
        return None

    if y2 > image.shape[0] or x2 > image.shape[1]:
        return None

    particle = image[y1:y2, x1:x2]

    if particle.shape != (h, w):
        return None

    return particle


# ============================================================
# INPUT MICROGRAPH
# ============================================================

input_file = "GPI_TL/binning_4_14sep05c_c_00003gr_00014sq_00005hl_00003es_c.mrc"

with mrcfile.open(input_file, permissive=True) as mrc:
    original_image = mrc.data.astype(np.float32)

pixel_size = 2.64

print("=" * 70)
print("MICROGRAPH")
print("=" * 70)
print("Original shape:", original_image.shape)

image = preprocess_image(original_image, sigma=1, pad_size=50)

print("Padded shape:", image.shape)


# ============================================================
# TEMPLATE CONFIGS
# ============================================================

template_configs = [
    {
        "file": "6BDF_XY_template.mrc",
        "diameter": 115,
        "threshold": 0.28
    },
    {
        "file": "6BDF_XZ_template.mrc",
        "diameter": 150,
        "threshold": 0.30
    },
    {
        "file": "6BDF_YZ_template.mrc",
        "diameter": 150,
        "threshold": 0.30
    }
]


# ============================================================
# PARAMETERS
# ============================================================

angles = np.arange(0, 181, 10)
noise_sigma = 0.0

output_dir = "extracted_particles"
os.makedirs(output_dir, exist_ok=True)


# ============================================================
# FIGURE
# ============================================================

fig, axes = plt.subplots(len(template_configs), 4, figsize=(22, 15))

if len(template_configs) == 1:
    axes = np.expand_dims(axes, axis=0)


# ============================================================
# MAIN LOOP
# ============================================================

for idx, cfg in enumerate(template_configs):

    template_file = cfg["file"]
    particle_size_px = max(1, int(cfg["diameter"] / pixel_size))
    threshold = cfg["threshold"]

    print("\n" + "=" * 70)
    print("TEMPLATE:", template_file)
    print("Particle size (px):", particle_size_px)
    print("Threshold:", threshold)
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD TEMPLATE
    # --------------------------------------------------------

    with mrcfile.open(template_file, permissive=True) as mrc:
        raw_template = mrc.data.astype(np.float32)

    template_shape = raw_template.shape

    template0 = preprocess_template(
        raw_template,
        sigma=1,
        pad_size=0,
        noise_sigma=noise_sigma
    )

    print("Template extraction shape:", template_shape)

    # --------------------------------------------------------
    # NCC ACCUMULATION
    # --------------------------------------------------------

    global_cc = np.zeros_like(image)

    for angle in angles:

        rotated = rotate(
            template0,
            angle,
            reshape=False,
            order=1,
            mode='reflect',
            cval=0
        )

        cc = match_template(image, rotated, pad_input=True)

        global_cc = np.maximum(global_cc, cc)

    # --------------------------------------------------------
    # DETECTION
    # --------------------------------------------------------

    coords = peak_local_max(
        global_cc,
        min_distance=max(1, particle_size_px // 2),
        threshold_abs=threshold
    )

    print("Detections:", len(coords))

    # --------------------------------------------------------
    # EXTRACT PARTICLES
    # --------------------------------------------------------

    template_name = os.path.splitext(
        os.path.basename(template_file)
    )[0]

    template_output_dir = os.path.join(
        output_dir,
        template_name
    )

    os.makedirs(template_output_dir, exist_ok=True)

    saved_count = 0

    for particle_id, (y, x) in enumerate(coords):

        particle = extract_particle(
            image=original_image,
            center_y=y,
            center_x=x,
            box_shape=template_shape
        )

        if particle is None:
            continue

        score = global_cc[y, x]

        output_path = os.path.join(
            template_output_dir,
            f"{template_name}_particle_{particle_id:04d}_score_{score:.3f}.mrc"
        )

        with mrcfile.new(output_path, overwrite=True) as mrc:
            mrc.set_data(particle.astype(np.float32))

        saved_count += 1

    print("Saved particles:", saved_count)
    print("Output folder:", template_output_dir)

    # --------------------------------------------------------
    # PLOT TEMPLATE
    # --------------------------------------------------------

    axes[idx, 0].imshow(template0, cmap="gray")
    axes[idx, 0].set_title(template_file)
    axes[idx, 0].axis("off")

    # --------------------------------------------------------
    # PLOT NCC MAP
    # --------------------------------------------------------

    im = axes[idx, 1].imshow(global_cc, cmap="inferno")
    axes[idx, 1].set_title(f"NCC map (thr={threshold})")
    plt.colorbar(im, ax=axes[idx, 1], fraction=0.046)

    # --------------------------------------------------------
    # PLOT DETECTIONS
    # --------------------------------------------------------

    axes[idx, 2].imshow(original_image, cmap="gray")

    for y, x in coords:

        score = global_cc[y, x]

        axes[idx, 2].add_patch(
            plt.Circle(
                (x, y),
                particle_size_px / 2,
                fill=False,
                color="lime",
                linewidth=1.5
            )
        )

        axes[idx, 2].text(
            x,
            y,
            f"{score:.2f}",
            color="yellow",
            fontsize=6
        )

    axes[idx, 2].set_title(
        f"Detections: {len(coords)}"
    )
    axes[idx, 2].axis("off")

    # --------------------------------------------------------
    # HISTOGRAM
    # --------------------------------------------------------

    axes[idx, 3].hist(global_cc.ravel(), bins=100)

    axes[idx, 3].axvline(
        threshold,
        linestyle="--",
        color="red",
        label=f"thr={threshold}"
    )

    axes[idx, 3].set_title("NCC distribution")
    axes[idx, 3].legend()


# ============================================================
# SHOW
# ============================================================

plt.tight_layout()
plt.show()

print("\nDone.")
print("Extracted particles saved in:", output_dir)