#!/usr/bin/env python3

# ============================================================
# MULTI-TEMPLATE NCC MATCHING
# TERMINAL VERSION
#
# Example:
#
# python3 ncc_matching.py \
#     image.mrc \
#     2.64 \
#     6BDF_XY_template.mrc:115:0.26 \
#     6BDF_XZ_template.mrc:150:0.30
#
# ============================================================

import os
import sys
import numpy as np
import mrcfile
import matplotlib.pyplot as plt

from scipy.ndimage import rotate, gaussian_filter
from skimage.feature import (
    match_template,
    peak_local_max
)


# ============================================================
# UTILITIES
# ============================================================

def add_padding(image, pad):

    return np.pad(
        image,
        ((pad, pad), (pad, pad)),
        mode="constant",
        constant_values=0
    )


def normalize(x, eps=1e-8):

    return (
        x - np.mean(x)
    ) / (
        np.std(x) + eps
    )


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(
    image,
    sigma=1,
    pad_size=50
):

    image = gaussian_filter(
        image,
        sigma=sigma
    )

    image = normalize(image)

    return add_padding(
        image,
        pad_size
    )


# ============================================================
# PREPROCESS TEMPLATE
# ============================================================

def preprocess_template(
    template,
    sigma=1,
    pad_size=0,
    noise_sigma=0.0
):

    template = gaussian_filter(
        template,
        sigma=sigma
    )

    if noise_sigma > 0:

        template += np.random.normal(
            0,
            noise_sigma,
            template.shape
        )

    template = normalize(template)

    return add_padding(
        template,
        pad_size
    )


# ============================================================
# PARTICLE EXTRACTION
# ============================================================

def extract_particle(
    image,
    center_y,
    center_x,
    box_shape
):

    h, w = box_shape

    half_h = h // 2
    half_w = w // 2

    y1 = center_y - half_h
    y2 = y1 + h

    x1 = center_x - half_w
    x2 = x1 + w

    if y1 < 0 or x1 < 0:
        return None

    if y2 > image.shape[0]:
        return None

    if x2 > image.shape[1]:
        return None

    particle = image[
        y1:y2,
        x1:x2
    ]

    if particle.shape != (h, w):
        return None

    return particle


# ============================================================
# DUPLICATE CHECK
# ============================================================

def is_duplicate(
    y,
    x,
    accepted_particles,
    distance_thresh
):

    for p in accepted_particles:

        dy = y - p["y"]
        dx = x - p["x"]

        dist = np.sqrt(
            dx**2 + dy**2
        )

        if dist < distance_thresh:
            return True

    return False


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) < 4:

        print(
            "\nUsage:\n"
            "python3 ncc_matching.py "
            "<image.mrc> "
            "<pixel_size> "
            "<template.mrc:diameter:threshold> "
            "[template2.mrc:diameter:threshold] ...\n"
        )

        sys.exit(1)

    # --------------------------------------------------------
    # INPUTS
    # --------------------------------------------------------

    input_file = sys.argv[1]

    pixel_size = float(
        sys.argv[2]
    )

    template_args = sys.argv[3:]

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    with mrcfile.open(
        input_file,
        permissive=True
    ) as mrc:

        original_image = (
            mrc.data.astype(np.float32)
        )

    print("=" * 70)
    print("MICROGRAPH")
    print("=" * 70)

    print(
        "Original shape:",
        original_image.shape
    )

    image = preprocess_image(
        original_image,
        sigma=1,
        pad_size=50
    )

    print(
        "Padded shape:",
        image.shape
    )

    # --------------------------------------------------------
    # TEMPLATE CONFIGS
    # --------------------------------------------------------

    template_configs = []

    for arg in template_args:

        try:

            template_file, diameter, threshold = (
                arg.split(":")
            )

            template_configs.append({

                "file": template_file,
                "diameter": float(diameter),
                "threshold": float(threshold)

            })

        except:

            print(
                f"\nInvalid template format:\n{arg}\n"
            )

            print(
                "Expected:\n"
                "template.mrc:diameter:threshold\n"
            )

            sys.exit(1)

    # --------------------------------------------------------
    # PARAMETERS
    # --------------------------------------------------------

    angles = np.arange(
        0,
        181,
        10
    )

    noise_sigma = 0.0

    output_dir = (
        "extracted_particles_unique"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    duplicate_distance_px = 15

    # --------------------------------------------------------
    # FIGURE
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        len(template_configs),
        4,
        figsize=(22, 6 * len(template_configs))
    )

    if len(template_configs) == 1:

        axes = np.expand_dims(
            axes,
            axis=0
        )

    # --------------------------------------------------------
    # STORE DETECTIONS
    # --------------------------------------------------------

    all_detections = []

    # ========================================================
    # TEMPLATE LOOP
    # ========================================================

    for idx, cfg in enumerate(template_configs):

        template_file = cfg["file"]

        particle_size_px = max(
            1,
            int(
                cfg["diameter"]
                / pixel_size
            )
        )

        threshold = cfg["threshold"]

        print("\n" + "=" * 70)
        print("TEMPLATE:", template_file)
        print(
            "Particle size (px):",
            particle_size_px
        )
        print(
            "Threshold:",
            threshold
        )
        print("=" * 70)

        # ----------------------------------------------------
        # LOAD TEMPLATE
        # ----------------------------------------------------

        with mrcfile.open(
            template_file,
            permissive=True
        ) as mrc:

            raw_template = (
                mrc.data.astype(np.float32)
            )

        template_shape = raw_template.shape

        template0 = preprocess_template(
            raw_template,
            sigma=1,
            pad_size=0,
            noise_sigma=noise_sigma
        )

        # ----------------------------------------------------
        # NCC
        # ----------------------------------------------------

        global_cc = np.zeros_like(image)

        for angle in angles:

            rotated = rotate(
                template0,
                angle,
                reshape=False,
                order=1,
                mode="reflect",
                cval=0
            )

            cc = match_template(
                image,
                rotated,
                pad_input=True
            )

            global_cc = np.maximum(
                global_cc,
                cc
            )

        # ----------------------------------------------------
        # DETECTIONS
        # ----------------------------------------------------

        coords = peak_local_max(
            global_cc,
            min_distance=max(
                1,
                particle_size_px // 2
            ),
            threshold_abs=threshold
        )

        print(
            "Raw detections:",
            len(coords)
        )

        template_name = os.path.splitext(
            os.path.basename(template_file)
        )[0]

        for y, x in coords:

            score = global_cc[y, x]

            all_detections.append({

                "y": y,
                "x": x,
                "score": score,
                "template": template_name,
                "template_shape": template_shape

            })

        # ----------------------------------------------------
        # PLOTS
        # ----------------------------------------------------

        axes[idx, 0].imshow(
            template0,
            cmap="gray"
        )

        axes[idx, 0].set_title(
            template_name
        )

        axes[idx, 0].axis("off")

        im = axes[idx, 1].imshow(
            global_cc,
            cmap="inferno"
        )

        axes[idx, 1].set_title(
            f"NCC map ({threshold})"
        )

        plt.colorbar(
            im,
            ax=axes[idx, 1]
        )

        axes[idx, 2].imshow(
            original_image,
            cmap="gray"
        )

        for y, x in coords:

            axes[idx, 2].add_patch(

                plt.Circle(
                    (x, y),
                    particle_size_px / 2,
                    fill=False,
                    color="yellow",
                    linewidth=1
                )
            )

        axes[idx, 2].set_title(
            f"Detections: {len(coords)}"
        )

        axes[idx, 2].axis("off")

        axes[idx, 3].hist(
            global_cc.ravel(),
            bins=100
        )

        axes[idx, 3].axvline(
            threshold,
            color="red",
            linestyle="--"
        )

        axes[idx, 3].set_title(
            "NCC distribution"
        )

    # ========================================================
    # SORT DETECTIONS
    # ========================================================

    all_detections = sorted(
        all_detections,
        key=lambda d: d["score"],
        reverse=True
    )

    accepted_particles = []

    saved_count = 0
    duplicate_count = 0

    # ========================================================
    # FILTER + SAVE
    # ========================================================

    for det in all_detections:

        y = det["y"]
        x = det["x"]

        score = det["score"]

        template_shape = det["template_shape"]

        if is_duplicate(
            y,
            x,
            accepted_particles,
            duplicate_distance_px
        ):

            duplicate_count += 1
            continue

        particle = extract_particle(
            image=original_image,
            center_y=y,
            center_x=x,
            box_shape=template_shape
        )

        if particle is None:
            continue

        output_path = os.path.join(

            output_dir,

            (
                f"particle_"
                f"{saved_count:05d}"
                f"_score_{score:.3f}.mrc"
            )

        )

        with mrcfile.new(
            output_path,
            overwrite=True
        ) as mrc:

            mrc.set_data(
                particle.astype(np.float32)
            )

        accepted_particles.append({

            "y": y,
            "x": x,
            "score": score

        })

        saved_count += 1

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(
        "Total detections:",
        len(all_detections)
    )

    print(
        "Unique particles kept:",
        saved_count
    )

    print(
        "Removed duplicates:",
        duplicate_count
    )

    print(
        "Output directory:",
        output_dir
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    main()