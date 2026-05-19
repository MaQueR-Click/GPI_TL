#!/usr/bin/env python3

import sys
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
import mrcfile

from Bio.PDB import PDBParser


def download_pdb(pdb_id):

    pdb_id = pdb_id.upper()

    pdb_url = (
        f"https://files.rcsb.org/download/{pdb_id}.pdb"
    )

    pdb_file = f"{pdb_id}.pdb"

    print(f"Downloading {pdb_id}...")

    with urllib.request.urlopen(pdb_url) as response:

        pdb_content = response.read().decode("utf-8")

    with open(pdb_file, "w") as f:

        f.write(pdb_content)

    print(f"Saved: {pdb_file}")

    return pdb_file


def extract_coordinates(pdb_file, pdb_id):

    parser = PDBParser(QUIET=True)

    structure = parser.get_structure(
        pdb_id,
        pdb_file
    )

    model = structure[0]

    coords = []

    for chain in model.get_chains():

        for residue in chain.get_residues():

            for atom in residue.get_atoms():

                coord = atom.get_coord()

                coords.append([
                    coord[0],
                    coord[1],
                    coord[2]
                ])

    return np.array(coords)


def create_projection(
    coord1,
    coord2,
    bins,
    box_size
):

    half_box = box_size / 2

    H, xedges, yedges = np.histogram2d(
        coord1,
        coord2,
        bins=bins,
        range=[
            [-half_box, half_box],
            [-half_box, half_box]
        ]
    )

    if H.max() > 0:

        H = H / H.max()

    H = 1.0 - H

    return H.astype(np.float32)


def save_mrc(
    filename,
    image,
    pixel_size
):

    with mrcfile.new(
        filename,
        overwrite=True
    ) as mrc:

        mrc.set_data(
            image.astype(np.float32)
        )

        mrc.voxel_size = pixel_size

        mrc.header.origin = (
            0.0,
            0.0,
            0.0
        )


def main():

    if len(sys.argv) != 3:

        print(
            f"\nUsage:\n"
            f"python3 {sys.argv[0]} <PDB_ID> <pixel_size>\n"
        )

        sys.exit(1)

    pdb_id = sys.argv[1]
    pixel_size = float(sys.argv[2])

    try:

        # DOWNLOAD PDB
        pdb_file = download_pdb(pdb_id)

        # EXTRACT COORDINATES
        coords = extract_coordinates(
            pdb_file,
            pdb_id
        )

        print(f"Total atoms: {len(coords)}")

        # CENTER COORDINATES
        coords_centered = (
            coords - coords.mean(axis=0)
        )

        # BOX SIZE
        x_range = (
            coords_centered[:, 0].max()
            - coords_centered[:, 0].min()
        )

        y_range = (
            coords_centered[:, 1].max()
            - coords_centered[:, 1].min()
        )

        z_range = (
            coords_centered[:, 2].max()
            - coords_centered[:, 2].min()
        )

        padding = 0

        box_size_angstrom = max(
            x_range,
            y_range,
            z_range
        ) + 2 * padding

        img_size = int(
            np.ceil(
                box_size_angstrom
                / pixel_size
            )
        )

        print(
            f"Pixel size: {pixel_size} Å/pixel"
        )

        print(
            f"Box size: {box_size_angstrom:.2f} Å"
        )

        print(
            f"Image size: {img_size} x {img_size}"
        )

        # CREATE PROJECTIONS
        proj_xy = create_projection(
            coords_centered[:, 0],
            coords_centered[:, 1],
            img_size,
            box_size_angstrom
        )

        proj_xz = create_projection(
            coords_centered[:, 0],
            coords_centered[:, 2],
            img_size,
            box_size_angstrom
        )

        proj_yz = create_projection(
            coords_centered[:, 1],
            coords_centered[:, 2],
            img_size,
            box_size_angstrom
        )

        # SAVE MRC FILES
        save_mrc(
            f"{pdb_id}_XY_template.mrc",
            proj_xy,
            pixel_size
        )

        save_mrc(
            f"{pdb_id}_XZ_template.mrc",
            proj_xz,
            pixel_size
        )

        save_mrc(
            f"{pdb_id}_YZ_template.mrc",
            proj_yz,
            pixel_size
        )

        print("\nSaved templates:")

        print(
            f"  - {pdb_id}_XY_template.mrc"
        )

        print(
            f"  - {pdb_id}_XZ_template.mrc"
        )

        print(
            f"  - {pdb_id}_YZ_template.mrc"
        )

        # DISPLAY
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(15, 5)
        )

        axes[0].imshow(
            proj_xy.T,
            cmap="gray",
            origin="lower"
        )

        axes[0].set_title(
            "XY Projection"
        )

        axes[1].imshow(
            proj_xz.T,
            cmap="gray",
            origin="lower"
        )

        axes[1].set_title(
            "XZ Projection"
        )

        axes[2].imshow(
            proj_yz.T,
            cmap="gray",
            origin="lower"
        )

        axes[2].set_title(
            "YZ Projection"
        )

        plt.tight_layout()

        plt.show()

    except Exception as e:

        print(f"Error: {e}")


if __name__ == "__main__":

    main()