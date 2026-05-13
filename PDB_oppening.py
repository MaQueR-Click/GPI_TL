import urllib.request
#import io
from Bio.PDB import PDBParser, PDBIO
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import mrcfile
import os

print("="*70)
print("PART 7: PDB FILES - PROTEIN STRUCTURES & 2D PROJECTIONS")
print("="*70)

print("\n" + "="*70)
print("📚 WHAT IS A PDB FILE?")
print("="*70)
print("""
PDB (Protein Data Bank) Format:
  - Standard text format for storing 3D atomic structures
  - Contains coordinates of atoms in proteins, DNA, RNA, small molecules
  - Publicly available at: https://www.rcsb.org/

PDB File Structure:
  - HEADER: Basic information about the structure
  - TITLE: Name/description of the molecule
  - ATOM records: X, Y, Z coordinates of each atom
  - HETATM: Heterogeneous atoms (ligands, water, ions)
  - CONECT: Connectivity information between atoms
  - END: Marks end of file

Each structure has a 4-character PDB ID (e.g., 6BDF)
  - First 3 characters: Unique identifier
  - Last character: Chain ID

Applications:
  - Drug discovery and molecular docking
  - Protein structure validation
  - Creating 2D projections (electron microscopy simulation)
  - Computational biology research
""")

print("\n" + "="*70)
print("📥 FETCHING PDB STRUCTURE: 6BDF")
print("="*70)

pdb_id = "6BDF"
pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"

try:
    print(f"Downloading {pdb_id} from RCSB PDB...")
    with urllib.request.urlopen(pdb_url, timeout=10) as response:
        pdb_content = response.read().decode('utf-8')
    print(f"✓ Successfully downloaded {len(pdb_content)} bytes")
    
    # Save to file
    pdb_file = f"{pdb_id}.pdb"
    with open(pdb_file, 'w') as f:
        f.write(pdb_content)
    print(f"✓ Saved to {pdb_file}")
    
    # Parse the PDB file
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, pdb_file)
    
    print("\n" + "="*70)
    print("🔬 STRUCTURE INFORMATION")
    print("="*70)
    
    # Extract information
    model = structure[0]
    chains = list(model.get_chains())
    
    print(f"\nStructure ID: {pdb_id}")
    print(f"Number of models: {len(structure)}")
    print(f"Number of chains: {len(chains)}")
    
    for chain in chains:
        residues = list(chain.get_residues())
        atoms = list(chain.get_atoms())
        print(f"\nChain {chain.id}:")
        print(f"  - Residues: {len(residues)}")
        print(f"  - Atoms: {len(atoms)}")
    
    print("\n" + "="*70)
    print("📋 PDB FILE FORMAT - ATOM RECORD EXPLANATION")
    print("="*70)
    print("""
Each ATOM line contains 12 fixed-width columns:

ATOM      1  N   ALA A   1      20.154  29.699   5.276  1.00 20.00           N
├─────┬────┬───┬──┬──┬──┬────────────────────┬──┬────┬────────┘
│     │    │   │  │  │  │  X, Y, Z           │  │B  │Element
│     │    │   │  │  │  │  Coordinates (Å)   │  │   │
│     │    │   │  │  │  │                    │Occ│   │
Column│    │   │  │  │  └─ Atomic positions  │   │   │
      │    │   │  │  │
   1-6│    │   │  │  └─ Residue #
      │    │   │  │
   7-11│  Name  │  └─ Chain ID (A, B, C...)
      │         │
      └─ Atom # └─ Residue type (ALA=Alanine, GLY=Glycine, etc.)

Key Columns:
  - 1-6: ATOM/HETATM record type
  - 7-11: Atom serial number
  - 13-16: Atom name (CA=Alpha carbon, CB=Beta carbon, N=Backbone N, etc.)
  - 18-20: Residue name (3-letter code)
  - 22: Chain identifier
  - 23-26: Residue sequence number
  - 31-38: X coordinate (Ångströms)
  - 39-46: Y coordinate
  - 47-54: Z coordinate
  - 55-60: Occupancy (1.0 = full occupancy, <1.0 = partial)
  - 61-66: Temperature factor/B-factor (protein flexibility)
  - 77-78: Element symbol
""")
    
    # Extract atomic coordinates
    print("\n" + "="*70)
    print("🧬 ATOMIC COORDINATES FROM 6BDF")
    print("="*70)
    
    all_atoms = []
    for chain in model.get_chains():
        for residue in chain.get_residues():
            for atom in residue.get_atoms():
                coord = atom.get_coord()
                all_atoms.append({
                    'name': atom.name,
                    'residue': residue.resname,
                    'residue_id': residue.id[1],
                    'x': coord[0],
                    'y': coord[1],
                    'z': coord[2],
                    'b_factor': atom.bfactor
                })
    
    print(f"Total atoms: {len(all_atoms)}")
    print(f"\nFirst 10 atoms:")
    print(f"{'Atom':6} {'Residue':8} {'Res#':5} {'X (Å)':10} {'Y (Å)':10} {'Z (Å)':10} {'B-factor':10}")
    print("-" * 60)
    for atom in all_atoms[:10]:
        print(f"{atom['name']:6} {atom['residue']:8} {atom['residue_id']:5d} {atom['x']:10.3f} {atom['y']:10.3f} {atom['z']:10.3f} {atom['b_factor']:10.2f}")
    
    # Extract coordinates for visualization
    coords = np.array([[atom['x'], atom['y'], atom['z']] for atom in all_atoms])
    
    print(f"\nCoordinate statistics:")
    print(f"  X range: [{coords[:, 0].min():.3f}, {coords[:, 0].max():.3f}] Å")
    print(f"  Y range: [{coords[:, 1].min():.3f}, {coords[:, 1].max():.3f}] Å")
    print(f"  Z range: [{coords[:, 2].min():.3f}, {coords[:, 2].max():.3f}] Å")
    
    print("\n" + "="*70)
    print("🎯 CREATING DENSITY-BASED 2D PROJECTIONS (SUMMED GRAYSCALE)")
    print("="*70)

    # Center coordinates
    coords_centered = coords - coords.mean(axis=0)


 # Pixel size imposed (Å/pixel)
    pixel_size = image_pixel_size

# Padding around molecule
    padding = 5 # Å

# Determine physical box size
    x_range = coords_centered[:, 0].max() - coords_centered[:, 0].min()
    y_range = coords_centered[:, 1].max() - coords_centered[:, 1].min()
    z_range = coords_centered[:, 2].max() - coords_centered[:, 2].min()

# Largest dimension defines box size
    box_size_angstrom = max(x_range, y_range, z_range) + 2 * padding

# Convert to pixels using desired pixel size
    img_size = int(np.ceil(box_size_angstrom / pixel_size))

    print(f"Pixel size: {pixel_size} Å/pixel")
    print(f"Box size: {box_size_angstrom:.2f} Å")
    print(f"Image size: {img_size} x {img_size} pixels")
 
    def create_projection(coord1, coord2, bins, box_size):

        half_box = box_size / 2

        # Centered coordinate system
        range1 = [-half_box, half_box]
        range2 = [-half_box, half_box]

        # 2D histogram
        H, xedges, yedges = np.histogram2d(
            coord1,
            coord2,
            bins=bins,
            range=[range1, range2]
        )

        # Normalize
        H = H / H.max()

        # Invert grayscale
        H = 1.0 - H

        return H

    # Create projections
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

    # ============================================================
    # SAVE PROJECTIONS AS MRC FILES
    # ============================================================

    print("\n" + "="*70)
    print("💾 SAVING PROJECTIONS AS MRC TEMPLATES")
    print("="*70)

    # Convert to float32 (standard for MRC)
    proj_xy_mrc = proj_xy.astype(np.float32)
    proj_xz_mrc = proj_xz.astype(np.float32)
    proj_yz_mrc = proj_yz.astype(np.float32)

# Save XY projection
    with mrcfile.new(f"{pdb_id}_XY_template.mrc", overwrite=True) as mrc:
        mrc.set_data(proj_xy_mrc)

    # Pixel size metadata
        mrc.voxel_size = pixel_size

    # Header updates
        mrc.header.origin = (0.0, 0.0, 0.0)

    print(f"✓ Saved: {pdb_id}_XY_template.mrc")

# Save XZ projection
    with mrcfile.new(f"{pdb_id}_XZ_template.mrc", overwrite=True) as mrc:
        mrc.set_data(proj_xz_mrc)
        mrc.voxel_size = pixel_size
        mrc.header.origin = (0.0, 0.0, 0.0)

    print(f"✓ Saved: {pdb_id}_XZ_template.mrc")

# Save YZ projection
    with mrcfile.new(f"{pdb_id}_YZ_template.mrc", overwrite=True) as mrc:
        mrc.set_data(proj_yz_mrc)
        mrc.voxel_size = pixel_size
        mrc.header.origin = (0.0, 0.0, 0.0)

    print(f"✓ Saved: {pdb_id}_YZ_template.mrc")

    print("\nMRC Template Information:")
    print(f"  - Pixel size: {pixel_size} Å/pixel")
    print(f"  - Image size: {img_size} x {img_size}")
    print("  - Data type: float32")
    print("  - Compatible with cryo-EM software:")
    print("      * RELION")
    print("      * cryoSPARC")
    print("      * EMAN2")
    print("      * IMOD")
    print("      * Warp")

    # Plot projections
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # XY projection
    axes[0].imshow(proj_xy.T, cmap='gray', origin='lower')
    axes[0].set_title('XY Projection (sum over Z)', fontweight='bold')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')

    # XZ projection
    axes[1].imshow(proj_xz.T, cmap='gray', origin='lower')
    axes[1].set_title('XZ Projection (sum over Y)', fontweight='bold')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Z')

    # YZ projection
    axes[2].imshow(proj_yz.T, cmap='gray', origin='lower')
    axes[2].set_title('YZ Projection (sum over X)', fontweight='bold')
    axes[2].set_xlabel('Y')
    axes[2].set_ylabel('Z')

    plt.tight_layout()
    plt.show()

    print("\nSummed Density Projections:")
    print("  - Pixel intensity = number of atoms projected into that pixel")
    print("  - Brighter = higher density")
    print("  - Approximates ideal projection images (no noise, no CTF)")
    print("Header:", mrcfile.header)
    # 3D visualization
    fig = plt.figure(figsize=(12, 5))
    
    # 3D scatter plot
    ax4 = fig.add_subplot(1, 2, 1, projection='3d')
    scatter = ax4.scatter(coords_centered[:, 0], coords_centered[:, 1], coords_centered[:, 2], 
                          c=np.linalg.norm(coords_centered, axis=1), cmap='viridis', s=20, alpha=0.6)
    ax4.set_xlabel('X (Å)', fontsize=10)
    ax4.set_ylabel('Y (Å)', fontsize=10)
    ax4.set_zlabel('Z (Å)', fontsize=10)
    ax4.set_title(f'{pdb_id} - 3D Structure', fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=ax4, label='Distance from origin (Å)')
    
    # 3D scatter plot - rotated view
    ax5 = fig.add_subplot(1, 2, 2, projection='3d')
    scatter2 = ax5.scatter(coords_centered[:, 0], coords_centered[:, 1], coords_centered[:, 2], 
                           c=np.linalg.norm(coords_centered, axis=1), cmap='viridis', s=20, alpha=0.6)
    ax5.view_init(elev=20, azim=45)  # Different viewing angle
    ax5.set_xlabel('X (Å)', fontsize=10)
    ax5.set_ylabel('Y (Å)', fontsize=10)
    ax5.set_zlabel('Z (Å)', fontsize=10)
    ax5.set_title(f'{pdb_id} - 3D Structure (Rotated)', fontsize=12, fontweight='bold')
    plt.colorbar(scatter2, ax=ax5, label='Distance from origin (Å)')
    plt.tight_layout()
    plt.show()


except urllib.error.URLError as e:
    print(f"❌ Error downloading PDB file: {e}")
    print("   Make sure you have internet connection")
except Exception as e:
    print(f"❌ Error: {type(e).__name__}: {e}")