"""particle extraction from micrographs using reference-based template matching"""
import os
import numpy as np
import mrcfile
import matplotlib.pyplot as plt

def coordinate_extraction (image, box_size, coordinate_reference):
    np.array(image)== image.astype(np.float32)
    extracted_particles= []
    for coord in coordinate_reference:
        x, y= coord
        x_start= int(x - box_size/2)
        x_end= int(x + box_size/2)
        y_start= int(y - box_size/2)
        y_end= int(y + box_size/2)
        particle= image[y_start:y_end, x_start:x_end]
        extracted_particles.append(particle)
    return extracted_particles

input_file = "/home/tlaborde/GPI_proj/padded_200_binning_4_14sep05c_c_00003gr_00014sq_00005hl_00003es_c.mrc"
image =input_file

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
print(image_array.shape)

box_size = 128
coordinate_reference = [(500, 500)]
print(coordinate_reference)

extracted_particles = coordinate_extraction(image_array, box_size, coordinate_reference)
plt.imshow(
    extracted_particles[0],
    cmap='gray')
min_val = np.percentile(extracted_particles[0], 1)
max_val = np.percentile(extracted_particles[0], 99)
plt.title("Extracted Particle")
plt.axis("off")
plt.show()
