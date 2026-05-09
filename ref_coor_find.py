import mrcfile
import numpy as np
import matplotlib.pyplot as plt

# =========================
# LOAD MRC IMAGE
# =========================

mrc_file = "padded_200_binning_4_14sep05c_c_00003gr_00014sq_00005hl_00003es_c.mrc"

with mrcfile.open(mrc_file, permissive=True) as mrc:
    image = mrc.data.astype(np.float32)

# If stack -> first slice
if image.ndim == 3:
    image = image[0]

print("Shape :", image.shape)
print("dtype :", image.dtype)

# =========================
# DISPLAY IMAGE
# =========================

fig, ax = plt.subplots(figsize=(8,8))

im = ax.imshow(
    image,
    cmap='gray',
    origin='upper'
)

plt.colorbar(im, ax=ax, label='Intensity')

# =========================
# MOUSE EVENT
# =========================

def mouse_move(event):

    if event.inaxes != ax:
        return

    # Mouse coordinates
    x = int(event.xdata)
    y = int(event.ydata)

    # Check boundaries
    if (
        x >= 0 and x < image.shape[1]
        and
        y >= 0 and y < image.shape[0]
    ):

        intensity = image[y, x]

        # Update title
        ax.set_title(
            f"X={x}   Y={y}   Intensity={intensity:.3f}"
        )

        fig.canvas.draw_idle()

# Connect event
fig.canvas.mpl_connect(
    'motion_notify_event',
    mouse_move
)

plt.show()