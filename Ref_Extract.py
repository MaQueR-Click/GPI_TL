import mrcfile
import numpy as np
import matplotlib.pyplot as plt

# =========================
# LOAD MRC IMAGE
# =========================

mrc_file = "padded_200_binning_4_14sep05c_c_00003gr_00014sq_00005hl_00003es_c.mrc"   # <- change path here

with mrcfile.open(mrc_file, permissive=True) as mrc:
    image = mrc.data.astype(np.float32)

# If 3D stack -> take first slice
if image.ndim == 3:
    image = image[0]

print("Shape :", image.shape)
print("dtype :", image.dtype)

# =========================
# DISPLAY IMAGE
# =========================

fig, ax = plt.subplots(figsize=(8, 8))

im = ax.imshow(
    image,
    cmap='gray',
    origin='upper',
    vmin=np.percentile(image, 1),
    vmax=np.percentile(image, 99)
)

plt.colorbar(im, ax=ax, label='Intensity')

ax.set_title("Click on image")

# =========================
# CLICK EVENT FUNCTION
# =========================

def on_click(event):

    # Check mouse inside axes
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

        # Pixel intensity
        intensity = image[y, x]

        # Print in terminal
        print(
            f"X={x}, Y={y}, Intensity={intensity}"
        )

        # Update title
        ax.set_title(
            f"X={x} | Y={y} | Intensity={intensity:.3f}"
        )

        # Refresh figure
        fig.canvas.draw_idle()

# =========================
# CONNECT EVENT
# =========================

fig.canvas.mpl_connect(
    'button_press_event',
    on_click
)

# =========================
# SHOW
# =========================

plt.show()