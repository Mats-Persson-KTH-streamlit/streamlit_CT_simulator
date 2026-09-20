"""Central configuration: all constants from Specification.txt live here.

Every other module imports what it needs from this file instead of
hardcoding numbers, so the physical/geometric model is defined exactly once.
"""

import numpy as np

# --- Phantom / image geometry ---------------------------------------------
IMAGE_SIZE_PX = 500                                     # square array side, pixels
FOV_DIAMETER_CM = 50.0                                   # field of view diameter
PIXEL_SIZE_CM = FOV_DIAMETER_CM / IMAGE_SIZE_PX          # 0.1 cm = 1 mm

# --- HU / attenuation model --------------------------------------------------
HU_MIN = -1000
HU_MAX = 1000
MU_WATER = 0.2  # cm^-1, linear attenuation coefficient of water (at 0 HU)


def hu_to_mu(hu):
    """Convert CT number(s) in Hounsfield Units to linear attenuation (cm^-1)."""
    return MU_WATER * (hu + 1000) / 1000


# --- Sinogram / detector geometry -------------------------------------------
T_RANGE_CM = (-25.0, 25.0)   # detector extent
N_T = 500                    # number of detector positions
N_THETA = 900                # number of angles spanning one half rotation
THETA_STEP_DEG = 180.0 / N_THETA  # 0.2 deg

# Bin-center grids, derived once at import time.
T_POSITIONS_CM = np.linspace(
    T_RANGE_CM[0] + PIXEL_SIZE_CM / 2,
    T_RANGE_CM[1] - PIXEL_SIZE_CM / 2,
    N_T,
)  # -24.95 .. 24.95 cm, step 0.1 cm
THETA_DEG = np.arange(N_THETA) * THETA_STEP_DEG  # 0.0 .. 179.8 deg

# --- Phantom circle -----------------------------------------------------------
# Spec text says both "circular image with diameter 500 pixels" and, later,
# "outside the 500-pixel-radius circle" -- a radius of 500 px cannot fit inside
# a 500x500 image, so the first (diameter=500 px) is the one taken as correct:
# the circle is inscribed in the square, radius = half the image side.
CIRCLE_RADIUS_PX = IMAGE_SIZE_PX / 2  # 250 px
CIRCLE_RADIUS_CM = FOV_DIAMETER_CM / 2  # 25 cm

# --- Display / physical limits -----------------------------------------------
P_MAX = 10.0  # projection plot y-axis / sinogram color-scale cap (visual only)

# --- Scan animation -----------------------------------------------------------
T_SCAN_SECONDS = 5.0

# --- UI defaults (not dictated by the spec, but centralized for the same
# reason as everything else above: one place to tune, not scattered literals).
DEFAULT_BRUSH_HU = 0  # water; int to match HU_MIN/HU_MAX/the slider's step (all int)
BRUSH_SIZES_PX = (3, 8, 16, 30)  # brush radii offered in the brush panel
DEFAULT_BRUSH_SIZE_PX = BRUSH_SIZES_PX[1]

# Gantry rotation is set with a slider (0-360 deg) rather than a drag gesture,
# so the drawing canvas is exactly the phantom's own resolution -- no extra
# margin needed around it.
CANVAS_SIZE_PX = IMAGE_SIZE_PX

# Reference grid drawn inside the gantry circle, in the phantom-reproduction
# panel (not the interactive canvas, so it never contaminates image_hu).
GRID_SPACING_CM = 10.0
