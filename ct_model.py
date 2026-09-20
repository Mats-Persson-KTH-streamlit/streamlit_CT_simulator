"""Pure numpy/skimage physics: no Streamlit imports, no session state.

Geometry convention (verified empirically against skimage.transform.radon):
image x = columns (right positive), y = rows measured upward (i.e. row 0 is
the top, so image-space y = -row-offset). This is a standard math frame, and
skimage's radon gives, for angle theta (deg):

    t = x * cos(theta) + y * sin(theta)

which is the textbook Radon-transform parameterization. theta=0 therefore
integrates along vertical rays (t = x), matching "theta=0 -> vertical
projection rays" in the spec. The detector/source rotate counter-clockwise
as theta increases; at theta=0 the detector sits below the image (negative
y), matching "detector below the image" in the spec.
"""

import numpy as np
from skimage.transform import radon

import config

# Pixel-center coordinate grid used for axis labels (image x/y and detector t
# all share this 500-sample array over the same physical span).
_COORDS_CM = config.T_POSITIONS_CM

# CIRCLE_MASK intentionally does NOT use the continuous _COORDS_CM grid above.
# skimage.transform.radon's rotation axis for an even-sized image is pinned to
# the integer pixel index shape//2 (250), half a pixel off the symmetric
# -24.95..24.95 center implied by _COORDS_CM. Matching skimage's own integer
# convention here (rather than our continuous one) keeps the phantom's
# zero-outside-circle region exactly consistent with what radon() itself
# assumes, avoiding boundary artifacts; the mismatch is 0.5 mm on a 250 mm
# radius (0.2%) and irrelevant to the axis labels used elsewhere.
_center_idx = config.IMAGE_SIZE_PX // 2
_row_idx, _col_idx = np.ogrid[: config.IMAGE_SIZE_PX, : config.IMAGE_SIZE_PX]
_dist2_px = (_row_idx - _center_idx) ** 2 + (_col_idx - _center_idx) ** 2

CIRCLE_MASK = _dist2_px <= config.CIRCLE_RADIUS_PX**2  # True inside the phantom circle


def blank_image_hu():
    """A fresh, all-air phantom: -1000 HU everywhere."""
    return np.full((config.IMAGE_SIZE_PX, config.IMAGE_SIZE_PX), config.HU_MIN, dtype=float)


def enforce_constraints(image_hu):
    """Clip in-circle values to [HU_MIN, HU_MAX] and force outside-circle to HU_MIN.

    Returns a new array; does not mutate the input.
    """
    out = np.clip(image_hu, config.HU_MIN, config.HU_MAX)
    out = np.where(CIRCLE_MASK, out, config.HU_MIN)
    return out


def project_single_angle(image_hu, theta_deg):
    """Line integral p(t) at one gantry angle.

    Returns an array of length config.N_T, dimensionless, aligned with
    config.T_POSITIONS_CM (index i <-> T_POSITIONS_CM[i]).
    """
    mu = config.hu_to_mu(image_hu)
    sino = radon(mu, theta=[theta_deg], circle=True)
    return sino[:, 0] * config.PIXEL_SIZE_CM


def full_sinogram(image_hu):
    """Full sinogram over all spec angles.

    Returns an array of shape (config.N_T, config.N_THETA), dimensionless,
    column j <-> config.THETA_DEG[j], row i <-> config.T_POSITIONS_CM[i].
    """
    mu = config.hu_to_mu(image_hu)
    sino = radon(mu, theta=config.THETA_DEG, circle=True)
    return sino * config.PIXEL_SIZE_CM


def detector_position_cm(theta_deg, radius_cm=config.CIRCLE_RADIUS_CM):
    """(x, y) of the detector-line center at the given gantry angle."""
    r = np.deg2rad(theta_deg)
    return radius_cm * np.sin(r), -radius_cm * np.cos(r)


def source_position_cm(theta_deg, radius_cm=config.CIRCLE_RADIUS_CM):
    """(x, y) of the x-ray source, diametrically opposite the detector."""
    x, y = detector_position_cm(theta_deg, radius_cm)
    return -x, -y
