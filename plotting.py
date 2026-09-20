"""Pure rendering functions: array/scalar in, matplotlib Figure out.

No Streamlit imports, no session_state access. Callers (app.py, scan.py) are
responsible for displaying the returned Figure (st.pyplot) and closing it
afterwards (plt.close(fig)) to avoid leaking figures across reruns/frames.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

import config

_BG = "#0b1220"
_FG = "#e8eef7"
_DASH = "#a6caec"
_RAY = "#4e95d9"


def _style_axes(ax):
    ax.set_facecolor(_BG)
    for spine in ax.spines.values():
        spine.set_color(_FG)
    ax.tick_params(colors=_FG)
    ax.xaxis.label.set_color(_FG)
    ax.yaxis.label.set_color(_FG)
    ax.title.set_color(_FG)


def _ray_endpoints(theta_deg, t_offset_cm, radius_cm):
    """(start, end) of a ray line at detector-offset t_offset, spanning
    +/- radius_cm along the source->detector direction. See ct_model module
    docstring for the (t, s) geometry convention this mirrors."""
    r = np.deg2rad(theta_deg)
    cos_t, sin_t = np.cos(r), np.sin(r)
    cx, cy = t_offset_cm * cos_t, t_offset_cm * sin_t
    start = (cx - radius_cm * sin_t, cy + radius_cm * cos_t)  # source side
    end = (cx + radius_cm * sin_t, cy - radius_cm * cos_t)  # detector side
    return start, end


def _draw_phantom_grid(ax, boundary):
    """Narrow yellow reference grid, GRID_SPACING_CM apart in x and y, clipped
    to the gantry circle so only the portion inside it is visible."""
    lim = config.CIRCLE_RADIUS_CM
    n = int(lim // config.GRID_SPACING_CM)
    offsets = np.arange(-n, n + 1) * config.GRID_SPACING_CM
    for off in offsets:
        v_line = ax.plot([off, off], [-lim, lim], color="yellow", linewidth=0.6, alpha=0.7, zorder=1)[0]
        v_line.set_clip_path(boundary)
        h_line = ax.plot([-lim, lim], [off, off], color="yellow", linewidth=0.6, alpha=0.7, zorder=1)[0]
        h_line.set_clip_path(boundary)


def render_image_panel(image_hu, theta_deg, n_rays=7, figsize=(5, 5)):
    """Circular phantom + rotating parallel-beam x-ray/detector overlay.

    Cropped to exactly the phantom's own extent (the 500x500 px / 50 cm FOV)
    -- no space plotted beyond the circle -- and each arrow's length is the
    chord of the circle at its own perpendicular offset, so every arrow
    terminates exactly on the circle boundary rather than overshooting into
    space that no longer exists. figsize defaults to 5x5in so that, at
    matplotlib's default 100 dpi, the rendered panel is a crisp 500x500 px
    image -- one pixel per FOV pixel -- meant to be displayed at that same
    500 px width rather than stretched to fill its column.
    """
    fig, ax = plt.subplots(figsize=figsize)
    # Pure black (not _BG) so the plot area outside the image extent seams
    # invisibly into the phantom's own -1000 HU (= black) corners/background.
    fig.patch.set_facecolor("black")
    _style_axes(ax)
    ax.set_facecolor("black")

    extent = [
        config.T_RANGE_CM[0],
        config.T_RANGE_CM[1],
        config.T_RANGE_CM[0],
        config.T_RANGE_CM[1],
    ]
    ax.imshow(
        image_hu,
        cmap="gray",
        vmin=config.HU_MIN,
        vmax=config.HU_MAX,
        origin="upper",
        extent=extent,
        zorder=0,
    )

    boundary = Circle(
        (0, 0), config.CIRCLE_RADIUS_CM, fill=False, linestyle="--", edgecolor=_FG, linewidth=1.5, zorder=2
    )
    ax.add_patch(boundary)
    _draw_phantom_grid(ax, boundary)

    for t_off in np.linspace(-0.8 * config.CIRCLE_RADIUS_CM, 0.8 * config.CIRCLE_RADIUS_CM, n_rays):
        # Chord half-length at this perpendicular offset -- the longest a ray
        # can be drawn at this offset while still ending exactly on the circle.
        chord_half_length = np.sqrt(config.CIRCLE_RADIUS_CM**2 - t_off**2)
        start, end = _ray_endpoints(theta_deg, t_off, chord_half_length)
        arrow = FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=10, color=_RAY, linewidth=1.0, alpha=0.85,
            zorder=3,
        )
        ax.add_patch(arrow)

    lim = config.CIRCLE_RADIUS_CM
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return fig


def render_projection_plot(p_t, figsize=(4.2, 2.4)):
    """p(t) line plot, y-axis capped at config.P_MAX (display-only clip)."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(_BG)
    _style_axes(ax)

    ax.plot(config.T_POSITIONS_CM, p_t, color=_RAY, linewidth=1.5)
    ax.set_xlim(config.T_RANGE_CM)
    ax.set_ylim(0, config.P_MAX)
    ax.set_xlabel("t (cm)")
    ax.set_ylabel("p(t)")
    fig.tight_layout()
    return fig


def render_sinogram(sinogram, theta_deg, figsize=(4.2, 3.0)):
    """Sinogram image (t vs theta) with a dashed vertical line marking the
    gantry angle.

    The sinogram only spans one half rotation (0-180 deg): a parallel-beam
    projection at theta and at theta+180 are identical but mirrored, i.e.
    the same physical measurement. theta_deg (from the 0-360 deg gantry
    slider) is therefore wrapped into [0, 180) before marking it, so the
    line still lands on the matching column when the gantry is past 180 deg
    instead of simply falling outside the plotted range.

    sinogram: (config.N_T, config.N_THETA) array or None (never scanned) or
    containing NaN columns (not-yet-revealed, mid-scan) -- NaN renders as
    the panel background so "unfilled" reads as empty rather than as a
    spurious zero-signal (black) reading.
    """
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(_BG)
    _style_axes(ax)

    if sinogram is None:
        sinogram = np.full((config.N_T, config.N_THETA), np.nan)

    cmap = plt.get_cmap("gray").copy()
    cmap.set_bad(_BG)

    extent = [0, config.N_THETA * config.THETA_STEP_DEG, config.T_RANGE_CM[0], config.T_RANGE_CM[1]]
    ax.imshow(
        sinogram,
        cmap=cmap,
        vmin=0,
        vmax=config.P_MAX,
        origin="lower",
        extent=extent,
        aspect="auto",
    )

    ax.axvline(theta_deg % 180.0, color=_DASH, linewidth=2, linestyle=(0, (6, 4)))
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel(r"$\vartheta$ (deg)")
    ax.set_ylabel("t (cm)")
    fig.tight_layout()
    return fig


def render_hu_colorbar(brush_hu, figsize=(6, 1.2)):
    """Decorative HU gradient bar with a marker at the selected brush_hu.

    Purely visual (the actual selection widget is a native st.slider in
    drawing.py); this mirrors the mockup's gradient bar above/below it.

    The axes is placed by hand to span the *entire* figure width (rect
    left=0, width=1) rather than via tight_layout, so the rendered gradient
    reaches both edges of the image exactly -- lining it up with the native
    slider's full-width track above it once Streamlit stretches both to the
    same column width. tight_layout's own small default padding, or any
    left-side margin left in place for tick labels, would otherwise make the
    bar a few pixels narrower than the slider.
    """
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor(_BG)
    ax = fig.add_axes([0.0, 0.42, 1.0, 0.56])
    ax.set_facecolor(_BG)
    for spine in ax.spines.values():
        spine.set_visible(False)

    gradient = np.linspace(config.HU_MIN, config.HU_MAX, 512).reshape(1, -1)
    ax.imshow(
        gradient,
        cmap="gray",
        vmin=config.HU_MIN,
        vmax=config.HU_MAX,
        extent=[config.HU_MIN, config.HU_MAX, 0, 1],
        aspect="auto",
    )
    ax.axvline(brush_hu, color=_DASH, linewidth=2.5)
    ax.set_xlim(config.HU_MIN, config.HU_MAX)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("CT number (HU)", color=_FG, fontsize=17, labelpad=6)
    return fig
