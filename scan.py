"""The "Scan" button: compute the full sinogram once, then animate a 5s
progressive left-to-right reveal, per the spec:

  "the detector position moves back to 0 degrees, then scans from 0 to 180
  degrees during T_scan=5 seconds, and then moves back to its original
  position [...] the scan stops at theta=180 [...] then the dashed line
  moves back to the original position theta."

Computing the whole sinogram up front (one radon() call) rather than
per-frame keeps every animation frame fast -- only the *reveal* is animated,
not the underlying transform. The upfront computation itself is treated as a
brief pre-scan pause, not counted against T_SCAN_SECONDS; the loop below
paces only the visible sweep to that duration.

A fixed step count tuned against local, headless render timing turned out
to badly overshoot T_SCAN_SECONDS in a real browser session -- Streamlit's
per-call serialization/websocket overhead dominates and varies a lot by
machine, and a fixed count can't adapt to that. Instead, each iteration
recomputes how many columns *should* be revealed from elapsed wall-clock
time (col_end = fraction-of-T_SCAN-elapsed * N_THETA): fast hardware yields
many small steps, slow hardware yields fewer/chunkier ones, and either way
the loop naturally reaches full reveal once elapsed >= T_SCAN_SECONDS -- no
per-machine tuning needed. The rotating image/projection panels refresh on
their own coarser wall-clock interval rather than every sinogram step, since
they cost roughly 2x a sinogram-only frame.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import config
import ct_model
import plotting
import state

IMAGE_REFRESH_INTERVAL_S = 0.1  # how often the image/projection panels refresh


def run_scan(placeholders):
    """placeholders: dict with 'image', 'projection', 'sinogram' -> st.empty()."""
    image_hu = st.session_state[state.IMAGE_HU]
    theta_before = st.session_state[state.THETA_DEG]

    full_sino = ct_model.full_sinogram(image_hu)

    loop_start = time.time()
    last_image_update = 0.0  # force an image/projection refresh on the first iteration
    col_end = 0
    while col_end < config.N_THETA:
        elapsed = time.time() - loop_start
        frac = min(1.0, elapsed / config.T_SCAN_SECONDS)
        col_end = max(col_end + 1, min(config.N_THETA, int(round(frac * config.N_THETA))))
        theta_now = config.THETA_DEG[col_end - 1]

        partial_sino = full_sino.copy()
        partial_sino[:, col_end:] = np.nan
        _render_sinogram_frame(placeholders, theta_now, partial_sino)

        now = time.time()
        if col_end == config.N_THETA or now - last_image_update >= IMAGE_REFRESH_INTERVAL_S:
            _render_image_and_projection(placeholders, image_hu, theta_now, full_sino[:, col_end - 1])
            last_image_update = now

    # Detector returns to its original (pre-scan) angle; the fully-revealed
    # sinogram stays, but the dashed line and image/projection panels go back.
    # theta_before is left untouched in session_state: it's the angle slider's
    # own widget key, and it was never changed by this loop (only rendered
    # from a local variable), so it's already correct.
    st.session_state[state.SINOGRAM] = full_sino
    _render_sinogram_frame(placeholders, theta_before, full_sino)
    _render_image_and_projection(
        placeholders, image_hu, theta_before, ct_model.project_single_angle(image_hu, theta_before)
    )


def _render_sinogram_frame(placeholders, theta_deg, sinogram):
    fig_s = plotting.render_sinogram(sinogram, theta_deg)
    placeholders["sinogram"].pyplot(fig_s, clear_figure=True, width='stretch')
    plt.close(fig_s)


def _render_image_and_projection(placeholders, image_hu, theta_deg, p_t):
    fig_img = plotting.render_image_panel(image_hu, theta_deg)
    placeholders["image"].pyplot(fig_img, clear_figure=True, width=config.IMAGE_SIZE_PX)
    plt.close(fig_img)

    fig_p = plotting.render_projection_plot(p_t)
    placeholders["projection"].pyplot(fig_p, clear_figure=True, width='stretch')
    plt.close(fig_p)
