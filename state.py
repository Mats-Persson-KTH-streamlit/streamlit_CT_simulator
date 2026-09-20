"""Single source of truth for st.session_state keys and their defaults.

Every other module reads/writes session_state only through the key names
defined here (as plain string literals matching these), and app.py is the
only place that calls init_state(). No other module should invent new
session_state keys ad hoc.
"""

import streamlit as st

import config
import ct_model

IMAGE_HU = "image_hu"
BRUSH_HU = "brush_hu"
BRUSH_SIZE_PX = "brush_size_px"
THETA_DEG = "theta_deg"
SINOGRAM = "sinogram"  # None until the first completed scan


def init_state():
    """Populate any missing session_state keys with their defaults. Idempotent."""
    if IMAGE_HU not in st.session_state:
        st.session_state[IMAGE_HU] = ct_model.blank_image_hu()
    if BRUSH_HU not in st.session_state:
        st.session_state[BRUSH_HU] = config.DEFAULT_BRUSH_HU
    if BRUSH_SIZE_PX not in st.session_state:
        st.session_state[BRUSH_SIZE_PX] = config.DEFAULT_BRUSH_SIZE_PX
    if THETA_DEG not in st.session_state:
        st.session_state[THETA_DEG] = 0.0
    if SINOGRAM not in st.session_state:
        st.session_state[SINOGRAM] = None
